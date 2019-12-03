# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from keras.layers import Dense, BatchNormalization, Input, Dropout
from keras.layers import Activation, concatenate, GRU, Dropout
import keras.backend as K
from concurrent import futures
import logging
import grpc
import time
import numpy as np

#import model_class_tpu as mc_tpu
import model_class as mc
import server_tools_pb2
import server_tools_pb2_grpc
import pandas as pd
import tensorflow as tf
from keras.models import Model

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
PORT = '50051'

global processes, max_id, results, max_client_ids
global max_client_id, new_client_permitted, times
processes = {}
max_client_ids = {}
max_client_id = 0
max_id = 0
new_client_permitted = True
results = {}
times = {}


class Model4ExpLLLow(mc.ClassModel):
    def get_outputs(self):
        self.name = '4layersExpLLLow'
        RATE = 0.001
        self.epochs = 11
        h = self.inputs
        h = Dropout(rate=RATE)(h)
        h = Dense(36, activation='relu')(h)
        norm = Dropout(rate=RATE)(h)
        h = Dense(11, activation='relu')(norm)
        norm = Dropout(rate=RATE)(h)
        h = Dense(3, activation='relu')(norm)
        return Dense(1, activation='linear', name='output')(h)

class ModelFacile(object):
    def __call__(self, inputs):
        net = tf.layers.dropout(inputs, rate = 0.001)
        net = tf.layers.dense(net, 36, activation = 'relu')
        norm = tf.layers.dropout(net, rate = 0.001)
        net = tf.layers.dense(norm, 11, activation = 'relu')
        norm = tf.layers.dropout(net, rate = 0.001)
        net = tf.layers.dense(norm, 3, activation = 'relu')
        return tf.layers.dense(net, 1, activation = 'linear', name='output')
    
def verify_request(request):
    logging.info("Client id is " + str(request.client_id))
    logging.info("Batch size is " + str(request.batch_size))
    return (request.client_id in max_client_ids) and request.batch_size > 0

def model_fn(features, labels, mode, params):
    model = ModelFacile()

    if mode == tf.estimator.ModeKeys.PREDICT:
        predicted_values = model(features)
        predictions = {
            'probabilities': predicted_values,
        }
        return tf.contrib.tpu.TPUEstimatorSpec(mode, predictions=predictions)

class MnistServer(server_tools_pb2_grpc.MnistServerServicer):
    estimator = None
    MODEL_DIR = 'gs://harrisgroup-ctpu/facile/model/'
    DATA_DIR = 'gs://harrisgroup-ctpu/facile/data/'
    TPU_NAME='uw-tpu'
    ZONE_NAME='us-central1-b'
    PROJECT_NAME = 'harrisgroup-223921'
    NUM_ITERATIONS = 1000 # Number of iterations per TPU training loop
    TRAIN_STEPS = 5000
    NUM_SHARDS = 8 # Number of shards (TPU chips).
    BATCH_SIZE = 32
    def __init__(self):
        print("Creating the estimator")
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            self.TPU_NAME,
            zone=self.ZONE_NAME,
            project=self.PROJECT_NAME)

        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=self.MODEL_DIR,
            session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True),
            tpu_config=tf.contrib.tpu.TPUConfig(self.NUM_ITERATIONS, self.NUM_SHARDS))
    
        self.estimator = tf.contrib.tpu.TPUEstimator(
            model_fn=model_fn,
            use_tpu=True,
            train_batch_size=self.BATCH_SIZE,
            eval_batch_size=self.BATCH_SIZE,
            predict_batch_size=self.BATCH_SIZE,
            params={"data_dir": self.DATA_DIR},
            config=run_config,
            warm_start_from = tf.compat.v1.estimator.WarmStartSettings(ckpt_to_initialize_from=
                                                                   "gs://harrisgroup-ctpu/facile/model/",
                                                                  vars_to_warm_start=".*"))
    
    def StartJobWait(self, request, context):
        logging.info("StartJobWait begins")
        if not verify_request(request):
            logging.error(
                "Request is invalid and failed the verification processes")
            return server_tools_pb2.PredictionMessage(
                complete=False, prediction=b'',
                error='Invalid data package', infer_time=0)
        logging.info(
            """Request is valid and data preparation
               succeeds, ml prediction begins""")
        """
        Initializing and standardizing X
        """
        sample = mc.Sample()
        sample.X = pd.read_json(request.x_input.decode('utf-8')) # OK!
        sample.X.drop(['PU', 'pt'], 1, inplace=True)
        sample.idx = np.random.permutation(sample.X.shape[0])
        print('Standardizing...')
        mu, std = np.mean(sample.X, axis=0), np.std(sample.X, axis=0)
        sample.standardize(mu, std)
        n_inputs = sample.X.shape[1]

        # Define class model
        # model = Model4ExpLLLow(n_inputs, True) # True for GPU and false for CPU
        # model.load_model("weights.h5")

        #sample.infer(model)
        #predictions, infer_time = model.predict(sample.X, 32)
        start_time = time.time()
        #preds = self.estimator.predict(sample.X, self.BATCH_SIZE)
        #predicted_values = next(prediction)
        #for pred_dict in prediction:
        #    predicted_values.append(list(pred_dict.values())[0][0])  
        def predict_input_fn(params):
            batch_size = self.BATCH_SIZE
            print("DATASET TYPE: ", type(sample.X.astype('float32').values))
            print("DATASET: ", sample.X.astype('float32').values)
            dataset_predict = tf.data.Dataset.from_tensor_slices(sample.X.astype('float32').values)
            return dataset_predict.batch(batch_size)
    
        preds = self.estimator.predict(predict_input_fn)
        predictionsarr = []
        for prediction in preds:
            pred = prediction['probabilities']
            predictionsarr.append(pred)
        predictions = np.asmatrix(predictionsarr)
        infer_time = time.time() - start_time

        # Need this otherwise two workers will be conflicted
        K.clear_session()
        # print("------------PREDICTION-----------")
        #print(preds)
        #print(predictions)
        #print(type(predictions))
        # print(predictions.tobytes())
        # print(predictions.shape)
        # print(predictions[:,0].shape)
        # print(type(predictions[0]))
        # print(predictions[:,0])
        # print(predictions.dtype)
        #print(predictions.type)
        #print(list(predictions))
        #print("Type of prediction is ", type(predictions))
        #print("What can I call ", dir(predictions))
        #print("Prediction shape ", predictions.shape())
        #prediction = np.array(predictions.values())
        #print(prediction)
        return server_tools_pb2.PredictionMessage(
            complete=True,
            prediction=(predictions[:,0].tobytes()),
            error='',
            infer_time=infer_time)

    def RequestClientID(self, request, context):
        global max_client_id, new_client_permitted, max_client_ids
        while not new_client_permitted:
            pass

        new_client_permitted = False
        client_id = str(max_client_id)
        max_client_id += 1
        new_client_permitted = True

        max_client_ids[client_id] = 0
        return server_tools_pb2.IDMessage(new_id=client_id, error='')


def serve():
    options = [('grpc.max_receive_message_length', 2047*1024*1024 )]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),options = options)
    server_tools_pb2_grpc.add_MnistServerServicer_to_server(
        MnistServer(), server)
   
    server.add_insecure_port('[::]:' + PORT)
    server.start()
    print("READY")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    logging.basicConfig()
    logging.root.setLevel(logging.NOTSET)
    logging.basicConfig(level=logging.NOTSET)
    serve()
