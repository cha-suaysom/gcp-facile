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
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from keras.layers import Dense, BatchNormalization, Input, Dropout
from keras.layers import Activation, concatenate, GRU, Dropout
from keras.models import load_model
import keras.backend as K
from concurrent import futures
import logging
import grpc
import time
import numpy as np
from tensorflow.python.client import device_lib

import model_class as mc
import server_tools_pb2
import server_tools_pb2_grpc
import pandas as pd
from keras.models import Model
import tensorflow as tf

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
PORT = '50051'
tf.logging.set_verbosity(tf.logging.ERROR)

global processes, max_id, results, max_client_ids
global max_client_id, new_client_permitted, times, weight_file
processes = {}
max_client_ids = {}
max_client_id = 0
max_id = 0
new_client_permitted = True
results = {}
times = {}

def verify_request(request):
    logging.info("Client id is " + str(request.client_id))
    logging.info("Batch size is " + str(request.batch_size))
    return (request.client_id in max_client_ids) and request.batch_size > 0


class FacileServer(server_tools_pb2_grpc.FacileServerServicer):
    
#     weight_file = None;
#     def __init__(self):
#         start_time =time.time()
#         self.weight_file = load_model("weights.h5", compile=False)
#         self.weight_file._make_predict_function()
#         input_tensor_shape = self.weight_file.input_shape
#         init_tensor_shape = [1]
#         for i in range(len(input_tensor_shape)-1):
#             init_tensor_shape.append(input_tensor_shape[i+1])
#         print(init_tensor_shape)
#         init_tensor = np.zeros(init_tensor_shape)
#         self.weight_file.predict(init_tensor, 1)
#         finish_time = time.time() - start_time
#         print("Loading model time ", finish_time)

    def StartJobWait(self, request, context):
        whole_time = 0
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

        
        #Initializing and standardizing X
        
        start_time =time.time()
        #X = pd.read_json(request.data.decode('utf-8')) # OK!
        X = np.frombuffer(request.data)
        X.reshape((request.num_data, int(len(X)/request.num_data)))
        finish_time = time.time()-start_time
        print("Time spent decoding ", finish_time)
        batch_size = request.batch_size

        logging.info("List all devices")
        for tf_device in device_lib.list_local_devices():
           logging.info(tf_device)

        logging.info("List all available GPU OUTSIDE with Keras")
        for gpu_machine in K.tensorflow_backend._get_available_gpus():
            logging.info(gpu_machine)
            
        weight_file = load_model("weights.h5", compile=False)

        with tf.device('/cpu:0'):
            start_time = time.time()
            #predictions = self.weight_file.predict(X, batch_size)
            predictions = weight_file.predict(X, batch_size)
            infer_time_CPU = time.time()-start_time
            logging.info("Infer time is "+ str(infer_time_CPU))
            whole_time += infer_time_CPU
            logging.info("------------------------")

        with tf.device('/gpu:0'):
            start_time = time.time()
            #predictions = self.weight_file.predict(X, batch_size)
            predictions = weight_file.predict(X, batch_size)
            infer_time_GPU = time.time()-start_time
            logging.info("Infer time is "+ str(infer_time_GPU))
            whole_time += infer_time_GPU
            logging.info("------------------------")
        # Need this otherwise two workers will be conflicted
        K.clear_session()
        print("Whole time is ", whole_time)
        return server_tools_pb2.PredictionMessage(
            complete=True,
            prediction=predictions[:,0].tobytes(),
            error='',
            infer_time_CPU = infer_time_CPU,
            infer_time_GPU = infer_time_GPU)

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
    server_tools_pb2_grpc.add_FacileServerServicer_to_server(
        FacileServer(), server)
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
