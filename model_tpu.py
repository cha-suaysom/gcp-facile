
import os, sys, time, threading
import tensorflow as tf
import numpy as np
import pandas as pd
# Disable depreciation warnings and limit verbosity during training
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


MODEL_DIR = 'gs://harrisgroup-ctpu/facile/model/'
DATA_DIR = 'gs://harrisgroup-ctpu/facile/data/'
TPU_NAME='jtdinsmo-tpu-2'
ZONE_NAME='us-central1-b'
PROJECT_NAME = 'harrisgroup-223921'
NUM_ITERATIONS = 50 # Number of iterations per TPU training loop
TRAIN_STEPS = 100000
EVALUATE_STEPS = 1000
INFERENCE_TIME_THRESHOLD = 10 # Seconds
NUM_SHARDS = 8 # Number of shards (TPU chips).
USE_TPU = True

lock = threading.Lock()

#tf.enable_eager_execution()

# DEFINE THE NETWORK

class ModelFacile(object):
    def __call__(self, inputs):
        net = tf.layers.dropout(inputs, rate = 0.001)
        net = tf.layers.dense(net, 36, activation = 'relu')
        norm = tf.layers.dropout(net, rate = 0.001)
        net = tf.layers.dense(norm, 11, activation = 'relu')
        norm = tf.layers.dropout(net, rate = 0.001)
        net = tf.layers.dense(norm, 3, activation = 'relu')
        return tf.layers.dense(net, 1, activation = 'linear', name='output')


def input_facile(train_x,train_y,batch_size):
     dataset = tf.data.Dataset.from_tensor_slices((tf.cast(train_x,tf.float32),tf.cast(train_y,tf.float32)))
     return dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

def model_fn(features, labels, mode, params):
    model = ModelFacile()

    if mode == tf.estimator.ModeKeys.PREDICT:
        predicted_values = model(features)
        #print("Shape of predicted values is ", tf.shape(predicted_values))
        predictions = {
            'probabilities': predicted_values,
        }
        return tf.contrib.tpu.TPUEstimatorSpec(mode, predictions=predictions)

    predicted_values = model(features)
    loss = tf.losses.mean_squared_error(labels=labels,predictions=predicted_values)

    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
      if USE_TPU:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
      train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
      return tf.contrib.tpu.TPUEstimatorSpec(mode, loss=loss, train_op=train_op)


# CREATE AND PREDICT WITH TPUS

def create_estimator(batch_size):
    print("Creating the estimator")
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        TPU_NAME,
        zone=ZONE_NAME,
        project=PROJECT_NAME)

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=MODEL_DIR,
        session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True),
        tpu_config=tf.contrib.tpu.TPUConfig(NUM_ITERATIONS, NUM_SHARDS))

    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        use_tpu=USE_TPU,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        predict_batch_size=batch_size,
        params={"data_dir": DATA_DIR},
        config=run_config)
    return estimator

def predict(data, batch_size,estimator, results=None, times=None, job_id=None):
    assert (results is None and times is None and job_id is None) or not (results is None or times is None or job_id is None)
    lock.acquire()
    start_time = time.time()

    print("Predicting")
    def predict_input_fn(params):
        batch_size = params["batch_size"]
        dataset_predict = tf.data.Dataset.from_tensor_slices(data.astype('float32'))
        return dataset_predict.batch(batch_size)

    predictions = estimator.predict(predict_input_fn)
    i = 0
    predicted_values = []
    for pred_dict in predictions:
        print(pred_dict.keys())
        print(pred_dict.values())
        break
    #    predicted_values.append(list(pred_dict.values())[0][0])
    #predicted_values =np.array(predicted_values)/np.sum(predicted_values)
    #print("Number of prediction is ",i)
    #print(predicted_values[:10])

    predict_time = time.time() - start_time
    lock.release()

    return predictions, predict_time

if __name__=="__main__":
     train_x = pd.read_pickle("input/X_HB.pkl")[:]
     train_x.drop(['PU','pt'],1,inplace=True)
     #train_y = pd.read_pickle("input/Y_HB.pkl")[:][['genE']]
     estimator = create_estimator(32)

     #estimator.train(
     # input_fn=lambda params: input_facile(train_x,train_y,
     #     params["batch_size"]),
     # max_steps=TRAIN_STEPS)
     for i in range(10):
       start_time = time.time()
       predict(train_x[:1000],32,estimator)
       predict_time = time.time() - start_time
       print("Time used in the prediction is ", predict_time)
