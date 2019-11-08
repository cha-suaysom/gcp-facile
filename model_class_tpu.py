
import os, sys, time, threading
import tensorflow as tf
import numpy as np
import pandas as pd
# Disable depreciation warnings and limit verbosity during training
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

#tf.logging.set_verbosity(0)



# IMG_EDGE = 28
MODEL_DIR = 'gs://harrisgroup-ctpu/facile/model/'
DATA_DIR = 'gs://harrisgroup-ctpu/facile/data/'
#MODEL_DIR = "/home/nsuaysom/gcp-facile"
#DATA_DIR = "/home/nsuaysom/gcp-facile/input/"
TPU_NAME='jtdinsmo-tpu-2'
ZONE_NAME='us-central1-b'
PROJECT_NAME = 'harrisgroup-223921'
NUM_ITERATIONS = 50 # Number of iterations per TPU training loop
TRAIN_STEPS = 10
EVALUATE_STEPS = 1000
INFERENCE_TIME_THRESHOLD = 10 # Seconds
NUM_SHARDS = 8 # Number of shards (TPU chips).
USE_TPU = True

lock = threading.Lock()

# DEFINE THE NETWORK

class ModelCNN(object):
    def __call__(self, inputs):
        #norm = tf.layers.dropout(rate =0.001)(input)
        #net = tf.layers.dense(11,activtion = 'relu')(norm)
        #net = tf.layers.conv2d(inputs, 32, [5, 5], activation=tf.nn.relu, name='conv1')
        #net = tf.layers.max_pooling2d(net, [2, 2], 2, name='pool1')
        #net = tf.layers.conv2d(net, 64, [5, 5], activation=tf.nn.relu, name='conv2')
        #net = tf.layers.max_pooling2d(net, [2, 2], 2, name='pool2')
        #net = tf.layers.flatten(net, name='flat')
        #net = tf.layers.dense(net, NUM_CLASSES, activation=None, name='fc1')
        return tf.layers.dense(inputs, 36, activation = 'linear', name='output')

# class Model4ExpLLLow(mc.ClassModel):
#     def get_outputs(self):
#         self.name = '4layersExpLLLow'
#         RATE = 0.001
#         self.epochs = 11
#         h = self.inputs
#         h = Dropout(rate=RATE)(h)
#         h = Dense(36, activation='relu')(h)
#         norm = Dropout(rate=RATE)(h)
#         h = Dense(11, activation='relu')(norm)
#         norm = Dropout(rate=RATE)(h)
#         h = Dense(3, activation='relu')(norm)
#         return Dense(1, activation='linear', name='output')(h)

# DEFINE THE INPUT FUNCTIONS

def input_facile(train_x,train_y,batch_size):
     #X = pd.read_pickle("input/X_HB.pkl")[:1000]
     #X.drop(['PU','pt'],1,inplace=True)
     #Y = pd.read_pickle("input/Y_HB.pkl")[:1000]
     train_y = train_y*0+1
     train_y.iloc[0] = 2
     train_y.iloc[1] = 2
     print(train_y)
     dataset = tf.data.Dataset.from_tensor_slices((tf.cast(train_x,tf.float32),tf.cast(train_y,tf.int32)))
     #dataset = tf.cast(dataset, tf.float32)
     return dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

def model_fn(features, labels, mode, params):
    # del params# Unused
    # image = features
    # if isinstance(image, dict):
    #     image = features["image"]

    #image = tf.reshape(image, [-1, IMG_EDGE, IMG_EDGE, 1])
    #net = tf.feature_column.input_layer(features, [5,5,5])
    #for units in [5,5,5]:
    #  net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    model = ModelCNN()
    #print("Type of features is", type(features), features.shape)
    #print("Type of Labels is ", type(labels), labels.shape)
    #print(features.shape)
    #features = tf.reshape(features,[-1,12,1])

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = model(features)
        predictions = {
            'class_ids': tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits),
        }
        return tf.contrib.tpu.TPUEstimatorSpec(mode, predictions=predictions)

    logits = model(features)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)

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

    #estimator = create_estimator(batch_size)

    print("Predicting")
    def predict_input_fn(params):
        batch_size = params["batch_size"]
        dataset_predict = tf.data.Dataset.from_tensor_slices(data.astype('float32'))
        return dataset_predict.batch(batch_size)

    predictions = estimator.predict(predict_input_fn)
    i = 0
    for pred_dict in predictions:
        print(pred_dict.keys())
        print(pred_dict.values())
        i+=1
        if(i>=10):
          break
    # print("Getting labels")
    # labels = []
    # for pred_dict in predictions:
    #     labels.append(pred_dict['probabilities'])
    # labels = np.array(labels).astype('float64')

    predict_time = time.time() - start_time
    lock.release()

    return predictions, predict_time

if __name__=="__main__":
     #predict(data,32)
     train_x = pd.read_pickle("input/X_HB.pkl")[:1000]
     train_x.drop(['PU','pt'],1,inplace=True)
     train_y = pd.read_pickle("input/Y_HB.pkl")[:1000][['genE']]
     estimator = create_estimator(32)
     estimator.train(
      input_fn=lambda params: input_facile(train_x,train_y,
          params["batch_size"]),
      max_steps=TRAIN_STEPS)
     predict(train_x,32,estimator)
