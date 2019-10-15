#!/usr/bin/env python
from keras.models import Model, load_model
from keras.layers import Dense, BatchNormalization, Input, Dropout, Activation, concatenate, GRU
from keras.utils import np_utils
from keras.optimizers import Adam, Nadam, SGD
import keras.backend as K
from tensorflow.python.framework import graph_util, graph_io
import os
import numpy as np
import pandas as pd
import time
import tensorflow as tf

VALSPLIT = 0.2  # 0.7
np.random.seed(5)
Nrhs = 2100000


def _make_parent(path):
    os.system('mkdir -p %s' % ('/'.join(path.split('/')[:-1])))

# Sample class
# X,Y,kin,idx,Yhat,time


class Sample(object):
    def __init__(self):
        pass

    def setvalues(name, base, region):
        self.name = name

        if region != '':
            self.X = pd.read_pickle(
                '%s/%s_%s.pkl' %
                (base, 'X', region))[
                :Nrhs]
            self.Y = pd.read_pickle(
                '%s/%s_%s.pkl' %
                (base, 'Y', region))[
                :Nrhs]
            self.kin = pd.read_pickle(
                '%s/%s_%s.pkl' % (base, 'X', region))[:Nrhs][['PU', 'ieta', 'iphi', 'pt']]
        else:
            self.X = np.load(
                '%s/%s.pkl' %
                (base, 'X'), allow_pickle=True)[
                :Nrhs]
            self.Y = np.load(
                '%s/%s.pkl' %
                (base, 'Y'), allow_pickle=True)[
                :Nrhs]
            self.kin = np.load('%s/%s.pkl' % (base, 'X'),
                               allow_pickle=True)[:Nrhs][['PU', 'ieta', 'iphi', 'pt']]

        print(self.X.shape, self.Y.shape)
        self.X.drop(['PU', 'pt'], 1, inplace=True)
        self.idx = np.random.permutation(self.X.shape[0])

    @property
    def tidx(self):
        if VALSPLIT == 1 or VALSPLIT == 0:
            return self.idx
        else:
            return self.idx[int(VALSPLIT * len(self.idx)):]

    @property
    def vidx(self):
        if VALSPLIT == 1 or VALSPLIT == 0:
            return self.idx
        else:
            return self.idx[:int(VALSPLIT * len(self.idx))]

    def infer(self, model, batch_size, limit=None):
        if limit is None:
            self.Yhat, self.time = model.predict(self.X, batch_size)
        else:
            self.Yhat, self.time = model.predict(self.x[:limit], batch_size)

    def standardize(self, mu, std):
        self.X = (self.X - mu) / std

# Model class


class ClassModel(object):
    def __init__(self, n_inputs, gpu):
        self._hidden = 0
        self.n_inputs = n_inputs
        self.n_targets = 1

        self.inputs = Input(shape=(n_inputs,), name='input')

        if gpu:
            with tf.device('/gpu:0'):
                self.outputs = self.get_outputs()
        else:
            with tf.device('/cpu:0'):
                self.outputs = self.get_outputs()

        self.model = Model(inputs=self.inputs, outputs=self.outputs)
        self.model.compile(optimizer=Adam(), loss='mse')

        self.model.summary()

    def get_outputs(self):
        # To be overridden
        self.name = None
        self.epochs = 20
        return None

    def train(self, sample, num_epochs=0, mahi=False):
        if num_epochs == 0:
            num_epochs = self.epochs
        tX = sample.X.values[sample.tidx]
        vX = sample.X.values[sample.vidx]
        if mahi:
            tY = sample.Y[['energy']].values[sample.tidx]
            vY = sample.Y[['energy']].values[sample.vidx]
        else:
            tY = sample.Y[['genE']].values[sample.tidx]
            vY = sample.Y[['genE']].values[sample.vidx]

        history = self.model.fit(
            tX,
            tY,
            batch_size=2024,
            epochs=num_epochs,
            shuffle=True,
            validation_data=(
                vX,
                vY))

        with open('history.log', 'w') as flog:
            history = history.history
            flog.write(','.join(history.keys()) + '\n')
            for l in zip(*history.values()):
                flog.write(','.join([str(x) for x in l]) + '\n')

    def save_as_keras(self, path):
        _make_parent(path)
        self.model.save(path)
        print('Saved to', path)

    def save_as_tf(self, path):
        _make_parent(path)
        sess = K.get_session()
        print([l.op.name for l in self.model.inputs], '->',
              [l.op.name for l in self.model.outputs])
        graph = graph_util.convert_variables_to_constants(
            sess, sess.graph.as_graph_def(), [
                n.op.name for n in self.model.outputs])
        p0 = '/'.join(path.split('/')[:-1])
        p1 = path.split('/')[-1]
        graph_io.write_graph(graph, p0, p1, as_text=False)
        print('Saved to', path)

    def predict(self, *args, **kwargs):
        start_time = time.time()
        predictions = self.model.predict(*args, **kwargs)
        total_time = time.time() - start_time
        return predictions, total_time

    def load_model(self, path):
        self.model = load_model(path)
