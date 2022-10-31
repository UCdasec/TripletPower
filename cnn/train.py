#!/usr/bin python3.6
import os
import sys
import argparse
import pdb
import h5py
import time

import tensorflow as tf
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

import tools.loadData as loadData
import tools.model_zoo as model_zoo


def load_training_data(opts):
    '''data loading function'''
    target_byte = opts.target_byte
    leakage_model = opts.leakage_model
    data_path = opts.input
    tmp = opts.attack_window.split('_')
    attack_window = [int(tmp[0]), int(tmp[1])]
    max_trace_num = opts.max_trace_num
    method = opts.preprocess

    whole_pack = np.load(data_path)
    traces, text_in, key = loadData.load_data_base(whole_pack, attack_window, method, train_num=max_trace_num)
    inp_shape = (traces.shape[1], 1)
    assert(traces.shape[0]==opts.max_trace_num)
    loadData.data_info(traces.shape, text_in.shape, key)
    labels = loadData.get_labels(text_in, key[target_byte], target_byte, leakage_model)
    return traces, labels, inp_shape


def print_run_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        print('[LOG -- RUN TIME] -- current function [{}] run time is {:f}'.format(func.__name__, end-start))
    return wrapper


# Training high level function
@print_run_time
def train_model(opts, X_profiling, Y_profiling, model, epochs, batch_size=100, verbose=False):
    ''' train the model '''
    # make resDir and modelDir
    modelDir = os.path.join(opts.output, 'model')
    os.makedirs(modelDir, exist_ok=True)
    model_save_file = os.path.join(modelDir, 'best_model.h5')

    # Save model every epoch
    checkpointer = ModelCheckpoint(model_save_file, monitor='val_accuracy', verbose=verbose, save_best_only=True, mode='max')
    earlyStopper = EarlyStopping(monitor='val_accuracy', mode='max', patience=10)
    #callbacks = [checkpointer, earlyStopper]
    callbacks = [checkpointer]

    # Get the input layer shape
    input_layer_shape = model.get_layer(index=0).input_shape
    if isinstance(input_layer_shape, list):
        input_layer_shape = input_layer_shape[0]

    # Sanity check
    Reshaped_X_profiling = loadData.sanity_check(input_layer_shape, X_profiling)

    clsNum = 9 if 'HW'==opts.leakage_model else 256
    print('[LOG] -- class number is: ', clsNum)
    Y_profiling = to_categorical(Y_profiling, clsNum)
    hist = model.fit(x=Reshaped_X_profiling, y=Y_profiling,
                        validation_split=0.1, batch_size=batch_size,
                        verbose=verbose, epochs=epochs,
                        shuffle=True, callbacks=callbacks)

    print('[LOG] -- model save to path: {}'.format(model_save_file))

    loss_val = hist.history['loss']
    x = list(range(1, len(loss_val)+1))
    fig_save_path = os.path.join(modelDir, 'loss.png')
    title = 'loss curve'
    xlabel = 'loss'
    ylabel = 'epoch'
    plot_figure(x, loss_val, fig_save_path, title, xlabel, ylabel)
    print('{LOG} -- loss figure save to path: ', fig_save_path)


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-o', '--output', help='')
    parser.add_argument('-v', '--verbose', action='store_true', help='')
    parser.add_argument('-tb', '--target_byte', type=int, default=0, help='default value is 0')
    parser.add_argument('-lm', '--leakage_model', choices={'HW', 'ID'}, help='')
    parser.add_argument('--epochs', type=int, default=100, help='')
    parser.add_argument('-aw', '--attack_window', default='', help='overwrite the attack window')
    parser.add_argument('-mtn', '--max_trace_num', type=int, default=0, help='')
    parser.add_argument('-pp', '--preprocess', default='', choices={'norm', 'scaling', ''}, help='')
    opts = parser.parse_args()
    return opts


def plot_figure(x, y, fig_save_path, title_str, xlabel, ylabel):
    plt.title(title_str)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.plot(x, y)
    plt.savefig(fig_save_path)
    plt.show(block=False)


def main(opts):
    # get the params
    leakage_model = opts.leakage_model
    verbose = opts.verbose
    target_byte = opts.target_byte
    epochs = opts.epochs
    batch_size = 100

    # get the data and model
    #load traces
    X_profiling, Y_profiling, input_shape = load_training_data(opts)
    print('[LOG] -- trace data shape is: ', X_profiling.shape)

    print('[LOG] -- now train dnn model for {} leakage model...'.format(leakage_model))
    if 'HW' == leakage_model:
        best_model = model_zoo.cnn_best(input_shape, emb_size=9, classification=True)
    else:
        best_model = model_zoo.cnn_best(input_shape, emb_size=256, classification=True)
    best_model.summary()
    train_model(opts, X_profiling, Y_profiling, best_model, epochs, batch_size, verbose)


if __name__ == "__main__":
    opts = parseArgs(sys.argv)
    if tf.test.is_gpu_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    main(opts)

