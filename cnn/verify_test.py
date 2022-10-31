#!/usr/bin/python3
import os
import sys
import argparse
import pdb

import h5py
import numpy as np
import ast
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import train
import tools.checking_tool as checking_tool
import tools.loadData as loadData


def verify_test(model, dataset, key, plaintext_attack, labels, target_byte):
    # Get the input layer shape
    input_layer_shape = model.get_layer(index=0).input_shape[0]
    # Sanity check
    if input_layer_shape[1] != dataset.shape[1]:
        raise ValueError("Error: model input shape %d instead of %d is not expected ..." % (input_layer_shape[1], len(dataset[0, :])))

    # Adapt the data shape according our model input
    if len(input_layer_shape) == 2:
        print('# This is a MLP, no need to reshape the data')
    elif len(input_layer_shape) == 3:
        print('# This is a CNN, need to reshape the data')
        input_data = dataset[:, :, np.newaxis]
    else:
        raise ValueError("Error: model input shape length %d is not expected ..." % len(input_layer_shape))

    # Predict our probabilities
    predictions = model.predict(input_data)

    count = 0
    for p in range(predictions.shape[0]):
        text_i = plaintext_attack[p]
        one_predict = predictions[p]
        pred_label = np.argmax(one_predict)
        act_label = labels[p]
        if pred_label == act_label:
            count += 1

    total_counts = predictions.shape[0]
    acc = count / total_counts

    print('the test accuracy is: {:f}'.format(acc))


'''
def load_data(opts):
    # checking_tool.check_file_exists(ascad_database_file)
    # in_file = h5py.File(ascad_database_file, "r")
    inDir = opts.input
    keyfile = os.path.join(opts.input, 'key.txt')
    train_key, val_key = loadData.loadkey(keyfile)
    train_whole_pack, val_whole_pack = loadData.load_data(opts.input)

    # since it is testing, we do not need train_whole_pack, we can release the memory
    del train_whole_pack
    val_traces, val_text_ins, val_labels = val_whole_pack['data'], val_whole_pack['text_in'], val_whole_pack['label']
    return val_traces, val_text_ins, val_labels, val_key
'''


def get_the_labels(textins, key, target_byte):
    labels = []
    for i in range(textins.shape[0]):
        text_i = textins[i]
        label = loadData.aes_internal(text_i[target_byte], key[target_byte])
        labels.append(label)

    labels = np.array(labels)
    return labels


def load_data(opts, target_byte):
    # checking_tool.check_file_exists(ascad_database_file)
    # in_file = h5py.File(ascad_database_file, "r")
    keyfile = os.path.join(opts.input, 'key.txt')
    train_key, val_key = loadData.loadkey(keyfile)
    #val_key = train_key

    if 'STM32F3' == opts.dataset_name:
        val_data_file = os.path.join(opts.input, 'third_batch_STM32F3.npz')
    elif 'MPEG' == opts.dataset_name:
        val_data_file = os.path.join(opts.input, 'val_batch_MPEG.npz')
    else:
        raise NotImplementedError()

    start_idx, end_idx = opts.start_idx, opts.end_idx
    val_data_whole_pack = np.load(val_data_file)
    start_idx, end_idx = opts.start_idx, opts.end_idx
    val_traces = val_data_whole_pack['trace_mat'][:, start_idx:end_idx]
    val_textins = val_data_whole_pack['textin_mat']
    
    val_labels = get_the_labels(val_textins, val_key, target_byte)

    return val_traces, val_textins, val_labels, val_key


# Check a saved model against one of the testing databases Attack traces
def main(opts):
    # checking model file existence
    model_file = opts.model_file
    checking_tool.check_file_exists(model_file)
    target_byte = 0

    # Load profiling and attack data and metadata from the ASCAD database
    X_attack, plaintext_attack, Y_attack, key = load_data(opts, target_byte)
    #X_attack, plaintext_attack, Y_attack, key = load_data(opts)
    sample_num = 2000
    X_attack = X_attack[:sample_num]
    plaintext_attack = plaintext_attack[:sample_num]
    Y_attack = Y_attack[:sample_num]

    # Load model
    inp_shape = (500, 1)
    clsNum = 256
    model = train.cnn_best2(inp_shape, clsNum)
    print('loading weights...')
    model.load_weights(model_file)

    # run the accuracy test

    # We test the rank over traces of the Attack dataset, with a step of 10 traces
    print('start verify the results...')
    verify_test(model, X_attack, key, plaintext_attack, Y_attack, target_byte)


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-o', '--output', help='')
    parser.add_argument('-m', '--model_file', help='')
    parser.add_argument('-g', '--useGpu', action='store_true', help='')
    parser.add_argument('-v', '--verbose', action='store_true', help='')
    parser.add_argument('-dn', '--dataset_name', choices={'STM32F3', 'MPEG'}, help='')
    parser.add_argument('-si', '--start_idx', type=int, default=0, help='')
    parser.add_argument('-ei', '--end_idx', type=int, default=500, help='')
    opts = parser.parse_args()
    return opts


if __name__=="__main__":
    opts = parseArgs(sys.argv)
    if opts.useGpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    main(opts)
