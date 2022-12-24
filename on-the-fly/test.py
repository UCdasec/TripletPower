import os
import sys
import pdb
import argparse
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from tqdm import tqdm

import matplotlib.pyplot as plt
from joblib import dump, load
import tools.loadData as loadData
import train


def get_x_feat(x_test, feat_ext_path):
    # loading feature extractor & extract features for training
    print('loading the feature extractor from path: ', feat_ext_path)
    if len(x_test.shape) < 3:
        x_test = x_test[:, :, np.newaxis]
    feat_model = load_model(feat_ext_path)
    x_test = feat_model.predict(x_test)
    print('[LOG] -- shape of x_test: ', x_test.shape)
    del feat_model
    return x_test


def test_one_knn(opts, x_test, y_test, guess_key, real_key, knn_model_path, leakage_model):
    """ generate ranking raw data """
    # get all the params
    eval_type = opts.eval_type

    # trained knn models path
    knn_model = load(knn_model_path)
    print('[LOG] -- making predictions of guessed key {} and its corresopnding k-nn models {}'.format(guess_key, knn_model_path))

    y_preds = knn_model.predict(x_test)
    acc = accuracy_score(y_test, y_preds)
    print('[LOG] -- knn model: {}, guess key is: {}, real key is: {} test acc is: {}'.format(knn_model_path, guess_key, real_key, acc))

    del knn_model
    return acc


def plot_figure(x, y, real_key, xlabel, ylabel, title, fig_save_path):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks([0, real_key, 255])
    plt.plot(x, y)
    plt.savefig(fig_save_path)
    plt.show()
    plt.close('all')


def main(opts):
    ''' model train test '''
    # set the path and parameters
    model_root_dir = opts.modelDir
    leakage_model = 'HW'
    target_byte = opts.target_byte
    leakage_model = 'HW'
    model_root_dir = opts.modelDir

    # load the data
    test_x, plain_text, key = train.load_data(opts)
    real_key = key[target_byte]
    acc_list = []
    start_idx = opts.start_idx
    end_idx = opts.end_idx
    for guess_key in tqdm(range(start_idx, end_idx)):
        # set up corresponding model dir
        model_dir = os.path.join(model_root_dir, 'guess_key_{}'.format(guess_key))
        feat_model_path = os.path.join(model_dir, 'feat_model', 'best_model.h5')
        knn_model_path = os.path.join(model_dir, 'knn_model', 'knn_model.m')

        # extract x_test features
        test_x_feat = get_x_feat(test_x, feat_model_path)

        test_y = loadData.get_labels(plain_text, guess_key, target_byte, leakage_model)
        tmp_acc = test_one_knn(opts, test_x_feat, test_y, guess_key, real_key, knn_model_path, leakage_model)
        acc_list.append(tmp_acc)

        # release the memory
        del test_x_feat, test_y

    # plot and save the figure
    x = list(range(start_idx, end_idx))
    y = acc_list

    train_num = os.path.basename(model_root_dir)
    outDir = os.path.join(os.path.dirname(model_root_dir), 'res', train_num, opts.output)
    os.makedirs(outDir, exist_ok=True)
    fname = 'knn-256-acc_{}'.format(opts.idx)
    dpath = os.path.join(outDir, '{}'.format(fname))
    np.save(dpath, y)

    xlabel = 'key value'
    ylabel = 'accuracy'
    title = 'real key is {}'.format(real_key)
    fig_save_path = os.path.join(outDir, '{}.png'.format(fname))
    plot_figure(x, y, real_key, xlabel, ylabel, title, fig_save_path)
    print('[LOG] -- figure save to path: ', fig_save_path)

    print('[LOG] -- all done!')


def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-m', '--modelDir', help='')
    parser.add_argument('-o', '--output', help='')
    parser.add_argument('-tb', '--target_byte', type=int, default=2, help='')
    parser.add_argument('-aw', '--attack_window', help='')
    parser.add_argument('-pp', '--preprocess', default='', choices={'', 'norm', 'scaling'}, help='')
    parser.add_argument('-et', '--eval_type', help='')
    parser.add_argument('-tn', '--trace_num', type=int, default=20000, help='')
    parser.add_argument('-si', '--start_idx', default=0, type=int, help='')
    parser.add_argument('-ei', '--end_idx', default=0, type=int, help='')
    parser.add_argument('--idx', default=0, type=int, help='')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseArgs(sys.argv)
    main(opts)

