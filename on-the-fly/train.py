import os
import sys
import pdb
import argparse
import numpy as np
import pandas as pd
import random
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import warnings

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from joblib import dump, load

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda, Dot, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import RMSprop,SGD
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

# importing loadDataUtility for some functions required for data preprocessing
import tools.loadData as loadData
import tools.model_zoo as model_zoo
import mytools.tools as mytools
import tools.visualization as visualization

alpha_value = 0.5
warnings.filterwarnings('ignore')


def build_pos_pairs_for_id(classid, classid_to_ids):  # classid --> e.g. 0
    # pos_pairs is actually the combination C(10,2)
    # e.g. if we have 10 example [0,1,2,...,9]
    # and want to create a pair [a, b], where (a, b) are different and order does not matter
    # e.g. [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
    # (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9)...]
    # C(10, 2) = 45
    traces = classid_to_ids[classid]
    pos_pair_list = []
    traceNum = len(traces)
    for i in range(traceNum):
        for j in range(i+1, traceNum):
            pos_pair = (traces[i], traces[j])
            pos_pair_list.append(pos_pair)
    random.shuffle(pos_pair_list)
    return pos_pair_list


def build_positive_pairs(class_id_range, classes_to_ids):
    # class_id_range = range(0, num_classes)
    listX1 = []
    listX2 = []
    for class_id in class_id_range:
        pos = build_pos_pairs_for_id(class_id, classes_to_ids)
        # -- pos [(1, 9), (0, 9), (3, 9), (4, 8), (1, 4),...] --> (anchor example, positive example)
        for pair in pos:
            listX1 += [pair[0]]  # identity
            listX2 += [pair[1]]  # positive example
    perm = np.random.permutation(len(listX1))
    # random.permutation([1,2,3]) --> [2,1,3] just random
    # random.permutation(5) --> [1,0,4,3,2]
    # In this case, we just create the random index
    # Then return pairs of (identity, positive example)
    # that each element in pairs in term of its index is randomly ordered.
    return np.array(listX1)[perm], np.array(listX2)[perm]


# Build a loss which doesn't take into account the y_true, as# Build
# we'll be passing only 0
def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)


# The triplet loss
def cosine_triplet_loss(X):
    _alpha = float(alpha_value)
    positive_sim, negative_sim = X
    losses = K.maximum(0.0, negative_sim - positive_sim + _alpha)
    return K.mean(losses)


# ------------------- Hard Triplet Mining -----------
# Naive way to compute all similarities between all network traces.
def build_similarities(conv, all_imgs):
    embs = conv.predict(all_imgs)
    embs = embs / np.linalg.norm(embs, axis=-1, keepdims=True)
    all_sims = np.dot(embs, embs.T)
    return all_sims


def intersect(a, b):
    return list(set(a) & set(b))

def build_negatives(anc_idxs, pos_idxs, similarities, neg_imgs_idx, id_to_classid, num_retries=50):
    # If no similarities were computed, return a random negative
    if similarities is None:
        return random.sample(neg_imgs_idx, len(anc_idxs))
    final_neg = []
    # for each positive pair
    for (anc_idx, pos_idx) in zip(anc_idxs, pos_idxs):
        anchor_class = id_to_classid[anc_idx]
        # positive similarity
        sim = similarities[anc_idx, pos_idx]
        # find all negatives which are semi(hard)
        possible_ids = np.where((similarities[anc_idx] + alpha_value) > sim)[0]
        possible_ids = intersect(neg_imgs_idx, possible_ids)
        appended = False
        for iteration in range(num_retries):
            if len(possible_ids) == 0:
                break
            idx_neg = random.choice(possible_ids)
            if id_to_classid[idx_neg] != anchor_class:
                final_neg.append(idx_neg)
                appended = True
                break
        if not appended:
            final_neg.append(random.choice(neg_imgs_idx))
    return final_neg


class SemiHardTripletGenerator():
    def __init__(self, Xa_train, Xp_train, batch_size, all_traces, neg_traces_idx, id_to_classid, conv):
        self.batch_size = batch_size

        self.traces = all_traces
        self.Xa = Xa_train
        self.Xp = Xp_train
        self.cur_train_index = 0
        self.num_samples = Xa_train.shape[0]
        self.neg_traces_idx = neg_traces_idx
        self.all_anchors = list(set(Xa_train))
        self.mapping_pos = {v: k for k, v in enumerate(self.all_anchors)}
        self.mapping_neg = {k: v for k, v in enumerate(self.neg_traces_idx)}
        self.id_to_classid = id_to_classid
        if conv:
            self.similarities = build_similarities(conv, self.traces)
        else:
            self.similarities = None

    def next_train(self):
        while 1:
            self.cur_train_index += self.batch_size
            if self.cur_train_index >= self.num_samples:
                self.cur_train_index = 0

            # fill one batch
            traces_a = self.Xa[self.cur_train_index: self.cur_train_index + self.batch_size]
            traces_p = self.Xp[self.cur_train_index: self.cur_train_index + self.batch_size]
            traces_n = build_negatives(traces_a, traces_p, self.similarities, self.neg_traces_idx, self.id_to_classid)
            yield ([self.traces[traces_a], self.traces[traces_p], self.traces[traces_n]],
                    np.zeros( shape=(traces_a.shape[0]) ))


def getClsIdDict(allData, allLabel, sampleNumLimit):
    allData, allLabel = mytools.limitData(allData, allLabel, sampleLimit=sampleNumLimit)
    allData, allLabel = np.array(allData), np.array(allLabel)

    label2IdDict = defaultdict(list)
    Id2Label = defaultdict(int)
    for i in range(len(allLabel)):
        label = int(allLabel[i])
        label2IdDict[label].append(i)
        Id2Label[i] = label

    return allData, allLabel, label2IdDict, Id2Label


def plot_figure(x, y, save_path, title_str, xlabel, ylabel):
    plt.title(title_str)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y)
    plt.savefig(save_path)
    plt.show(block=False)
    plt.close()


def train(opts, x, y, model_dir):
    # parameters for training the triplet network
    batch_size = 100
    batch_size = min(x.shape[0]-1, batch_size)
    number_epoch = opts.epochs
    nsamples = opts.nsamples
    emb_size = 256
    learning_rate = 0.00001
    leakage_model = 'HW'
    nb_classes = 9
    target_byte = opts.target_byte

    desp_format = 'Triplet_Model with parameters -- Alpha: {:f}, Batch_size: {}, emb_size: {}, epochs: {}, target_byte: {}, nsamples: {}, leakage model: {}'
    description = desp_format.format(alpha_value, batch_size, emb_size, number_epoch, target_byte, nsamples, leakage_model)
    print('[LOG] -- ' + description)

    # load the power traces on which base model is to be trained (Pre-training phase)
    # leakage_model can be {ID: identity model} or {HW: hamingway model}
    # shape of the dataset
    print('[LOG] -- shape of X_profiling: ', x.shape, '\tshape of y_profiling: ', y.shape)
    print('[LOG] -- number of classes in the dataset: ', len(set(y)))

    x, y, label2IdDict, Id2Label = getClsIdDict(x, y, nsamples)

    # reshaping the dataset for training
    all_traces = x.reshape((x.shape[0], x.shape[1], 1))
    print('[LOG] -- reshaped of the dataset for training: ', all_traces.shape)

    # generating anchor and positive pairs
    Xa_train, Xp_train = build_positive_pairs(range(0, nb_classes), label2IdDict)

    # Gather the ids of all power traces that are used for training
    # This just union of two sets set(A) | set(B)
    all_traces_train_idx = list(set(Xa_train) | set(Xp_train))
    print("[LOG] -- shape of X_train Anchor: ", Xa_train.shape)
    print("[LOG] -- shape of X_train Positive: ", Xp_train.shape)

    # path to save triplet model and its statistics
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'best_model.h5')

    K.clear_session()

    opt = RMSprop(lr=learning_rate)
    # opt = SGD(learning_rate=learning_rate, decay=0.0000001)
    shared_conv2 = model_zoo.cnn_best(input_shape=(all_traces.shape[1], 1), emb_size=emb_size, classification=False)
    anchor = Input((all_traces.shape[1], 1), name='anchor')
    positive = Input((all_traces.shape[1], 1), name='positive')
    negative = Input((all_traces.shape[1], 1), name='negative')

    a = shared_conv2(anchor)
    p = shared_conv2(positive)
    n = shared_conv2(negative)

    # The Dot layer in Keras now supports built-in Cosine similarity using the normalize = True parameter.
    # From the Keras Docs:
    # keras.layers.Dot(axes, normalize=True)
    # normalize: Whether to L2-normalize samples along the dot product axis before taking the dot product.
    #  If set to True, then the output of the dot product is the cosine proximity between the two samples.
    pos_sim = Dot(axes=-1, normalize=True)([a, p])
    neg_sim = Dot(axes=-1, normalize=True)([a, n])

    # customized loss
    loss = Lambda(cosine_triplet_loss,output_shape=(1,))([pos_sim, neg_sim])
    model_triplet = Model(inputs=[anchor, positive, negative], outputs=loss)

    print(model_triplet.summary())

    # compiling the triplet model
    model_triplet.compile(loss=identity_loss, optimizer=opt)
    # At first epoch we don't generate hard triplets
    gen_hard = SemiHardTripletGenerator(Xa_train, Xp_train, batch_size, all_traces, all_traces_train_idx, label2IdDict, None)
    csv_logger = CSVLogger(model_dir + 'Training_Log_{}.csv'.format(description), append=True, separator=';')
    callback_list = [csv_logger]
    print('[LOG] -- Training a feature extractor ...')
    loss_list = []
    x_list = []
    max_loss = 10
    steps_per_epoch = Xa_train.shape[0]//batch_size
    for epoch in range(number_epoch):
        print("[LOG] -- built new semi-hard generator for epoch " + str(epoch + 1))
        hist = model_triplet.fit(gen_hard.next_train(), epochs=1, verbose=1,
                                 steps_per_epoch=steps_per_epoch,
                                 callbacks=callback_list)

        # adjust learning rate
        if ((epoch+1) % 40) == 0:
            new_learning_rate = learning_rate / 2
            K.set_value(model_triplet.optimizer.learning_rate, new_learning_rate)
            print('[LOG] -- learning rate adjust from {:f} to {:f}'.format(learning_rate, new_learning_rate))
            learning_rate = new_learning_rate

        # For no semi-hard_triplet
        # gen_hard = HardTripletGenerator(Xa_train, Xp_train, batch_size, all_traces, all_traces_train_idx, None)
        tmp_loss = hist.history['loss'][0]
        loss_list.append(tmp_loss)
        x_list.append(epoch)
        if tmp_loss < max_loss:
            print('[LOG] -- loss improved from {:f} to {:f}, save model to path: {}'.format(max_loss, tmp_loss, model_path))
            shared_conv2.save(model_path)
            max_loss = tmp_loss

        gen_hard = SemiHardTripletGenerator(Xa_train, Xp_train, batch_size, all_traces, all_traces_train_idx, label2IdDict, shared_conv2)

    loss_file = os.path.join(model_dir, 'loss.png')
    plot_figure(x_list, loss_list, loss_file, 'loss curve', 'epoch', 'loss')
    print('[LOG] -- feature extractor saved to path: {}, loss figure save to file: {}'.format(model_path, loss_file))
    return shared_conv2


def train_one_knn(opts, knn_model_dir, x_train, y_train):
    ''' train and save the knn model  '''
    os.makedirs(knn_model_dir, exist_ok=True)
    eval_type = opts.eval_type
    leakage_model = 'HW'
    n_neighbors = 10

    # make sure the data is correct
    print('[LOG] -- using {} leakage model now'.format(leakage_model))
    assert(set(y_train) == set(range(9)))

    knn = KNeighborsClassifier(n_neighbors=n_neighbors,  # n-shot Number of neighbors to use by default for kneighbors queries. n for n-shot learning
                               weights='distance',
                               p=2,  # Power parameter for the Minkowski metric.
                               metric='cosine',  # the distance metric to use for the tree.
                               algorithm='brute'  # Algorithm used to compute the nearest neighbors
                               )
    knn.fit(x_train, y_train)

    acc = accuracy_score(y_train, knn.predict(x_train))
    print('[LOG] -- acc on training data set is: ', acc)

    model_name = 'knn_model.m'
    knn_model_path = os.path.join(knn_model_dir, model_name)
    dump(knn, knn_model_path)
    print('[LOG] -- kNN model save to path: {}'.format(knn_model_path))


def load_data(opts):
    # set params
    dpath = opts.input
    trace_num = opts.trace_num
    tmp = opts.attack_window.split('_')
    attack_window = [int(tmp[0]), int(tmp[1])]
    leakage_model = 'HW'
    method = opts.preprocess

    whole_pack = np.load(dpath)
    traces, plain_text, key = loadData.load_data_base(whole_pack, attack_window, method, trace_num)
    traces = traces[:, : np.newaxis]
    return traces, plain_text, key


def show_embedding(feat_model, x, y, outpath):
    ''' show the embedding of the training data '''
    feat = feat_model.predict(x)
    visualization.visualize(feat, y, outpath)
    print('[LOG] -- visualization figure save to path: ', outpath)


def main(opts):
    ''' model train test '''
    # set the path and parameters
    model_root_dir = opts.output
    leakage_model = 'HW'

    '''
    if os.path.exists(model_root_dir):
        all_files = [d for d in os.listdir(model_root_dir) if os.path.isdir(os.path.join(model_root_dir, d))]
    else:
        all_files = []
    if [] == all_files:
        start_idx = 0
    else:
        all_vals = [int(tmp.split('_')[-1]) for tmp in all_files]
        start_idx = max(all_vals) - 1
    '''
    start_idx = opts.start_idx
    end_idx = opts.end_idx

    # load the data
    train_x, plain_text, key = load_data(opts)
    target_byte = opts.target_byte
    for guess_key in range(start_idx, end_idx):
        print('[LOG] -- now start training triplet model for guess_key: ', guess_key)
        model_dir = os.path.join(model_root_dir, 'guess_key_{}'.format(guess_key), 'feat_model')
        knn_model_dir = os.path.join(model_root_dir, 'guess_key_{}'.format(guess_key), 'knn_model')

        train_y = loadData.get_labels(plain_text, guess_key, target_byte, leakage_model)

        start = time.time()
        feat_model = train(opts, train_x, train_y, model_dir)
        end = time.time()
        print('[LOG -- RUN TIME] -- triplet nn training time with guess key {} is {:f}'.format(guess_key, end-start))

        # extract features for train data
        train_x_feat = feat_model.predict(train_x)
        print('{lOG} -- x train feat shape is: ', train_x_feat.shape)

        # compensate the missing class data
        needed_classes = set(range(9))
        if needed_classes != set(train_y):
            missing_class = needed_classes - set(train_y)
            print('[LOG] -- for guess_key {}, data of some class are missing, the missing class is: {}'.format(guess_key, missing_class))
            missing_data, missing_label = [], []
            for label in missing_class:
                missing_label.append(label)
                missing_data.append([0]*train_x_feat.shape[1])
            train_x_feat = np.concatenate((train_x_feat, missing_data), axis=0)
            train_y = np.concatenate((train_y, missing_label), axis=0)

        # train the knn
        train_one_knn(opts, knn_model_dir, train_x_feat, train_y)

        # show the embeddings
        outpath = os.path.join(model_dir, 'train_data_visualization.png')
        show_embedding(feat_model, train_x, train_y[:train_x.shape[0]], outpath)
        print('[LOG] -- finish training model for guess key: ', guess_key)

    print('[LOG] -- all done!')



def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='')
    parser.add_argument('-o', '--output', help='')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='')
    parser.add_argument('-ns', '--nsamples', type=int, help='')
    parser.add_argument('-tb', '--target_byte', type=int, default=2, help='')
    parser.add_argument('-aw', '--attack_window', help='')
    parser.add_argument('-et', '--eval_type', help='')
    parser.add_argument('-tn', '--trace_num', type=int, default=0, help='')
    parser.add_argument('-pp', '--preprocess', default='', choices={'', 'norm', 'scailing'}, help='')
    parser.add_argument('-pt', '--pre_train', action='store_true', help='')
    parser.add_argument('-si', '--start_idx', type=int, default=0, help='')
    parser.add_argument('-ei', '--end_idx', type=int, default=256, help='')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseArgs(sys.argv)
    if tf.test.is_gpu_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    main(opts)
