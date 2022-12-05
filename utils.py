from nis import match
import re

import numpy as np
import scipy.sparse as sp
import sklearn
import sklearn.metrics
import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import random
from columns import COLUMNS, SMALL_COLUMNS


def focal_loss(truth, pred, alpha=0.25, gamma=2.0):
    """Own focal loss fuction

    # would need one hot encoded target
    # from torchvision.ops.focal_loss import sigmoid_focal_loss
    https://pytorch.org/vision/stable/_modules/torchvision/ops/focal_loss.html

    https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
    https://paperswithcode.com/method/focal-loss
    """
    bce_loss = F.cross_entropy(truth, pred)
    return alpha * (1 - torch.exp(-bce_loss)) ** gamma * bce_loss


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {
        c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)
    }
    labels_onehot = np.array(
        list(map(classes_dict.get, labels)), dtype=np.int32
    )
    return labels_onehot


def loaddata(filename):
    df = pd.read_csv(filename, header=None, delimiter=",")
    a = np.array(df.as_matrix())
    return a


def load_custom_ts(
    path="/opt/et-data/preprocessed",
    columns=COLUMNS,
    duration_max=45,
    start=-np.Inf,
    stop=np.Inf,
    step="1.0",
    target_col="product_bought_healthy",
    random_seed=0,
    split_ratio=0.2,
    # within_allowed=True,
    tensor_format=True,
):
    # load csv
    df = pd.read_csv(
        f"{path}/preprocessed_v1_{duration_max}_sec_{step}step.csv"
    )
    df = df[(df.step > start) & (df.step < stop)]

    # label
    y = df.groupby(["subject", "task"]).first()[target_col].to_numpy()
    nclass = int(np.amax(y)) + 1

    # convert all features np array
    target_pattern = re.compile(r"^product_bought.*")
    head_pattern = re.compile(r"^head*")
    X = df.set_index(["subject", "task"])
    col_list = [
        col
        for col in X.columns
        if col not in ["step", "gender"]
        and not target_pattern.match(col)
        and not head_pattern.match(col)
    ]
    current_list = []
    for idx in X.index.unique():
        current_list.append(X.loc[idx][col_list].to_numpy())
    X = np.array(current_list)

    # create train / val / test mask and indices
    n_trails = int(y.shape[0])
    mask_1 = [1] * int(n_trails * split_ratio)  # val
    mask_2 = [2] * int(n_trails * split_ratio)  # test
    mask_0 = [0] * (n_trails - len(mask_1) - len(mask_2))  # train
    mask = np.array(mask_0 + mask_1 + mask_2)
    random.seed(random_seed)
    random.shuffle(mask)
    idx_train = np.where(mask == 0)[0]
    idx_val = np.where(mask == 1)[0]
    idx_test = np.where(mask == 2)[0]

    if tensor_format:
        X = torch.FloatTensor(X)
        labels = torch.LongTensor(y)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

    return X, labels, idx_train, idx_val, idx_test, nclass


def load_raw_ts(path, dataset, tensor_format=True):
    path = path + "raw/" + dataset + "/"
    x_train = np.load(path + "X_train.npy")
    y_train = np.load(path + "y_train.npy")
    x_test = np.load(path + "X_test.npy")
    y_test = np.load(path + "y_test.npy")
    ts = np.concatenate((x_train, x_test), axis=0)
    ts = np.transpose(ts, axes=(0, 2, 1))
    labels = np.concatenate((y_train, y_test), axis=0)
    nclass = int(np.amax(labels)) + 1

    train_size = y_train.shape[0]
    total_size = labels.shape[0]

    idx_train = range(train_size)
    idx_val = range(train_size, total_size)
    idx_test = range(train_size, total_size)

    if tensor_format:
        # features = torch.FloatTensor(np.array(features))
        ts = torch.FloatTensor(np.array(ts))
        labels = torch.LongTensor(labels)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

    return ts, labels, idx_train, idx_val, idx_test, nclass


def normalize(mx):
    """Row-normalize sparse matrix"""
    row_sums = mx.sum(axis=1)
    mx = mx.astype("float32")
    row_sums_inverse = 1 / row_sums
    f = mx.multiply(row_sums_inverse)
    return sp.csr_matrix(f).astype("float32")


def accuracy(output, labels):
    preds = output.max(1)[1].cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy_score = sklearn.metrics.accuracy_score(labels, preds)

    return accuracy_score


def fbeta(output, labels, average="binary", beta=1.5):
    preds = output.max(1)[1].cpu().numpy()
    labels = labels.cpu().numpy()
    fbeta_score = sklearn.metrics.fbeta_score(
        labels, preds, average=average, beta=beta
    )

    return fbeta_score


def confusion_matrix(output, labels, average="weighted", beta=1.5):
    preds = output.max(1)[1].cpu().numpy()
    labels = labels.cpu().numpy()
    cm = sklearn.metrics.confusion_matrix(labels, preds)

    return cm


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def output_conv_size(in_size, kernel_size, stride, padding):

    output = int((in_size - kernel_size + 2 * padding) / stride) + 1

    return output


def dump_embedding(
    proto_embed, sample_embed, labels, dump_file="./plot/embeddings.txt"
):
    proto_embed = proto_embed.cpu().detach().numpy()
    sample_embed = sample_embed.cpu().detach().numpy()
    embed = np.concatenate((proto_embed, sample_embed), axis=0)

    nclass = proto_embed.shape[0]
    labels = np.concatenate(
        (
            np.asarray([i for i in range(nclass)]),
            labels.squeeze().cpu().detach().numpy(),
        ),
        axis=0,
    )

    with open(dump_file, "w") as f:
        for i in range(len(embed)):
            label = str(labels[i])
            line = (
                label + "," + ",".join(["%.4f" % j for j in embed[i].tolist()])
            )
            f.write(line + "\n")
