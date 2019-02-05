from __future__ import print_function

import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """Load data."""
    DATA_PATH = 'data/'

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}ind.{}.{}".format(DATA_PATH, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}ind.{}.test.index".format(DATA_PATH, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder),
                                    max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)
    adj = adj.astype('float32')
    edges = np.array(G.edges())
    nodes = G.nodes()

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return nodes, edges, adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()

def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm

def preprocess_adj(adj, symmetric=True, self_loop="eye"):
    if self_loop == "eye":
        adj = adj + sp.eye(adj.shape[0])
    elif self_loop == "degree":
        adj = adj + sp.diags(np.power(np.array(adj.sum(1)), 0.3).flatten(), 0)
        # adj = adj + sp.diags(np.array(adj.sum(1)).flatten(), 0)
    else:
        pass
    adj = normalize_adj(adj, symmetric)
    return adj

def approx_expm(M, k):
    M_exp = sp.eye(M.shape[0]) + M
    M_factor = M
    for i in range(2,k):
        M_factor = M_factor * M / float(i)
        M_exp += M_factor
    return M_exp

def soft_threshold(M, threshold = 0, scale = 1):
    M = (M - threshold * sp.eye(M.shape[0])) * scale
    M_exp = linalg.expm(M)
    # print(type(M_exp), flush=True)
    return M_exp / M_exp.diagonal().sum()

def approx_soft_threshold(M, threshold = 0, scale = 1, k=4):
    M = (M - threshold * sp.eye(M.shape[0])) * scale
    M_exp = approx_expm(M, k)
    # print(type(M_exp), flush=True)
    return M_exp / M_exp.diagonal().sum()