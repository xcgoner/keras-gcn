from __future__ import print_function

import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence
import random
from kegra.utils import *

def shuffle_edges(edges, n_nodes, n_rounds):
    e = np.copy(edges)

    for i in range(n_rounds):
        e1 = random.randint(0, e.shape[0]-1)
        e2 = random.randint(0, e.shape[0]-1)
        if e1 == e2 or e[e1, 0] == e[e2, 0] or e[e1, 1] == e[e2, 1] or e[e1, 0] == e[e2, 1] or e[e1, 1] == e[e2, 0]:
            continue
        else:
            # swap edge
            tmp = e[e1, 1]
            e[e1, 1] = e[e2, 0]
            e[e2, 0] = tmp

    adj = sp.coo_matrix((np.ones(e.shape[0]), (e[:, 0], e[:, 1])),
                        shape=(n_nodes, n_nodes), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = make_sym_adj(adj)
    return adj

def shuffle_mix(edges, n_nodes, alpha):
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n_nodes, n_nodes), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = make_sym_adj(adj)
    shuffled_adj = shuffle_edges(edges, n_nodes, edges.shape[0] * 2)

    adj = adj * (1-alpha) + shuffled_adj * alpha

    return adj

