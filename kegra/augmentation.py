from __future__ import print_function

import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence
import random
import networkx as nx
from kegra.utils import *

def shuffle_edges(nodes, edges, n_nodes, n_rounds):
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

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(e)
    adj = nx.adjacency_matrix(G)
    adj = adj.astype('float32')
    return adj

def shuffle_mix(nodes, edges, n_nodes, alpha):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    adj = nx.adjacency_matrix(G)
    adj = adj.astype('float32')

    shuffled_adj = shuffle_edges(nodes, edges, n_nodes, edges.shape[0]*2)

    adj = adj * (1-alpha) + shuffled_adj * alpha

    return adj

