from absl import logging
import argparse
import numpy as np
import pandas as pd
import json
from scipy import sparse
import collections
import dgl
import os
import torch

def save_graph_as_numpy(g, file_name):
    indptr_name = file_name + '_indptr.dat'
    indices_name = file_name + '_indices.dat'
    conf_name = file_name + '_conf.json'

    adj = g.adj_sparse('csc')
    indptr = adj[0].numpy()
    indices = adj[1].numpy()

    indptr_mmap = np.memmap(indptr_name, mode='w+', shape=indptr.shape, dtype=indptr.dtype)
    indices_mmap = np.memmap(indices_name, mode='w+', shape=indices.shape, dtype=indices.dtype)

    indptr_mmap[:] = indptr[:]
    indices_mmap[:] = indices[:]

    indptr_mmap.flush()
    indices_mmap.flush()

    mmap_config = dict()
    mmap_config['num_nodes'] = g.num_nodes()
    mmap_config['indptr_shape'] = tuple(indptr.shape)
    mmap_config['indptr_dtype'] = str(indptr.dtype)
    mmap_config['indices_shape'] = tuple(indices.shape)
    mmap_config['indices_dtype'] = str(indices.dtype)

    json.dump(mmap_config, open(conf_name, 'w'))


g = dgl.graph(([0, 0, 1], [0, 2, 2]), num_nodes=8)
adj = g.adj_sparse('csc')
indptr = adj[0].numpy()
indices = adj[1].numpy()
print(indptr)
print(indices)
file_name = "/data/nvme1n1p1/msh/synthetic/dummy/dummy_x4x4_csc_0"
save_graph_as_numpy(g, file_name)

g = dgl.graph(([1], [6]), num_nodes=8)
adj = g.adj_sparse('csc')
indptr = adj[0].numpy()
indices = adj[1].numpy()
print(indptr)
print(indices)
file_name = "/data/nvme1n1p1/msh/synthetic/dummy/dummy_x4x4_csc_1"
save_graph_as_numpy(g, file_name)

g = dgl.graph(([0, 0, 0], [2, 3, 4]), num_nodes=8)
adj = g.adj_sparse('csc')
indptr = adj[0].numpy()
indices = adj[1].numpy()
print(indptr)
print(indices)
file_name = "/data/nvme1n1p1/msh/synthetic/dummy/dummy_x4x4_csc_2"
save_graph_as_numpy(g, file_name)

g = dgl.graph(([1, 0], [0, 3]), num_nodes=8)
adj = g.adj_sparse('csc')
indptr = adj[0].numpy()
indices = adj[1].numpy()
print(indptr)
print(indices)
file_name = "/data/nvme1n1p1/msh/synthetic/dummy/dummy_x4x4_csc_3"
save_graph_as_numpy(g, file_name)
