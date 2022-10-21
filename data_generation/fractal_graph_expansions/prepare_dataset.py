from ogb.nodeproppred import DglNodePropPredDataset
from scipy import sparse

import collections
import dgl
import os
import numpy as np
import json
import torch

name = "reddit"
base_dir = "/data/nvme1n1p1/msh/synthetic/"


def save_graph_as_numpy(graph, file_name):
    indptr_name = file_name + 'indptr.dat'
    indices_name = file_name + 'indices.dat'
    conf_name = file_name + 'conf.json'

    adj = graph.adj_sparse('csc')
    indptr = adj[0].numpy()
    indices = adj[1].numpy()

    indptr_mmap = np.memmap(indptr_name, mode='w+', shape=indptr.shape, dtype=indptr.dtype)
    indices_mmap = np.memmap(indices_name, mode='w+', shape=indices.shape, dtype=indices.dtype)

    indptr_mmap[:] = indptr[:]
    indices_mmap[:] = indices[:]

    indptr_mmap.flush()
    indices_mmap.flush()

    mmap_config = dict()
    mmap_config['num_nodes'] = graph.num_nodes()
    mmap_config['indptr_shape'] = tuple(indptr.shape)
    mmap_config['indptr_dtype'] = str(indptr.dtype)
    mmap_config['indices_shape'] = tuple(indices.shape)
    mmap_config['indices_dtype'] = str(indices.dtype)

    json.dump(mmap_config, open(conf_name, 'w'))


if name == "ogbn-products":
	dataset = DglNodePropPredDataset(name = "ogbn-products")
	graph, label = dataset[0] # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
	save_graph_as_numpy(graph, base_dir + "ogbn_products/raw/")

elif name == "reddit":
	dataset = dgl.data.RedditDataset()
	graph = dataset[0]
	save_graph_as_numpy(graph, base_dir + "reddit/raw/")
