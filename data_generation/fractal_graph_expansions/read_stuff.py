from absl import logging
import argparse
import numpy as np
import pandas as pd
import json
from scipy import sparse


def load_mmap_from_file(indptr_path, indices_path, conf_path):
    conf = json.load(open(conf_path, 'r'))

    indptr = np.memmap(indptr_path, mode='r', shape=tuple(conf['indptr_shape']), dtype=conf['indptr_dtype'])
    indices = np.memmap(indices_path, mode='r', shape=tuple(conf['indices_shape']), dtype=conf['indices_dtype'])
    num_nodes = conf['num_nodes']

    data = np.ones(conf['indices_shape'][0])
    matrix = sparse.csr_matrix((data, indices, indptr), shape=(num_nodes, num_nodes * args.col))

    return matrix


def read_file (data_name):
	data = np.load(data_name)
	for item in data.files:
		print(item)
		matrix = data[item]
		print("[{}] matrix.shape: {}, matrix: \n{}".format(data_name, matrix.shape, matrix))


def read_pickle_file (data_name):
	objects = []
	with (open(data_name, "rb")) as openfile:
	    while True:
	        try:
	            objects.append(pickle.load(openfile))
	        except EOFError:
	            break
	print(objects)


data1_name = "movielens_ndi_x2x4_0.npz"
data2_name = "movielens_ndi_x2x4_1.npz"

data1_my_name = "movielens_my_x2x4_0.npz"
data2_my_name = "movielens_my_x2x4_1.npz"
data_name = "movielens_x2x4.npz"

metadata_name = "movielens_metadata_x2x4"

# read_file(data1_name)
# read_file(data2_name)
# read_file(data1_my_name)
# read_file(data2_my_name)
# read_file(data_name)

# read_pickle_file(metadata_name)

num_edges = 0
multiplier = 4
dataset_name = "friendster"
# dataset_name = "ogbn_papers"
# dataset_name = "reddit"
for i in range(multiplier):
	file_name = "/home/nvadmin/msh/datasets/" + dataset_name + "/" + dataset_name + "_x" + str(multiplier) + "x" + str(multiplier) + "_csr_" + str(i) + "_conf.json"
	conf = json.load(open(file_name, 'r'))
	edge = conf['indices_shape'][0]
	num_edges += edge

print("{}_x{}x{} num_edges: {}".format(dataset_name, multiplier, multiplier, num_edges))

# conf_path = "/data/nvme1n1p1/msh/synthetic/twitter/twitter_x10x10_csc_0_conf.json"
# indptr_path = "/data/nvme1n1p1/msh/synthetic/twitter/twitter_x10x10_csc_0_indptr.dat"
# indices_path = "/data/nvme1n1p1/msh/synthetic/twitter/twitter_x10x10_csc_0_indices.dat"

# conf = json.load(open(conf_path, 'r'))
# indptr = np.memmap(indptr_path, mode='r', shape=tuple(conf['indptr_shape']), dtype=conf['indptr_dtype'])
# indices = np.memmap(indices_path, mode='r', shape=tuple(conf['indices_shape']), dtype=conf['indices_dtype'])

# print(indptr)
# print()
# print(indices)