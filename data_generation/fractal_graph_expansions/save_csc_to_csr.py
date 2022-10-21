from absl import app
from absl import logging
from absl import flags
import argparse
import numpy as np
import json
from scipy import sparse
import dgl
import torch

argparser = argparse.ArgumentParser()

argparser.add_argument('--dataset', type=str, default="dummy")
flags.DEFINE_string("dataset",
                    "dummy",
                    "dataset name")

args = argparser.parse_args()
FLAGS = flags.FLAGS

# assert args.dataset != "dummy"

# base_dir = "/data/nvme1n1p1/msh/synthetic/"
base_dir = "../../../dataset/np/"

def load_mmap_from_file(indptr_path, indices_path, conf_path):
	conf = json.load(open(conf_path, 'r'))
	
	indptr = torch.from_numpy(np.memmap(indptr_path, mode='r', shape=tuple(conf['indptr_shape']), dtype=conf['indptr_dtype']))
	indices = torch.from_numpy(np.memmap(indices_path, mode='r', shape=tuple(conf['indices_shape']), dtype=conf['indices_dtype']))
	num_nodes = conf['num_nodes']

	data = ('csc', (indptr, indices, []))
	g = dgl.graph(data, num_nodes=num_nodes)
	g = g.formats('csr')

	return num_nodes, g.adj_sparse('csr')[0].numpy(), g.adj_sparse('csr')[1].numpy()

def save_as_numpy(indptr, indices, num_nodes, file_name):
    indptr_name = file_name + '_indptr.dat'
    indices_name = file_name + '_indices.dat'
    conf_name = file_name + '_conf.json'

    indptr_mmap = np.memmap(indptr_name, mode='w+', shape=indptr.shape, dtype=indptr.dtype)
    indices_mmap = np.memmap(indices_name, mode='w+', shape=indices.shape, dtype=indices.dtype)

    indptr_mmap[:] = indptr[:]
    indices_mmap[:] = indices[:]

    indptr_mmap.flush()
    indices_mmap.flush()

    mmap_config = dict()
    mmap_config['num_nodes'] = num_nodes
    mmap_config['indptr_shape'] = tuple(indptr.shape)
    mmap_config['indptr_dtype'] = str(indptr.dtype)
    mmap_config['indices_shape'] = tuple(indices.shape)
    mmap_config['indices_dtype'] = str(indices.dtype)

    json.dump(mmap_config, open(conf_name, 'w'))


def main(_):
	indptr_path = base_dir + str(args.dataset) + "/indptr.dat"
	indices_path = base_dir + str(args.dataset) + "/indices.dat"
	conf_path = base_dir + str(args.dataset) + "/conf.json"

	logging.info("Loading mmap from file")
	num_nodes, csr_indptr, csr_indices = load_mmap_from_file(indptr_path, indices_path, conf_path)
	logging.info("Done loading mmap from file")

	logging.info("Saving indices, indptr, conf.json of csr version")
	file_name = base_dir + str(args.dataset) + "/csr"
	save_as_numpy(csr_indptr, csr_indices, num_nodes, file_name)
	logging.info("Done saving indices, indptr, conf.json of csr version")


if __name__ == "__main__":
	logging.set_verbosity(logging.INFO)
	app.run(main)
