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

base_dir = "/data/nvme1n1p1/msh/synthetic/"

def load_mmap_from_file(indptr_path, indices_path, conf_path):
	conf = json.load(open(conf_path, 'r'))
	
	indptr = torch.from_numpy(np.memmap(indptr_path, mode='r', shape=tuple(conf['indptr_shape']), dtype=conf['indptr_dtype']))
	indices = torch.from_numpy(np.memmap(indices_path, mode='r', shape=tuple(conf['indices_shape']), dtype=conf['indices_dtype']))
	num_nodes = conf['num_nodes']

	data = ('csc', (indptr, indices, []))
	g = dgl.graph(data, num_nodes=num_nodes)
	g = g.formats('csr')

	return num_nodes, g.adj_sparse('csr')[0].numpy()

def save_numpy(sorted_nodes, num_nodes):
    sorted_nodes_name = base_dir + str(args.dataset) + "/sorted_nodes.dat"
    conf_name = base_dir + str(args.dataset) + "/sorted_nodes_conf.json"

    sorted_nodes_mmap = np.memmap(sorted_nodes_name, mode='w+', shape=sorted_nodes.shape, dtype=sorted_nodes.dtype)

    sorted_nodes_mmap[:] = sorted_nodes[:]

    sorted_nodes_mmap.flush()

    mmap_config = dict()
    mmap_config['num_nodes'] = num_nodes
    mmap_config['sorted_nodes_shape'] = tuple(sorted_nodes.shape)
    mmap_config['sorted_nodes_dtype'] = str(sorted_nodes.dtype)

    json.dump(mmap_config, open(conf_name, 'w'))

def main(_):
	indptr_path = base_dir + str(args.dataset) + "/indptr.dat"
	indices_path = base_dir + str(args.dataset) + "/indices.dat"
	conf_path = base_dir + str(args.dataset) + "/conf.json"

	logging.info("Loading mmap from file")
	num_nodes, csr_indptr = load_mmap_from_file(indptr_path, indices_path, conf_path)
	logging.info("Done loading mmap from file")

	node_to_num_outgoing_edges_dict = dict()

	logging.info("Creating node : num_outgoing_edges dictionary")
	for node in range(num_nodes):
		if node % 10000000 == 0:
			logging.info("Processed %d nodes out of %d nodes (%2f percent)", node, num_nodes, float(node) / float(num_nodes) )
		node_to_num_outgoing_edges_dict[node] = csr_indptr[node + 1] - csr_indptr[node]
	logging.info("Done creating node : num_outgoing_edges dictionary")

	logging.info("Sorting dictionary")
	sorted_node_to_num_outgoing_edges_dict = dict(sorted(node_to_num_outgoing_edges_dict.items(), reverse=True, key=lambda item: item[1]))
	logging.info("Done sorting dictionary")

	logging.info("Converting dictionary to np.array")
	sorted_keys = np.array(list(sorted_node_to_num_outgoing_edges_dict.keys()))
	logging.info("Done converting dictionary to np.array")

	# print(type(sorted_keys))
	# print(sorted_keys.shape)
	# print(sorted_keys.dtype)
	# print(sorted_keys)

	logging.info("Saving sorted_nodes.dat and sorted_nodes_conf.json")
	save_numpy(sorted_keys, num_nodes)
	logging.info("Done saving sorted_nodes.dat and sorted_nodes_conf.json")

	sorted_nodes_path = base_dir + str(args.dataset) + "/sorted_nodes.dat"
	conf_path = base_dir + str(args.dataset) + "/sorted_nodes_conf.json"
	conf = json.load(open(conf_path, 'r'))
	sorted_nodes = np.memmap(sorted_nodes_path, mode='r', shape=tuple(conf['sorted_nodes_shape']), dtype=conf['sorted_nodes_dtype'])

	print(sorted_nodes)

if __name__ == "__main__":
	logging.set_verbosity(logging.INFO)
	app.run(main)
