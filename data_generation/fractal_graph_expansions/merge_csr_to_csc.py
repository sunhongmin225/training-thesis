from absl import logging
import argparse
import numpy as np
import pandas as pd
import json
from scipy import sparse

parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--row', type=int, default=2, help='num_row_multiplier used to run run_expansion.py')
parser.add_argument('--col', type=int, default=4, help='num_col_multiplier used to run run_expansion.py')
parser.add_argument('--prefix', type=str, default='ogbn_', help='prefix of dat and json files')
args = parser.parse_args()

assert args.row >= 1
assert args.col >= 1
assert args.prefix != None

if "ogbn_papers" in args.prefix:
    base_num_nodes = 111059956
elif "ogbn_products" in args.prefix:
    base_num_nodes = 2449029
elif "reddit" in args.prefix:
    base_num_nodes = 232965

# data_path = "./data/"
logging.set_verbosity(logging.INFO)


def save_array_as_numpy(indptr, indices, row_num_nodes, col_num_nodes, file_name):
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
    mmap_config['row_num_nodes'] = row_num_nodes
    mmap_config['col_num_nodes'] = col_num_nodes
    mmap_config['indptr_shape'] = tuple(indptr.shape)
    mmap_config['indptr_dtype'] = str(indptr.dtype)
    mmap_config['indices_shape'] = tuple(indices.shape)
    mmap_config['indices_dtype'] = str(indices.dtype)

    json.dump(mmap_config, open(conf_name, 'w'))


def load_mmap_from_file(indptr_path, indices_path, conf_path):
    conf = json.load(open(conf_path, 'r'))

    indptr = np.memmap(indptr_path, mode='r', shape=tuple(conf['indptr_shape']), dtype=conf['indptr_dtype'])
    indices = np.memmap(indices_path, mode='r', shape=tuple(conf['indices_shape']), dtype=conf['indices_dtype'])
    num_nodes = conf['num_nodes']

    data = np.ones(conf['indices_shape'][0])
    matrix = sparse.csr_matrix((data, indices, indptr), shape=(num_nodes, num_nodes * args.col))

    return matrix


indices_files = []
indptr_files = []
conf_files = []

for r in range(args.row):
    # indices_file = data_path + args.prefix + "x" + str(args.row) + "x" + str(args.col) + "_csr_" + str(r) + "_indices.dat"
    # indptr_file = data_path + args.prefix + "x" + str(args.row) + "x" + str(args.col) + "_csr_" + str(r) + "_indptr.dat"
    # conf_file = data_path + args.prefix + "x" + str(args.row) + "x" + str(args.col) + "_csr_" + str(r) + "_conf.json"
    indices_file = args.prefix + "x" + str(args.row) + "x" + str(args.col) + "_csr_" + str(r) + "_indices.dat"
    indptr_file = args.prefix + "x" + str(args.row) + "x" + str(args.col) + "_csr_" + str(r) + "_indptr.dat"
    conf_file = args.prefix + "x" + str(args.row) + "x" + str(args.col) + "_csr_" + str(r) + "_conf.json"
    indices_files.append(indices_file)
    indptr_files.append(indptr_file)
    conf_files.append(conf_file)

indptr_to_write = []
indices_to_write = []

for r in range(args.row):
    print("indptr_files[{}]: {}, indices_files[{}]: {}, conf_files[{}]: {}".format(r, indptr_files[r], r, indices_files[r], r, conf_files[r]))
    logging.info("Loading mmap from file")
    matrix = load_mmap_from_file(indptr_files[r], indices_files[r], conf_files[r])
    logging.info("Done loading mmap from file")

    if (r == 0):
        logging.info("Adding to indptr_to_write")
        indptr_to_write += matrix.indptr.tolist()
        logging.info("Done adding to indptr_to_write")
    else:
        logging.info("Adding to indptr_to_write")
        last_indptr = indptr_to_write[-1]
        curr_indptr = list(map(lambda x : x + last_indptr, matrix.indptr.tolist()))
        indptr_to_write += curr_indptr[1:]
        logging.info("Done adding to indptr_to_write")

    logging.info("Adding to indices_to_write")
    indices_to_write += matrix.indices.tolist()
    logging.info("Done adding to indices_to_write")

logging.info("Done processing.")
logging.info("Merging to one.")
indptr_to_write = np.array(indptr_to_write)
indices_to_write = np.array(indices_to_write)
indptr_row = indptr_to_write.shape[0]
indices_row = indices_to_write.shape[0]
print("indptr_row: {}".format(indptr_row))
print("indices_row: {}".format(indices_row))
data_to_write = [1] * indices_row
data_to_write = np.array(data_to_write)
row_num_nodes = indptr_row - 1
assert row_num_nodes == base_num_nodes * args.row
col_num_nodes = base_num_nodes * args.col
# print("csr indptr_to_write.shape: {}, csr indptr_to_write: {}".format(indptr_to_write.shape, indptr_to_write))
# print("csr indices_to_write.shape: {}, csr indices_to_write: {}".format(indices_to_write.shape, indices_to_write))
# print("csr data_to_write.shape: {}, csr data_to_write: {}".format(data_to_write.shape, data_to_write))
print("row_num_nodes: {}, col_num_nodes: {}".format(row_num_nodes, col_num_nodes))
logging.info("Making a single csr matrix out of given information.")
my_csr = sparse.csr_matrix((data_to_write, indices_to_write, indptr_to_write), shape=(row_num_nodes, col_num_nodes))
logging.info("Done making a single csr matrix out of given information.")
logging.info("Making a csc matrix out of csr matrix.")
my_csc = sparse.csc_matrix(my_csr)
logging.info("Done making a csc matrix out of csr matrix.")
# print("csc indptr_to_write.shape: {}, csc indptr_to_write: {}".format(my_csc.indptr.shape, my_csc.indptr))
# print("csc indices_to_write.shape: {}, csc indices_to_write: {}".format(my_csc.indices.shape, my_csc.indices))
# print("csc data_to_write.shape: {}, csc data_to_write: {}".format(my_csc.data.shape, my_csc.data))
logging.info("Saving array as numpy object (uncompressed).")
# file_name = data_path + args.prefix + "x" + str(args.row) + "x" + str(args.col) + "_csc"
file_name = args.prefix + "x" + str(args.row) + "x" + str(args.col) + "_csc"
save_array_as_numpy(my_csc.indptr, my_csc.indices, row_num_nodes=row_num_nodes, col_num_nodes=col_num_nodes, file_name=file_name)
logging.info("Done saving array as numpy object (uncompressed).\n\n")
