from absl import app
from absl import flags
from absl import logging
import argparse
import numpy as np
import pandas as pd
import json
from scipy import sparse

parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--row', type=int, default=4, help='num_row_multiplier used to run run_expansion.py')
parser.add_argument('--col', type=int, default=4, help='num_col_multiplier used to run run_expansion.py')
parser.add_argument('--prefix', type=str, default='ogbn_', help='prefix of dat and json files')
args = parser.parse_args()

assert args.row >= 1
assert args.col >= 1
assert args.prefix != None

flags.DEFINE_integer("row",
                     4,
                     "Factor by which the number of rows in the rating "
                     "matrix will be multiplied.")
flags.DEFINE_integer("col",
                     4,
                     "Factor by which the number of columns in the rating "
                     "matrix will be multiplied.")
flags.DEFINE_string("prefix",
                    "",
                    "Prefix to the path of the files that will be "
                    "produced. output_prefix/trainxAxB_C.npz and "
                    "output_prefix/testxAxB_C.npz will be created, "
                    "where A is num_row_multiplier, B is num_col_multiplier, "
                    "and C goes from 0 to (num_row_multiplier - 1).")
flags.DEFINE_integer("random_seed",
                     0,
                     "Random seed for all random operations.")

FLAGS = flags.FLAGS


if "ogbn_papers" in args.prefix:
    base_num_nodes = 111059956
elif "ogbn_products" in args.prefix:
    base_num_nodes = 2449029
elif "twitter" in args.prefix:
    base_num_nodes = 41652230
elif "friendster" in args.prefix:
    base_num_nodes = 65608366
else:
    base_num_nodes = 2

# logging.set_verbosity(logging.INFO)


def save_indptr_as_numpy(indptr, file_name):
    indptr_name = file_name + '_indptr.dat'
    indptr_mmap = np.memmap(indptr_name, mode='w+', shape=indptr.shape, dtype=indptr.dtype)
    indptr_mmap[:] = indptr[:]
    indptr_mmap.flush()
    return indptr.shape, indptr.dtype


def save_indices_as_numpy(indices, file_name):
    indices_name = file_name + '_indices.dat'
    indices_mmap = np.memmap(indices_name, mode='w+', shape=indices.shape, dtype=indices.dtype)
    indices_mmap[:] = indices[:]
    indices_mmap.flush()
    return indices.shape, indices.dtype


def save_conf(num_nodes, indptr_shape, indptr_dtype, indices_shape, indices_dtype, file_name):
    conf_name = file_name + '_conf.json'
    mmap_config = dict()
    mmap_config['num_nodes'] = num_nodes
    mmap_config['indptr_shape'] = tuple(indptr_shape)
    mmap_config['indptr_dtype'] = str(indptr_dtype)
    mmap_config['indices_shape'] = tuple(indices_shape)
    mmap_config['indices_dtype'] = str(indices_dtype)

    json.dump(mmap_config, open(conf_name, 'w'))


def save_array_as_numpy(indptr, indices, num_nodes, file_name):
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
    # Fix seed for reproducibility
    np.random.seed(FLAGS.random_seed)

    file_name = args.prefix + "x" + str(args.row) + "x" + str(args.col) + "_csc"
    conf_files = list()
    indptr_files = list()
    indices_files = list()

    for r in range(args.row):
        conf_file = args.prefix + "x" + str(args.row) + "x" + str(args.col) + "_csc_" + str(r) + "_conf.json"
        indptr_file = args.prefix + "x" + str(args.row) + "x" + str(args.col) + "_csc_" + str(r) + "_indptr.dat"
        indices_file = args.prefix + "x" + str(args.row) + "x" + str(args.col) + "_csc_" + str(r) + "_indices.dat"
        conf_files.append(conf_file)
        indptr_files.append(indptr_file)
        indices_files.append(indices_file)

    indptr_list = list()
    indices_list = list()

    for r in range(args.row):
        print("indptr_files[{}]: {}, indices_files[{}]: {}, conf_files[{}]: {}".format(r, indptr_files[r], r, indices_files[r], r, conf_files[r]))
        logging.info("Loading mmap from file")
        curr_conf = json.load(open(conf_files[r], 'r'))
        curr_indptr = np.memmap(indptr_files[r], mode='r', shape=tuple(curr_conf['indptr_shape']), dtype=curr_conf['indptr_dtype'])
        curr_indices = np.memmap(indices_files[r], mode='r', shape=tuple(curr_conf['indices_shape']), dtype=curr_conf['indices_dtype'])
        logging.info("Done loading mmap from file")
        logging.info("Appending curr_indptr to indptr_list")
        indptr_list.append(curr_indptr)
        indices_list.append(curr_indices + base_num_nodes * r)
        logging.info("Done appending curr_indptr to indptr_list")

    del(conf_files)
    del(indptr_files)
    del(indices_files)

    logging.info("Summing indptr_list to my_indptr")
    my_indptr = sum(indptr_list)
    logging.info("Done summing indptr_list to my_indptr")

    logging.info("Saving my_indptr as numpy object (uncompressed).")
    indptr_shape, indptr_dtype = save_indptr_as_numpy(my_indptr, file_name)
    print("indptr_shape: {}, indptr_dtype: {}".format(indptr_shape, indptr_dtype))
    logging.info("Done saving my_indptr as numpy object.")

    del(my_indptr)

    my_indices_structure = [[] for i in range(base_num_nodes * args.row)]
    logging.info("Creating my_indices_structrue")
    processed = 0
    for i in range(base_num_nodes * args.row):
        processed += 1
        if processed % 10000000 == 0:
            logging.info("Processed %d / %d (%2f percent) nodes", processed, base_num_nodes * args.row, 100 * processed / (base_num_nodes * args.row))
        for r in range(args.row):
            curr_indices = indices_list[r][indptr_list[r][i] : indptr_list[r][i+1]].tolist()
            if len(curr_indices) != 0:
                for curr_index in curr_indices:
                    my_indices_structure[i].append(curr_index)
            del(curr_indices)

    logging.info("Done creating my_indices_structrue")

    del(indptr_list)
    del(indices_list)

    my_indices = np.array([]).astype(int)
    processed = 0
    logging.info("Creating my_indices")
    for i in range(base_num_nodes * args.row):
        processed += 1
        if processed % 10000000 == 0:
            logging.info("Processed %d / %d (%2f percent) nodes", processed, base_num_nodes * args.row, 100 * processed / (base_num_nodes * args.row))
        for my_index in my_indices_structure[i]:
            my_indices = np.append(my_indices, [my_index])
    logging.info("Done creating my_indices")

    del(my_indices_structure)

    logging.info("Saving my_indices as numpy object (uncompressed).")
    indices_shape, indices_dtype = save_indices_as_numpy(my_indices, file_name)
    print("indices_shape: {}, indices_dtype: {}".format(indices_shape, indices_dtype))
    logging.info("Done saving my_indices as numpy object.")

    del(my_indices)

    logging.info("Saving conf json file.")
    save_conf(base_num_nodes * args.row, indptr_shape, indptr_dtype, indices_shape, indices_dtype, file_name)
    logging.info("Done saving conf json file.")


    # print("my csc indptr: {}, type(my_indptr): {}, my_indptr.dtype: {}".format(my_indptr, type(my_indptr), my_indptr.dtype))
    # print("my csc indices: {}, type(my_indices): {}, my_indices.dtype: {}".format(my_indices, type(my_indices), my_indices.dtype))

    # logging.info("Saving array as numpy object (uncompressed).")
    # save_array_as_numpy(my_indptr, my_indices, base_num_nodes * args.row, file_name)
    # logging.info("Done saving array as numpy object.\n\n")


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  app.run(main)
