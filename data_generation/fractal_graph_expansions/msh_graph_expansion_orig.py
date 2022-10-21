# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Fractal graph expander.

Detailed analysis in the deterministic case provided in
https://arxiv.org/abs/1901.08910.
Please refer the paper if you use this code.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import dgl
import os
import numpy as np
import json
import torch

from six.moves import xrange

from absl import logging

from scipy import sparse

import util
from random_matrix_ops import shuffle_sparse_coo_matrix

def to_dgl_graph(all_items_to_write, row_offset=0):

  src_ids_list = []
  dst_ids_list = []

  for src_id in xrange(len(all_items_to_write)):
    for dst_id in all_items_to_write[src_id]:
      src_ids_list.append(src_id + row_offset)
      dst_ids_list.append(dst_id)

  src_ids = torch.tensor(src_ids_list)
  dst_ids = torch.tensor(dst_ids_list)
  g = dgl.graph((src_ids, dst_ids), idtype=torch.int64, device='cuda:0')
  return g


def save_graph_as_numpy(g, file_name):
    indptr_name = file_name + '_indptr.dat'
    indices_name = file_name + '_indices.dat'
    conf_name = file_name + '_conf.json'

    adj = g.adj_sparse('csc')
    indptr = adj[0].cpu().numpy()
    indices = adj[1].cpu().numpy()

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


def save_array_as_numpy(indptr, indices, num_nodes, file_name):
    indptr_name = file_name + '_indptr.dat'
    indices_name = file_name + '_indices.dat'
    conf_name = file_name + '_conf.json'

    # adj = g.adj_sparse('csc')
    # indptr = adj[0].cpu().numpy()
    # indices = adj[1].cpu().numpy()

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


SparseMatrixMetadata = collections.namedtuple(
    "SparseMatrixMetadata", [
        "num_interactions",
        "num_rows",
        "num_cols",
    ])


def output_randomized_kronecker_to_pickle(left_matrix, right_matrix, indices_out_path, metadata_out_path=None, to_npz=True, remove_empty_rows=True):
  """Compute randomized Kronecker product and dump it on the fly.

  A standard Kronecker product between matrices A and B produces
                        [[a_11 B, ..., a_1n B],
                                  ...
                         [a_m1 B, ..., a_mn B]]
    (if A's size is (m, n) and B's size is (p, q) then A Kronecker B has size
    (m p, n q)).
    Here we modify the standard Kronecker product expanding matrices in
    https://cs.stanford.edu/~jure/pubs/kronecker-jmlr10.pdf
    and randomize each block-wise operation a_ij B in the Kronecker product as
    in https://arxiv.org/pdf/1901.08910.pdf section III.4.
    The matrix we produce is
                       [[F(a_11, B, w_11), ..., F(a_1n, B, w_1n)],
                                           ...
                       [F(a_m1, B, w_m1), ... , F(a_mn, B, w_mn)]]
    where (w_ij) is a sequence of pseudo random numbers and F is randomized
    operator which will:
      1) Shuffle rows and columns of B independently at random;
      2) Dropout elements of B with a rate 1 - a_ij to compute
        F(a_ij, B, w_ij).
    (It is noteworthy that there is an abuse of notation above when writing
    F(a_ij, B, w_ij) as each block-wise operation will in fact consume
    multiple elements of the sequence (w_ij)).

  Args:
    left_matrix: sparse SciPy csr matrix with values in [0, 1].
    right_matrix: sparse SciPy coo signed binary matrix. +1 values correspond
      to train set and -1 values correspond to test set.
    train_indices_out_path: path to output train file. The non zero indices of
      the resulting sparse matrix are dumped as a series of pickled records.
      As many shard will be created as there are rows in left matrix. The shard
      corresponding to row i in the left matrix has the suffix _i appended to
      its file name. Each shard contains a pickled list of list each of which
      corresponds to a users.
    test_indices_out_path: path to output train file. The non zero indices of
      the resulting sparse matrix are dumped as a series of pickled records.
      As many shard will be created as there are rows in left matrix. The shard
      corresponding to row i in the left matrix has the suffix _i appended to
      its file name. Each shard contains a pickled list of list each of which
      corresponds to a users.
    train_metadata_out_path: path to optional complementary output file
      containing the number of train rows (r), columns (c) and non zeros (nnz)
      in a pickled SparseMatrixMetadata named tuple.
    test_metadata_out_path: path to optional complementary output file
      containing the number of test rows (r), columns (c) and non zeros (nnz)
      in a pickled SparseMatrixMetadata named tuple.
    remove_empty_rows: whether to remove rows from the synthetic train and
      test matrices which are not present in the train or the test matrix.

  Returns:
    (metadata, train_metadata, test_metadata) triplet of SparseMatrixMetadata
      corresponding to the overall data set, train data set and test data set.
  """
  logging.info("Writing item sequences to pickle files %s.", indices_out_path)

  num_rows_total = 0
  num_removed_rows_total = 0
  num_cols_total = left_matrix.shape[1] * right_matrix.shape[1]
  num_interactions_total = 0

  print("msh::left_matrix.shape: {}".format(left_matrix.shape))
  print("msh::right_matrix.shape: {}".format(right_matrix.shape))

  # num_train_interactions = 0
  # num_test_interactions = 0

  if not set(right_matrix.data).issubset({-1, 1}):
    raise ValueError(
        "Values of sparse matrix should be -1 or 1 but are:",
        set(right_matrix.data))

  # all_items_to_write = []

  indptr_to_write = []
  indices_to_write = []

  for i in xrange(left_matrix.shape[0]):

    kron_blocks = []

    num_rows = 0
    num_removed_rows = 0
    num_interactions = 0

    # Construct blocks
    for j in xrange(left_matrix.shape[1]):

      dropout_rate = 1.0 - left_matrix[i, j]
      kron_block = shuffle_sparse_coo_matrix(right_matrix, dropout_rate)

      # if not set(kron_block.data).issubset({1}):
      #   raise ValueError("Values of sparse matrix should be 1 but are: ",
      #                    set(kron_block.data))

      kron_blocks.append(kron_block)

      logging.info("Done with element (%d, %d)", i, j)

    rows_to_write = sparse.hstack(kron_blocks).tocoo()
    # print("rows_to_write.indptr: {}".format(rows_to_write.indptr))
    # print("rows_to_write.indices: {}".format(rows_to_write.indices))

    train_rows_to_write = util.sparse_where_equal(rows_to_write, 1)
    # train_rows_to_write = util.sparse_where_equal_csc(rows_to_write, 1)
    print("train_rows_to_write.shape: {}, train_rows_to_write: \n{}".format(train_rows_to_write.shape, train_rows_to_write))
    print("train_rows_to_write.indptr.shape: {}, train_rows_to_write.indptr: {}".format(train_rows_to_write.indptr.shape, train_rows_to_write.indptr))
    print("train_rows_to_write.indices.shape: {}, train_rows_to_write.indices: {}".format(train_rows_to_write.indices.shape, train_rows_to_write.indices))
    print("train_rows_to_write.data.shape: {}, train_rows_to_write.data: {}".format(train_rows_to_write.data.shape, train_rows_to_write.data))

    if (i == 0):
      indptr_to_write += train_rows_to_write.indptr.tolist()
    else:
      last_indptr = indptr_to_write[-1]
      curr_indptr = list(map(lambda x : x + last_indptr, train_rows_to_write.indptr.tolist()))
      indptr_to_write += curr_indptr[1:]

    indices_to_write += train_rows_to_write.indices.tolist()

    # logging.info("Producing data set row by row")

    # all_items_to_write = []

    # Write Kronecker product line per line.
    # for k in xrange(right_matrix.shape[0]):

    #   train_items_to_write = train_rows_to_write.getrow(k).indices

    #   num_train = train_items_to_write.shape[0]

    #   if remove_empty_rows and (not num_train):
    #     # logging.info("Removed empty output row %d.", i * left_matrix.shape[0] + k)
    #     num_removed_rows += 1
    #     continue

    #   num_rows += 1
    #   num_interactions += num_train

    #   all_items_to_write.append(train_items_to_write)

    #   if k % 1000000 == 0:
    #     logging.info("Done producing data set row %d.", k)

    # logging.info("Done producing data set row by row.")

    # if (i == left_matrix.shape[0] - 1):
    #   if (to_npz):
    #     util.savez_two_column(all_items_to_write, row_offset=i * right_matrix.shape[0], file_name=indices_out_path + ("_%d" % i))
    #   else:
    #     g = to_dgl_graph(all_items_to_write, row_offset=i * right_matrix.shape[0])
    #     save_graph_as_numpy(g, file_name=indices_out_path + ("_%d" % i))

    # num_cols = rows_to_write.shape[1]
    # metadata = SparseMatrixMetadata(num_interactions=num_interactions,
    #                                 num_rows=num_rows, num_cols=num_cols)

    # logging.info("Done with left matrix row %d.", i)
    # logging.info("%d interactions written in shard.", num_interactions)
    # logging.info("%d rows removed in shard.", num_removed_rows)

    # num_rows_total += metadata.num_rows
    # num_removed_rows_total += num_removed_rows_total
    # num_interactions_total += metadata.num_interactions

    # logging.info("%d total interactions written.", num_interactions_total)
    # logging.info("%d total rows removed.", num_removed_rows_total)

  # logging.info("Done writing.")
  logging.info("Done processing.")
  indptr_to_write = np.array(indptr_to_write)
  indices_to_write = np.array(indices_to_write)
  indptr_row = indptr_to_write.shape[0]
  print("indptr_row: {}".format(indptr_row))
  indices_row = indices_to_write.shape[0]
  print("indices_row: {}".format(indices_row))
  data_to_write = [1] * indices_row
  data_to_write = np.array(data_to_write)
  num_nodes = indptr_row - 1
  print("csr indptr_to_write.shape: {}, csr indptr_to_write: {}".format(indptr_to_write.shape, indptr_to_write))
  print("csr indices_to_write.shape: {}, csr indices_to_write: {}".format(indices_to_write.shape, indices_to_write))
  print("csr data_to_write.shape: {}, csr data_to_write: {}".format(data_to_write.shape, data_to_write))
  print("num_nodes: {}".format(num_nodes))
  logging.info("Making a single csr matrix out of given information.")
  my_csr = sparse.csr_matrix((data_to_write, indices_to_write, indptr_to_write))
  logging.info("Done making a single csr matrix out of given information.")
  logging.info("Making a csc matrix out of csr matrix.")
  my_csc = sparse.csc_matrix(my_csr)
  logging.info("Done making a csc matrix out of csr matrix.")
  print("csc indptr_to_write.shape: {}, csc indptr_to_write: {}".format(my_csc.indptr.shape, my_csc.indptr))
  print("csc indices_to_write.shape: {}, csc indices_to_write: {}".format(my_csc.indices.shape, my_csc.indices))
  print("csc data_to_write.shape: {}, csc data_to_write: {}".format(my_csc.data.shape, my_csc.data))

  logging.info("Saving array as numpy object (uncompressed).")
  save_array_as_numpy(my_csc.indptr, my_csc.indices, num_nodes=num_nodes, file_name=indices_out_path)
  logging.info("Done saving array as numpy object (uncompressed).")

  # print("len(indptr_to_write): {}, indptr_to_write: [{}, {}, {}, ..., {}, {}, {}]".format(len(indptr_to_write), indptr_to_write[0], indptr_to_write[1], indptr_to_write[2], indptr_to_write[-3], indptr_to_write[-2], indptr_to_write[-1]))
  # print("len(indices_to_write): {}, indices_to_write: [{}, {}, {}, ..., {}, {}, {}]".format(len(indices_to_write), indices_to_write[0], indices_to_write[1], indices_to_write[2], indices_to_write[-3], indices_to_write[-2], indices_to_write[-1]))

  num_interactions_total = 0
  num_rows_total = 0
  metadata = SparseMatrixMetadata(
      num_interactions=num_interactions_total,
      num_rows=num_rows_total, num_cols=num_cols_total)

  if metadata_out_path is not None:
    util.write_metadata_to_file(
        metadata, metadata_out_path, tag="all")

  return metadata
