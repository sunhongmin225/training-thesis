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
"""Run fractal expansion of MovieLens 20m.

Detailed analysis of the deterministic case provided in
https://arxiv.org/abs/1901.08910.
Please refer the paper if you use this code.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from absl import app
from absl import flags
from absl import logging

import numpy as np

import util
from graph_analysis import sparse_svd
from graph_expansion import output_randomized_kronecker_to_pickle
from graph_reduction import normalize_matrix
from graph_reduction import resize_matrix


flags.DEFINE_string("indptr_file",
                    "none",
                    "indptr file")
flags.DEFINE_string("indices_file",
                    "none",
                    "indices file")
flags.DEFINE_string("conf_file",
                    "none",
                    "conf file")
flags.DEFINE_integer("num_row_multiplier",
                     16,
                     "Factor by which the number of rows in the rating "
                     "matrix will be multiplied.")
flags.DEFINE_integer("num_col_multiplier",
                     32,
                     "Factor by which the number of columns in the rating "
                     "matrix will be multiplied.")
flags.DEFINE_string("output_prefix",
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


def _create_index(df, colname):
  value_set = sorted(set(df[colname].values))
  num_unique = len(value_set)
  return dict(zip(value_set, range(num_unique)))


def _create_row_col_indices(ratings_df):
  """Maps user and items ids to their locations in the rating matrix."""
  user_id_to_user_idx = _create_index(ratings_df, "userId")
  item_id_to_item_idx = _create_index(ratings_df, "movieId")

  ratings_df["row"] = ratings_df["userId"].apply(
      lambda x: user_id_to_user_idx[x])
  ratings_df["col"] = ratings_df["movieId"].apply(
      lambda x: item_id_to_item_idx[x])

  return ratings_df

def main(_):

  # Fix seed for reproducibility
  np.random.seed(FLAGS.random_seed)

  logging.info("Loading files...")
  logging.info("indptr_file: %s", FLAGS.indptr_file)
  logging.info("indices_file: %s", FLAGS.indices_file)
  logging.info("conf_file: %s", FLAGS.conf_file)
  matrix = util.load_mmap_from_file(FLAGS.indptr_file, FLAGS.indices_file, FLAGS.conf_file)
  logging.info("Done loading files")

  reduced_num_rows = FLAGS.num_row_multiplier
  reduced_num_cols = FLAGS.num_col_multiplier
  k = min(reduced_num_rows, reduced_num_cols)
  logging.info("Computing SVD of training matrix (top %d values).", k)
  (u_train, s_train, v_train) = sparse_svd(
      matrix, k, max_iter=None)
  logging.info("Done computing SVD of training matrix.")

  logging.info("Creating reduced rating matrix (size %d, %d)", reduced_num_rows,
               reduced_num_cols)
  reduced_matrix = resize_matrix(
      (u_train, s_train, v_train), reduced_num_rows, reduced_num_cols)
  logging.info("Normalizing reduced rating matrix (size %d, %d)", reduced_num_rows,
               reduced_num_cols)
  reduced_matrix = normalize_matrix(reduced_matrix)
  logging.info("Normalizing and creating reduced rating matrix done.")

  average_sampling_rate = reduced_matrix.mean()
  logging.info("Average sampling rate: %2f.", average_sampling_rate)
  logging.info("Expected number of synthetic train samples: %s",
               average_sampling_rate * num_train_ratings)
  logging.info("Expected number of synthetic test samples: %s",
               average_sampling_rate * num_test_ratings)

  # Mark test data by a bit flip.
  logging.info("Creating coo matrix.")
  matrix = matrix.tocoo()
  logging.info("Done creating coo matrix.")

  output_file = (FLAGS.output_prefix + "x" + 
      str(reduced_num_rows) + "x" + str(reduced_num_cols))
  output_file_metadata = None

  logging.info("Creating synthetic data set and dumping to %s.",
               output_file)
  output_randomized_kronecker_to_pickle(
      left_matrix=reduced_train_matrix,
      right_matrix=matrix,
      train_indices_out_path=output_file,
      train_metadata_out_path=output_file_metadata)
  logging.info("Done creating synthetic data set and dumping to %s.",
               output_file)


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  app.run(main)
