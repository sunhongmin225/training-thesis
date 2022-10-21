import collections
import dgl
import os
import numpy as np
import json
import torch
from six.moves import xrange


def savez_two_column(matrix, row_offset, file_name):
  """Savez_compressed obj to file_name."""
  tc = []
  for u, items in enumerate(matrix):
    user = row_offset + u
    for item in items:
      tc.append([user, item])

  print("tc: {}".format(tc))

  # np.savez_compressed(file_name, np.asarray(tc))


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

    print("indptr: {}".format(indptr))
    print("indices: {}".format(indices))

    # indptr_mmap = np.memmap(indptr_name, mode='w+', shape=indptr.shape, dtype=indptr.dtype)
    # indices_mmap = np.memmap(indices_name, mode='w+', shape=indices.shape, dtype=indices.dtype)

    # indptr_mmap[:] = indptr[:]
    # indices_mmap[:] = indices[:]

    # indptr_mmap.flush()
    # indices_mmap.flush()

    mmap_config = dict()
    mmap_config['num_nodes'] = g.num_nodes()
    mmap_config['indptr_shape'] = tuple(indptr.shape)
    mmap_config['indptr_dtype'] = str(indptr.dtype)
    mmap_config['indices_shape'] = tuple(indices.shape)
    mmap_config['indices_dtype'] = str(indices.dtype)

    # json.dump(mmap_config, open(conf_name, 'w'))

    print("mmap_config: {}".format(mmap_config))


all_items_to_write = []
all_items_to_write.append([0, 1, 2, 4])
all_items_to_write.append([2, 6, 7, 8])

g = to_dgl_graph(all_items_to_write, row_offset=0)
print("g.adj_sparse('csc'): {}".format(g.adj_sparse('csc')))
file_name = "sample"
save_graph_as_numpy(g, file_name)

file_name = "sample.npz"
savez_two_column(all_items_to_write, row_offset=0, file_name=file_name)

a = []
a += [0, 2, 3, 6]
print(a)

b = [0, 1, 2, 5]
c = [0, 3, 5, 6]
last = a[-1]
b = list(map(lambda x : x + last, b))
a += b[1:]
last = a[-1]
c = list(map(lambda x : x + last, c))
a += c[1:]
print(a)
