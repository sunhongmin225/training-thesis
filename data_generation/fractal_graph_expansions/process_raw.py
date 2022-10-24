import collections
import dgl
import os
import numpy as np
import json
import torch

# data_path = "/home/nvadmin/msh/datasets/twitter/raw/twitter-2010.txt"
# write_file_path = "/home/nvadmin/msh/datasets/twitter/raw/"
# num_nodes = 41652230
# num_edges = 1468364884
# num_lines = 1468365182
data_path = "/home/nvadmin/msh/datasets/friendster/raw/com-friendster.ungraph.txt"
write_file_path = "/home/nvadmin/msh/datasets/friendster/raw/"
num_nodes = 65608366
num_edges = 1806067135
num_lines = 1806067139
# num_lines = 10

def save_graph_as_numpy(g, write_file_path):
    indptr_name = write_file_path + 'indptr.dat'
    indices_name = write_file_path + 'indices.dat'
    conf_name = write_file_path + 'conf.json'

    adj = g.adj_sparse('csc')
    # indptr = adj[0].cpu().numpy()
    # indices = adj[1].cpu().numpy()
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


file = open(data_path, "r")
lines = file.readlines()
line_num = 0

src_ids = list()
dst_ids = list()
node_dict = dict()

print("Start processing file.")

if "friendster" in data_path:
    print("Creating node_dict for friendster")
    node_id = 0
    for line in lines:
        line_num += 1
        if line_num < 5:
            continue
        if line_num % 10000000 == 0:
            print("Processed {} / {} lines.".format(line_num, num_lines))
        src_id, dst_id = line.split()
        if src_id not in node_dict:
            node_dict[src_id] = node_id
            node_id += 1
        if dst_id not in node_dict:
            node_dict[dst_id] = node_id
            node_id += 1

# print(node_dict)
if "friendster" in data_path:
    print("Done creating node_dict for friendster.")

line_num = 0

for line in lines:
    line_num += 1
    if "friendster" in data_path and line_num < 5:
        continue
    if line_num % 10000000 == 0:
        print("Processed {} / {} lines.".format(line_num, num_lines))
    src_id, dst_id = line.split()
    if "friendster" in data_path:
        src_ids.append(node_dict[src_id])
        dst_ids.append(node_dict[dst_id])
    else:
        src_ids.append(int(src_id))
        dst_ids.append(int(dst_id))

assert line_num == num_lines

print("Done processing file.\nConverting list to torch.tensor.")

src_ids = torch.tensor(src_ids)
dst_ids = torch.tensor(dst_ids)

# print(src_ids)
# print(dst_ids)

print("Done converting list to torch.tensor.\nCreating graph using tensor.")

# g = dgl.graph((src_ids, dst_ids), num_nodes=num_nodes)
g = dgl.graph((src_ids, dst_ids))
if "friendster" in data_path:
    g = dgl.to_bidirected(g)

print("Done creating graph using tensor.\nSaving graph as numpy.")

# print(g)
# print(g.edges())
save_graph_as_numpy(g, write_file_path)

print("Done saving graph as numpy.")
