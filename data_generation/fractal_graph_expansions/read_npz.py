import numpy as np
import pickle

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
read_file(data_name)

read_pickle_file(metadata_name)