#!/bin/bash

# indptr_file="/data/nvme1n1p1/msh/synthetic/ogbn_papers/raw/indptr.dat"
# indices_file="/data/nvme1n1p1/msh/synthetic/ogbn_papers/raw/indices.dat"
# conf_file="/data/nvme1n1p1/msh/synthetic/ogbn_papers/raw/conf.json"
# output_prefix="/data/nvme1n1p1/msh/synthetic/ogbn_papers/ogbn_papers_"

# indptr_file="/data/nvme1n1p1/msh/synthetic/ogbn_products/raw/indptr.dat"
# indices_file="/data/nvme1n1p1/msh/synthetic/ogbn_products/raw/indices.dat"
# conf_file="/data/nvme1n1p1/msh/synthetic/ogbn_products/raw/conf.json"
# output_prefix="/data/nvme1n1p1/msh/synthetic/ogbn_products/ogbn_products_"

indptr_file="/data/nvme1n1p1/msh/synthetic/friendster/raw/indptr.dat"
indices_file="/data/nvme1n1p1/msh/synthetic/friendster/raw/indices.dat"
conf_file="/data/nvme1n1p1/msh/synthetic/friendster/raw/conf.json"
output_prefix="/data/nvme1n1p1/msh/synthetic/friendster/friendster_"

# ogbn_products
# 30 => 4,854,132,381
# 50 => 8,491,364,056
# 55 => 11,216,798,343
# 64 => 15,606,468,085
# 80 => 17,102,635,685
# 90 => 20,243,659,298
num_row_multiplier=10
num_col_multiplier=10

python3 run_expansion.py --indptr_file=$indptr_file --indices_file=$indices_file --conf_file=$conf_file --output_prefix=$output_prefix --num_row_multiplier=$num_row_multiplier --num_col_multiplier=$num_col_multiplier

# python3 merge_csr_to_csc.py --row=$num_row_multiplier --col=$num_col_multiplier --prefix=$output_prefix