#!/bin/bash

# indptr_file="/data/nvme1n1p1/msh/synthetic/ogbn_papers/raw/indptr.dat"
# indices_file="/data/nvme1n1p1/msh/synthetic/ogbn_papers/raw/indices.dat"
# conf_file="/data/nvme1n1p1/msh/synthetic/ogbn_papers/raw/conf.json"
# output_prefix="/data/nvme1n1p1/msh/synthetic/ogbn_papers/ogbn_papers_"

indptr_file="/data/nvme1n1p1/msh/synthetic/ogbn_products/raw/indptr.dat"
indices_file="/data/nvme1n1p1/msh/synthetic/ogbn_products/raw/indices.dat"
conf_file="/data/nvme1n1p1/msh/synthetic/ogbn_products/raw/conf.json"
output_prefix="/data/nvme1n1p1/msh/synthetic/ogbn_products/ogbn_products_"

# indptr_file="/data/nvme1n1p1/msh/synthetic/reddit/raw/indptr.dat"
# indices_file="/data/nvme1n1p1/msh/synthetic/reddit/raw/indices.dat"
# conf_file="/data/nvme1n1p1/msh/synthetic/reddit/raw/conf.json"
# output_prefix="/data/nvme1n1p1/msh/synthetic/reddit/reddit_"

num_row_multiplier=128
num_col_multiplier=128

python3 run_expansion.py --indptr_file=$indptr_file --indices_file=$indices_file --conf_file=$conf_file --output_prefix=$output_prefix --num_row_multiplier=$num_row_multiplier --num_col_multiplier=$num_col_multiplier

python3 merge_csr_to_csc.py --row=$num_row_multiplier --col=$num_col_multiplier --prefix=$output_prefix