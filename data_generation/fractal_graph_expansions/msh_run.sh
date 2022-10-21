#!/bin/bash

indptr_file="../../../dataset/np/ogbn_papers/indptr.dat"
indices_file="../../../dataset/np/ogbn_papers/indices.dat"
conf_file="../../../dataset/np/ogbn_papers/conf.json"
output_prefix="movielens_"
num_row_multiplier=2
num_col_multiplier=4
input_csv_file="./data/ml-20m/ratings.csv"

python3 msh_run_expansion.py --input_csv_file=$input_csv_file --num_row_multiplier=$num_row_multiplier --num_col_multiplier=$num_col_multiplier --output_prefix=$output_prefix


# python3 run_expansion.py --indptr_file=$indptr_file --indices_file=$indices_file --conf_file=$conf_file --output_prefix=$output_prefix --num_row_multiplier=$num_row_multiplier --num_col_multiplier=$num_col_multiplier
