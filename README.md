# Dataset Expansion 하는 방법

1. `git clone https://github.com/sunhongmin225/training-thesis`

2. `cd training-thesis/data_generation/fractal_graph_expansions`

3. `run.sh`에서 expand하고 싶은 target dataset에 따라 적절한 line을 uncomment한다.

* e.g., `twitter`를 expand할 경우의 `run.sh`의 contents:

```
#!/bin/bash

indptr_file="/home/nvadmin/msh/datasets/twitter/raw/indptr.dat" # change directory appropriately
indices_file="/home/nvadmin/msh/datasets/twitter/raw/indices.dat"
conf_file="/home/nvadmin/msh/datasets/twitter/raw/conf.json"
output_prefix="/home/nvadmin/msh/datasets/twitter/twitter_"

num_row_multiplier=6
num_col_multiplier=6

python3 run_expansion.py --indptr_file=$indptr_file --indices_file=$indices_file --conf_file=$conf_file --output_prefix=$output_prefix --num_row_multiplier=$num_row_multiplier --num_col_multiplier=$num_col_multiplier

python3 merge_csr_to_csc.py --row=$num_row_multiplier --col=$num_col_multiplier --prefix=$output_prefix
```

* 마지막 줄의 `python3 merge_cscs_to_csc.py`는 무시
* `num_row_multiplier`와 `num_col_multiplier`은 같은 값을 준다. 이 값이 커질수록 output dataset의 크기가 커진다. `twitter`은 6, `products`는 128을 줬던 것 같고 `papers`, `friendster`는 각각 몇을 줬는지 잘 기억이 나지 않는다. 대강 10 정도의 값을 줘보고 output dataset의 크기를 보고 empirical 하게 늘려가면 된다.

4. `bash run.sh`
