DATASET=${DATASET:-ml-20m}
USER_MUL=${USER_MUL:-3}
ITEM_MUL=${ITEM_MUL:-4}
DATA_DIR="./data"

DATA_PATH=${DATA_DIR}/${DATASET}x${USER_MUL}x${ITEM_MUL}/

mkdir -p ${DATA_PATH}
python3 run_expansion.py --input_csv_file ${DATA_DIR}/${DATASET}/ratings.csv --num_row_multiplier ${USER_MUL} --num_col_multiplier ${ITEM_MUL} --output_prefix ${DATA_PATH}
