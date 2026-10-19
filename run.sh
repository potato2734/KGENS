#!/usr/bin/env bash
set -e
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"
export PYTHONPATH="$(pwd):$(pwd)/..:${PYTHONPATH:-}"
python -u -c 'import torch; print(torch.__version__)'

CODE_PATH="codes"
DATA_PATH="kg_data"
SAVE_PATH="models"

MODE="$1"
MODEL="$2"
DATASET="$3"
GPU_DEVICE="$4"
SAVE_ID="$5"


BATCH_SIZE="$6"
NEGATIVE_SAMPLE_SIZE="$7"
HIDDEN_DIM="$8"
GAMMA="$9"
ALPHA="${10}"
LEARNING_RATE="${11}"
MAX_STEPS="${12}"
TEST_BATCH_SIZE="${13}"
SQUARE="${14}"
BETA="${15}"
TAU="${16}"
LAMBDA="${17}"
MSSG="${18}"
EXTRA_ARGS=("${@:19}")

FULL_DATA_PATH="${DATA_PATH}/${DATASET}"
if [ "$MSSG" = "" ]; then
SAVE="${SAVE_PATH}/${MODEL}_${DATASET}/${MODEL}_${DATASET}_${SAVE_ID}"
else
SAVE="${SAVE_PATH}/${MODEL}_${DATASET}/${MODEL}_${DATASET}_${SAVE_ID}_${MSSG}"
fi

if [ "$MODE" = "train" ]; then
    echo "Start Training......"

    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" python -u "${CODE_PATH}/run.py" \
        --do_train \
        --cuda \
        --do_valid \
        --do_test \
        --data_path "$FULL_DATA_PATH" \
        --model "$MODEL" \
        -n "$NEGATIVE_SAMPLE_SIZE" \
        -b "$BATCH_SIZE" \
        -d "$HIDDEN_DIM" \
        -g "$GAMMA" \
        -a "$ALPHA" \
        --beta "$BETA" \
        --lam "$LAMBDA" \
        -lr "$LEARNING_RATE" \
        --max_steps "$MAX_STEPS" \
        --test_batch_size "$TEST_BATCH_SIZE" \
        -s "$SQUARE" \
        --seed $SAVE_ID \
        "${EXTRA_ARGS[@]}" \
        -save "$SAVE" \
        --data "$DATASET" \
        --tau "$TAU" \

elif [ "$MODE" = "valid" ]; then
    echo "Start Evaluation on Valid Data Set......"

    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" python -u "${CODE_PATH}/run.py" \
        --do_valid \
        --cuda \
        --data "$DATASET" \
        -init "$SAVE"

elif [ "$MODE" = "test" ]; then
    echo "Start Evaluation on Test Data Set......"

    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" python -u "${CODE_PATH}/run.py" \
        --do_test \
        --cuda \
        --data "$DATASET" \
        -init "$SAVE"

else
    echo "Unknown MODE: $MODE"
    exit 1
fi