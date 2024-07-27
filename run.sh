#!/bin/sh
DEVICE_ID=0  # which GPU is going to be used
export CUDA_VISIBLE_DEVICES=${DEVICE_ID}

BASEDIR=$(cd $(dirname $0) && pwd)

DS_ROOT="/sda2/datasets/stl_10"


TRAIN_LABELLED_FILE="/sda2/datasets/stl_10/train.json"
TRAIN_UNLABELLED_FILE="/sda2/datasets/stl_10/unlabelled_train.json"

TEST_FILE="/sda2/datasets/stl_10/test.json"

python3 -W ignore "main.py" \
    --experiment-name "MixMatch" \
    --ds-root $DS_ROOT \
    --dataset-name "STL-10" \
    --train-labelled-file $TRAIN_LABELLED_FILE \
    --train-unlabelled-file $TRAIN_UNLABELLED_FILE \
    --test-file $TEST_FILE \
    --num-classes "10" \
    --lr "0.01" \
    --batch-size "256" \
    --num-epochs "500" \
    --resized-shape "(96,96)" \
    --crop-size "(96,96)" \
    --prob-random-h-flip "0.5" \
    --jax-platform "cuda" \
    --mem-frac "0.975" \
    --progress-bar \
    --train \
    --tracking-uri "sqlite:///mixmatch.db" \
    --logdir "${BASEDIR}/logdir" \
    --num-threads 4 \
    --prefetch-size 8 \
    # --run-id "f21cf92e7fb84a8d9ee4faec1a215b0e"