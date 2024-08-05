#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

RESULT_DIR="./rome_results"
CHUNKS=${#GPULIST[@]}

for i in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[i]} python compute_neurons_multimodal_rome_edit.py \
    --device_id $i \
    --num_chunks $CHUNKS &
done
