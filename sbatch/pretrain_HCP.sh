#!/bin/bash

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="6,7"

uv run jupyter nbconvert scripts/main.ipynb --to python --output-dir scripts/nbconvert

CONFIG="HCP_large_pt1" model_name="HCP_large_pt1_2gpu" \
    uv run --env-file .env \
    torchrun --standalone --nproc_per_node=2 scripts/nbconvert/main.py
