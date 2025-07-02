#!/bin/bash

python analysis/fourier.py \
    --max-position-embeddings 8192\
    --original-max-position-embeddings 4096 \
    --min-tokens 4096 --max-tokens 16384 \
    --yarn 4 \