#!/bin/bash
set -x

# $1: epochs
# $2: hardware mode, either --cpu or --gpu
# $3: the hardware config type, ie numThreads or numGPUs
# $4: the value for the hardware config type

source .venv/bin/activate
cd "$(dirname "$0")"

echo "KEEP_PROB = 0.5
LEARNING_RATE = 1e-5
BATCH_SIZE =50
PARAMETER_FILE = \"checkpoint/variable.ckpt\"
MAX_ITER = $1" > config.py

python3 Train.py $2 $3 $4