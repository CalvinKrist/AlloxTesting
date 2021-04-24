#!/bin/bash
set -x

module load python3
module load cuda-toolkit-10.0
module load cudnn-7.0.5

source .venv/bin/activate
cd "$(dirname "$0")"

# $1: epochs
# $2: hardware mode, either --cpu or --gpu
# $3: the hardware config type, ie numThreads or numGPUs
# $4: the value for the hardware config type
python3.7 train.py $1 $2 $3 $4