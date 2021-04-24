#!/bin/bash
set -x

module load python3
module load cuda-toolkit-10.0
module load cudnn-7.0.5

source .venv/bin/activate
cd "$(dirname "$0")"

cd examples
python inception_cifar.py --train \
  --lr "$1" \
  --bsize "$2" \
  --keep_prob "$3" \
  --maxepoch "$4" \
  "$5" "$6" "$7"