#!/bin/bash
set -x

source .venv/bin/activate
cd "$(dirname "$0")"

cd examples
python3 inception_cifar.py --train \
  --lr "$1" \
  --bsize "$2" \
  --keep_prob "$3" \
  --maxepoch "$4" \
  "$5" "$6" "$7"