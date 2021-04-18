#!/bin/bash
module load python3
source .venv/bin/activate
python -c "from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())"
