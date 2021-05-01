#!/bin/bash
set -ex;

cd alloxtesting/estimator
git pull
python3 main.py "$@"