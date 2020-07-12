#!/bin/bash -
set -u
set -e

docker run --gpus all -it --rm -p 6006:6006 \
  -v $PWD/output_model:/output_model \
  -v $PWD:/source \
  --name=detection od

