#!/bin/bash -
set -u
set -e

MODEL_NAME=ssd_mobilenet_v1_coco_2018_01_28
TEST_DATA_DIR=test0001

export PYTHONPATH=`pwd`:`pwd`/slim

### setup
apt install protobuf-compiler -y
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf1/setup.py .
python -m pip install .

### train
python object_detection/model_main.py \
  --pipeline_config_path="object_detection/${MODEL_NAME}/pipeline.config" \
  --model_dir="./object_detection/${TEST_DATA_DIR}/save" \
  --alsologtostderr

