#!/bin/bash -
set -u
set -e

MODEL_NAME=ssd_mobilenet_v1_coco_2018_01_28
TEST_DATA_DIR=test0001

### train
python object_detection/model_main.py \
  --pipeline_config_path="object_detection/${MODEL_NAME}/pipeline.config" \
  --model_dir="/output_model/" \
  --alsologtostderr

