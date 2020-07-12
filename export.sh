#!/bin/bash -
set -u
set -e

MODEL_NAME=${1:-ssd_mobilenet_v1_coco_2018_01_28}
MODEL_FILE_NAME=${2:-model.ckpt}

PIPELINE_PATH=/detection/${MODEL_NAME}/pipeline.config
TARGET_MODEL_PATH=/detection/${MODEL_NAME}/${MODEL_FILE_NAME}
EXPORT_MODEL_PATH=/detection/frozen_model

## setup
apt-get update
apt-get install -y protobuf-compiler python-pil python-lxml python-tk
pip install jupyter
pip install matplotlib
export PYTHONPATH=`pwd`:`pwd`/slim

# setup project
protoc object_detection/protos/*.proto --python_out=.

# patch project
cd ..
patch --forward -p1 < object_detection_exporter_py.patch
cd research

### clear output path
rm -rf ${EXPORT_MODEL_PATH}/*

### train
python object_detection/export_inference_graph.py \
  --input_type=image_tensor \
  --pipeline_config_path="${PIPELINE_PATH}" \
  --trained_checkpoint_prefix="${TARGET_MODEL_PATH}" \
  --output_directory="${EXPORT_MODEL_PATH}"

