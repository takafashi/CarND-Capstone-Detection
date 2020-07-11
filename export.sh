#!/bin/bash -
set -u
set -e

MODEL_NAME=ssd_mobilenet_v1_coco_2018_01_28
EXPORT_MODEL=${1}
OUTPUT_DIR=/output_model/frozen

rm -rf ${OUTPUT_DIR}/*

### train
python object_detection/export_inference_graph.py \
  --input_type=image_tensor \
  --pipeline_config_path="object_detection/${MODEL_NAME}/pipeline.config" \
  --trained_checkpoint_prefix="${EXPORT_MODEL}" \
  --output_directory="${OUTPUT_DIR}"

