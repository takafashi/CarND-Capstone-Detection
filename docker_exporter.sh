#!/bin/bash -
set -u
set -e

MODEL_NAME=${1:-ssd_mobilenet_v1_coco_2018_01_28}
MODEL_FILE_NAME=${2:-model.ckpt}

### if model isn't existed, download ssd mobilenet v1 coco
if [ ! -d ${MODEL_NAME} ]; then
  curl -O http://download.tensorflow.org/models/object_detection/${MODEL_NAME}.tar.gz
  tar -zxvf ${MODEL_NAME}.tar.gz
  rm ${MODEL_NAME}.tar.gz
fi

### if models-rep isn't existed, download tensorflow models
if [ ! -d models ]; then
  git clone https://github.com/tensorflow/models.git
  cd models
  git checkout 1f34fcafc1454e0d31ab4a6cc022102a54ac0f5b
  cd ..
fi

### set script and models
cp export.sh models/research/
cp object_detection_exporter_py.patch models/

docker run -it --rm \
  -v $PWD:/detection \
  --name detection \
  tensorflow/tensorflow:1.4.1 \
  bash
