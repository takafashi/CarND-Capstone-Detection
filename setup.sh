#!/bin/bash -

MODEL_NAME=ssd_mobilenet_v1_coco_2018_01_28
TEST_DATA_DIR=test0001
CUR_DIR=`pwd`

### download ssd mobilenet v1 coco
curl -O http://download.tensorflow.org/models/object_detection/${MODEL_NAME}.tar.gz
tar -zxvf ${MODEL_NAME}.tar.gz
cp pipeline.config ${MODEL_NAME}/

### download tensorflow models
git clone --depth 1 https://github.com/tensorflow/models.git

### set model data
mv ${MODEL_NAME} models/research/object_detection/
cp -r ${TEST_DATA_DIR} models/research/object_detection/

### set scripts
cp train.sh models/research/
cp export.sh models/research/

### build docker
cd models
docker build -f research/object_detection/dockerfiles/tf1/Dockerfile -t od .
cd ${CUR_DIR}

