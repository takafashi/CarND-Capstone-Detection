#!/bin/bash -
set -u
set -e

docker run -it --rm -v $PWD:/detection --name=detection tensorflow/tensorflow:1.15.2 bash

