#!/bin/bash

export BASE_PATH=$(cd ../; pwd)
echo $BASE_PATH
export PYTHONPATH=$BASE_PATH

python -u $BASE_PATH/orient_simple.py \
--pc $BASE_PATH/data/ok.xyz \
--export_dir $BASE_PATH/demos/ok_simple \
--estimate_normals