#!/bin/bash

export BASE_PATH=$(cd ../; pwd)
echo $BASE_PATH
export PYTHONPATH=$BASE_PATH

python -u $BASE_PATH/orient_pointcloud.py \
--pc $BASE_PATH/data/vase.xyz \
--export_dir $BASE_PATH/demos/vase \
--models $BASE_PATH/pre_trained/hands2.pt \
$BASE_PATH/pre_trained/hands.pt \
$BASE_PATH/pre_trained/manmade.pt \
--iters 5 \
--propagation_iters 4 \
--number_parts 25 \
--minimum_points_per_patch 100 \
--weighted_prop \
--estimate_normals \
--diffuse
