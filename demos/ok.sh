#!/bin/bash

export BASE_PATH=$(cd ../; pwd)
echo $BASE_PATH
export PYTHONPATH=$BASE_PATH

python -u $BASE_PATH/orient_pointcloud.py \
--pc $BASE_PATH/data/ok.xyz \
--export_dir $BASE_PATH/demos/ok \
--models $BASE_PATH/pre_trained/hands2.pt \
$BASE_PATH/pre_trained/hands.pt \
$BASE_PATH/pre_trained/manmade.pt \
--iters 10 \
--propagation_iters 5 \
--number_parts 30 \
--minimum_points_per_patch 100 \
--curvature_threshold 0.01 \
--diffuse \
--weighted_prop \
--estimate_normals
