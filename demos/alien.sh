#!/bin/bash

export BASE_PATH=$(cd ../; pwd)
echo $BASE_PATH
export PYTHONPATH=$BASE_PATH

python -u $BASE_PATH/orient_large.py \
--pc $BASE_PATH/data/interpolate/consolidated.xyz \
--export_dir $BASE_PATH/demos/alien \
--models $BASE_PATH/pre_trained/hands2.pt \
$BASE_PATH/pre_trained/hands.pt \
$BASE_PATH/pre_trained/manmade.pt \
--iters 5 \
--propagation_iters 4 \
--number_parts 41 \
--minimum_points_per_patch 100 \
--diffuse \
--weighted_prop \
--estimate_normals \
--curvature_threshold 0.01 \
--n 50