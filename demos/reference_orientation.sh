#!/bin/bash

export BASE_PATH=$(cd ../; pwd)
echo $BASE_PATH
export PYTHONPATH=$BASE_PATH

python -u $BASE_PATH/reference_orientation.py \
--input $BASE_PATH/data/interpolate/consolidated.xyz \
--output $BASE_PATH/data/interpolate/result.xyz \
--reference $BASE_PATH/data/interpolate/reference.xyz