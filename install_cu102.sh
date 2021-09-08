#!/bin/bash

# create and activate env
conda create -n dipole python=3.6
source /path-to-anaconda3/etc/profile.d/conda.sh
conda activate dipole

#pytorch1.8:
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

#pytorch-geometric
conda install pytorch-geometric -c rusty1s -c conda-forge

pip install argparse
pip install open3d==0.9.0