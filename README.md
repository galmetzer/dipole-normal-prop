# Orienting Point Clouds with Dipole Propagation
<img src='docs/images/output.gif' align="right" width=325>
 Leveraging the power of neural networks and a robust and stable dipole propagation technique.

### SIGGRAPH 2021 [[Paper]](https://arxiv.org/abs/2105.01604) [[Project Page]](https://galmetzer.github.io/dipole-normal-prop/)<br>
by [Gal Metzer](https://galmetzer.github.io/),
[Rana Hanocka](https://www.cs.tau.ac.il/~hanocka/),
[Denis Zorin](https://cims.nyu.edu/gcl/denis.html),
[Raja Giryes](http://web.eng.tau.ac.il/~raja),
[Daniele Panozzo](https://cims.nyu.edu/gcl/daniele.html),
and [Daniel Cohen-Or](https://danielcohenor.com/)

# Getting Started

### Installation
- Clone this repo

#### Setup Conda Environment
To run the code, it is advised to create a new conda environment by running `conda create -n dipole python=3.6`

and then install the required libraries below:

- [PyTorch](https://pytorch.org/) version 1.6.0 (might work with later versions as well) <br>
  install from [PyTorch getting started](https://pytorch.org/get-started/locally/) with your appropriate CUDATOOLKIT
- [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric) <br>
may use the helper installation script `./install_pytorch_geometric.sh` or from [here](https://github.com/rusty1s/pytorch_geometric). <br>
  ** if using the helper script make sure to change the declared variables inside the script to match your installed versions **
- [argparse](https://docs.python.org/3/library/argparse.html) `pip install argparse`
- [Open3D](http://www.open3d.org/) `pip install open3d` (optional for faster normal estimation)

`./install.sh` is a full installation script for pytorch 1.6.0 and cudatoolkit 10.1.

# Running Examples
The code contains three main orientation scripts
- `orient_pointcloud.py` used to orient relatively small point clouds, taking into account all input points.
- `orient_large.py` used to orient relatively large point clouds, using a representative set of points for each patch.
- `reference_orientation.py` used to transfer the orientation from an oriented source point cloud to an un-oriented target point cloud.

The `/demos` folder contains demo scripts for running the code on point clouds from the `/data` folder.

Simple `cd demos` and run any of the scripts:

#### Example Point Clouds
- `fandisk.sh`
- `hand.sh` 
- `ok.sh`
- `vase.sh`
- `galera.sh`
- `boxunion.sh`
- `flower.sh`

#### Large Example Point Clouds
Example on a large point clouds
- `lion.sh`
- `alien.sh`

#### Reference orientation
Example of using the dipole field to transfer the orientation from an oriented source point cloud
to an un-oriented target point cloud.
- `reference_orientation.sh` calculated on an output of [Self-Sampling](https://galmetzer.github.io/self-sample/) <br> 
which does not have normal information for the produced consolidated point cloud.

#### Simple Point Propagation
Example of using dipole propagation individually per point i.e. without patching and network steps.
- `ok_simple.sh`

#

In this implantation we added an extra step to fix the global orientation such that the normals are not only ***consistent***, <br>
but also points ***outside*** the surface, which was not included in the original paper. <br>

This technique relies on the fact that the dipole potential for a closed and correctly oriented shape is ***zero*** outside, 
and strictly positive or negative inside. Depending on whether the normals point inside or outside. <br>
The dipole potential is measured on a square lattice around the oriented point cloud,
and the sign of the average potential is used to determine whether all normals point outside. 


# Data
Data thanks to [Self-Sampling](https://galmetzer.github.io/self-sample/) [NestiNet](https://github.com/sitzikbs/Nesti-Net), [Mano](https://mano.is.tue.mpg.de/), 
[threedscans](https://threedscans.com/), [COSEG](http://irc.cs.sdu.edu.cn/~yunhai/public_html/ssl/ssd.htm).

# Citation
If you find this code useful, please consider citing our paper
```
@article{metzer2021orienting,
  title={Orienting Point Clouds with Dipole Propagation},
  author={Metzer, Gal and Hanocka, Rana and Zorin, Denis and Giryes, Raja and Panozzo, Daniele and Cohen-Or, Daniel},
  journal={arXiv preprint arXiv:2105.01604},
  year={2021}
}
```

# Questions / Issues
If you have questions or issues running this code, please open an issue.
