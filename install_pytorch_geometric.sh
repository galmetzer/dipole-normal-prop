CUDA=cu101
TORCH=1.6.0

echo "*******************"
echo "Installing for pytorch ${TORCH} and CUDA ${CUDA}"
echo "Edit install_pytorch_geometric.sh for different versions"
echo "*******************"


pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html

