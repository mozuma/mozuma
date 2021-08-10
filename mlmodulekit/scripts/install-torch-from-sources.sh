set -e

TORCH_VERSION=$1

conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
conda install -c pytorch magma-cuda110

git clone --depth 1 --recursive --branch v${TORCH_VERSION} https://github.com/pytorch/pytorch
cd pytorch
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export PYTORCH_BUILD_VERSION=${TORCH_VERSION}
export PYTORCH_BUILD_NUMBER=0
python setup.py install

cd ..
rm -rf pytorch
