set -e

TORCHVISION_VERSION=$1

git clone --depth 1 --recursive --branch v${TORCHVISION_VERSION} https://github.com/pytorch/vision
cd vision

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
BUILD_VERSION=${TORCHVISION_VERSION} python setup.py install

cd ..
rm -rf vision
