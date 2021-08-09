set -e

MMCV_VERSION=$1

git clone --depth 1 --recursive --branch v${MMCV_VERSION} https://github.com/open-mmlab/mmcv.git
cd mmcv

MMCV_WITH_OPS=1 python setup.py install

cd ..
rm -rf mmcv
