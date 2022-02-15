# if any of the commands in your code fails for any reason, the entire script fails
set -o errexit
# fail exit if one of your pipe command fails
set -o pipefail
# traces commands before executing them
set -o xtrace

# Install Pillow-SIMD w/ libjpeg-turbo
# See: https://fastai1.fast.ai/performance.html
if [ $1 = "rebuild-pillow-simd" ]; then
    # this means we already installed libjpeg-turbo
    # and just need to rebuild Pillow-SIMD
    CFLAGS="${CFLAGS} -mavx2" pip install --upgrade --no-cache-dir --force-reinstall --no-binary :all: --compile pillow-simd
else
    conda uninstall -y --force pillow pil jpeg libtiff libjpeg-turbo
    pip uninstall -y pillow pil jpeg libtiff libjpeg-turbo
    conda install -yc conda-forge libjpeg-turbo
    CFLAGS="${CFLAGS} -mavx2" pip install --upgrade --no-cache-dir --force-reinstall --no-binary :all: --compile pillow-simd
    conda install -y jpeg libtiff
fi