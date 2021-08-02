IMAGE_NAME ?= lsirepfl/pytorch

PYTHON_VERSION ?= 3.7.10
CUDA_VERSION ?= 11.0
TORCH_VERSION ?= 1.7.1
TORCHVISION_VERSION ?= 0.8.2

IMAGE_TAG_PREFIX ?=
CUDA_MAGMA_VERSION := $(subst .,,${CUDA_VERSION})
IMAGE_TAG := ${IMAGE_TAG_PREFIX}v1.7.1-py${PYTHON_VERSION}-cu${CUDA_MAGMA_VERSION}

# Access to LSIR public assets bucket. See https://github.com/LSIR/dataplatform-infra/tree/main/lsir-public-assets#read-bucket-content
AWS_ACCESS_KEY_ID ?=
AWS_SECRET_ACCESS_KEY ?=

build:
	@docker build . -f Dockerfile.from-binary \
		--build-arg PYTHON_VERSION=${PYTHON_VERSION} \
		--build-arg CUDA_VERSION=${CUDA_VERSION} \
		--build-arg TORCH_VERSION=${TORCH_VERSION} \
		--build-arg TORCHVISION_VERSION=${TORCHVISION_VERSION} \
		-t ${IMAGE_NAME}:${IMAGE_TAG}

wheel-build:
ifeq ($(IMAGE_TAG_PREFIX),)
	@echo "The variable IMAGE_TAG_PREFIX must be set to install PyTorch from wheels"
else
	@docker build . -f Dockerfile.from-wheel \
		--build-arg PYTHON_VERSION=${PYTHON_VERSION} \
		--build-arg CUDA_VERSION=${CUDA_VERSION} \
		--build-arg TORCH_VERSION=${TORCH_VERSION} \
		--build-arg TORCHVISION_VERSION=${TORCHVISION_VERSION} \
		-t ${IMAGE_NAME}:${IMAGE_TAG}
endif

download-wheels:
	@pip install wheel awscli
	@AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
		AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
		aws --endpoint-url https://sos-ch-gva-2.exo.io s3 sync s3://lsir-public-assets/wheels/ ./deps/
