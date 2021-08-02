BASE_IMAGE ?= lsirepfl/pytorch:v1.7.1-py3.7.10-cu110
IMAGE_NAME ?= lsirepfl/mlmodule
MLMODULE_BUILD_VERSION ?= 0.0.dev0
MLMODULE_WHEEL_NAME := mlmodule-$(MLMODULE_BUILD_VERSION)-py3-none-any.whl
IMAGE_TAG_PREFIX ?= v
IMAGE_TAG ?= ${IMAGE_TAG_PREFIX}${MLMODULE_BUILD_VERSION}
CPU_ONLY_TESTS ?= n

ifeq ($(MLMODULE_BUILD_VERSION), 0.0.dev0)
.PHONY: dist
endif

docker-image: dist
	@docker build \
		--build-arg MLMODULE_BUILD_VERSION=${MLMODULE_BUILD_VERSION} \
		--build-arg MLMODULE_WHEEL_NAME=${MLMODULE_WHEEL_NAME} \
		--build-arg BASE_IMAGE=${BASE_IMAGE} \
		-t ${IMAGE_NAME}:${IMAGE_TAG} .

test-docker-image: docker-image
	@docker build --build-arg MLMODULE_BUILD_VERSION=${MLMODULE_BUILD_VERSION} \
		--build-arg IMAGE_NAME=${IMAGE_NAME} \
		--build-arg IMAGE_TAG=${IMAGE_TAG} \
		-f Dockerfile.test \
		-t ${IMAGE_NAME}:test-${IMAGE_TAG} .
	@docker run --rm \
		-e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
		-e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
		-e CPU_ONLY_TESTS=${CPU_ONLY_TESTS} \
		${IMAGE_NAME}:test-${IMAGE_TAG} \
		conda run -n app --no-capture-output pytest /app/tests

release-docker-image: test-docker-image
ifeq (${IMAGE_TAG},0.0.dev0)
# Cannot push a default dev version
	@echo "Set the MLMODULE_BUILD_VERSION to a valid PEP 440 version before pushing an image (see https://www.python.org/dev/peps/pep-0440/)"
else
	@docker push ${IMAGE_NAME}:${IMAGE_TAG}
endif

dist: dist/$(MLMODULE_WHEEL_NAME)

dist/$(MLMODULE_WHEEL_NAME): $(shell find src/mlmodule/ -name "*.py" -print)
	@MLMODULE_BUILD_VERSION=$(MLMODULE_BUILD_VERSION) python -m build --wheel

install:
	@pip install .[full]

install-minimal:
	@pip install .

develop: env-install
	@pip install .[test]

env-install:
	@pip install -r requirements.txt

env-sync:
	@pip-sync

requirements:
	@pip-compile --extra full --upgrade

test:
	@pytest tests
