IMAGE_NAME ?= lsirepfl/mlmodule
MLMODULE_BUILD_VERSION ?= 0.0.dev0
IMAGE_TAG ?= ${MLMODULE_BUILD_VERSION}

.PHONY: build

docker-image: build
	@docker build --build-arg MLMODULE_BUILD_VERSION=${MLMODULE_BUILD_VERSION} -t ${IMAGE_NAME}:${IMAGE_TAG} .

test-docker-image: docker-image
	@docker build --build-arg MLMODULE_BUILD_VERSION=${MLMODULE_BUILD_VERSION} \
		--build-arg IMAGE_NAME=${IMAGE_NAME} \
		--build-arg IMAGE_TAG=${IMAGE_TAG} \
		-f Dockerfile.test \
		-t ${IMAGE_NAME}:test-${IMAGE_TAG} .
	@docker run --rm \
		-e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
		-e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
		${IMAGE_NAME}:test-${IMAGE_TAG} \
		conda run -n app --no-capture-output pytest /app/tests

release-docker-image: test-docker-image
ifeq (${IMAGE_TAG},0.0.dev0)
# Cannot push a default dev version
	@echo "Set the MLMODULE_BUILD_VERSION to a valid PEP 440 version before pushing an image (see https://www.python.org/dev/peps/pep-0440/)"
else
	@docker push ${IMAGE_NAME}:${IMAGE_TAG}
endif

build:
	@rm dist/*
	@python -m build --wheel

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
