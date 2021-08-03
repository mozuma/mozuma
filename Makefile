BASE_IMAGE ?= lsirepfl/pytorch:v1.7.1-py3.7.10-cu110
IMAGE_NAME ?= lsirepfl/mlmodule
MLMODULE_BUILD_VERSION ?= 0.0.dev0
MLMODULE_WHEEL_NAME := mlmodule-$(subst -,_,${MLMODULE_BUILD_VERSION})-py3-none-any.whl
IMAGE_TAG_PREFIX ?= v
IMAGE_TAG ?= ${IMAGE_TAG_PREFIX}${MLMODULE_BUILD_VERSION}
CPU_ONLY_TESTS ?= n

.PHONY: docker-image test-docker-image release-docker-image dist install install-minimal develop env-install \
	env-sync requirements.txt test help

docker-image: dist
	@docker build \
		--build-arg MLMODULE_BUILD_VERSION=${MLMODULE_BUILD_VERSION} \
		--build-arg MLMODULE_WHEEL_NAME=${MLMODULE_WHEEL_NAME} \
		--build-arg BASE_IMAGE=${BASE_IMAGE} \
		-t ${IMAGE_NAME}:${IMAGE_TAG} .

test-docker-image: docker-image	##@Release Test MLModule in the PyTorch base image
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

dist: dist/$(MLMODULE_WHEEL_NAME)	##@Release Builds MLModule wheel in dist/ folder

dist/$(MLMODULE_WHEEL_NAME): $(shell find src/mlmodule/ -name "*.py" -print)
	@python -m pip install build
	@MLMODULE_BUILD_VERSION=$(MLMODULE_BUILD_VERSION) python -m build --wheel

install:	##@Release full installation of MLModule with all dependencies to run any model in mlmodule.contrib
	@pip install .[full]

install-minimal:	##@Release Minimal installation of MLModule with no dependencies to run models in mlmodule.contrib
	@pip install .

develop: env-install	##@Development Install mlmodule as an editable module (pip install -e ...)
	@pip install -e .[test]

env-install:	##@Development Install requirements.txt into current environment
	@pip install -r requirements.txt

env-sync:	##@Development Synchronize current Python environment with requirements.txt (removes dependencies)
	@pip-sync

requirements.txt:	##@Development Update requirements.txt
	@pip-compile --extra full --upgrade

test:	##@Development Run tests
	@pytest tests


#COLORS
GREEN  := $(shell tput -Txterm setaf 2)
WHITE  := $(shell tput -Txterm setaf 7)
YELLOW := $(shell tput -Txterm setaf 3)
RESET  := $(shell tput -Txterm sgr0)

# Add the following 'help' target to your Makefile
# And add help text after each target name starting with '\#\#'
# A category can be added with @category
HELP_FUN = \
    %help; \
    while(<>) { push @{$$help{$$2 // 'options'}}, [$$1, $$3] if /^([a-zA-Z\-\.]+)\s*:.*\#\#(?:@([a-zA-Z\-]+))?\s(.*)$$/ }; \
    print "usage: make [target]\n\n"; \
    for (sort keys %help) { \
    print "$$_:\n"; \
    for (@{$$help{$$_}}) { \
    $$sep = " " x (32 - length $$_->[0]); \
    print "  $$_->[0]$$sep$$_->[1]\n"; \
    }; \
    print "\n"; }

help: ##@Help Show this help.
	@perl -e '$(HELP_FUN)' $(MAKEFILE_LIST)
