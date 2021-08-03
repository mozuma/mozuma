BASE_IMAGE ?= lsirepfl/pytorch:v1.7.1-py3.7.10-cu110
IMAGE_NAME ?= lsirepfl/mlmodule
MLMODULE_BUILD_VERSION ?= 0.0.dev0
MLMODULE_WHEEL_NAME := mlmodule-$(subst -,_,${MLMODULE_BUILD_VERSION})-py3-none-any.whl
IMAGE_TAG_PREFIX ?= v
IMAGE_TAG ?= ${IMAGE_TAG_PREFIX}${MLMODULE_BUILD_VERSION}
CPU_ONLY_TESTS ?= n

.PHONY: test-docker-image dist help

test-docker-image: dist	##@Release Test MLModule in the PyTorch base image
	@docker build \
		--build-arg MLMODULE_BUILD_VERSION=${MLMODULE_BUILD_VERSION} \
		--build-arg MLMODULE_WHEEL_NAME=${MLMODULE_WHEEL_NAME} \
		--build-arg BASE_IMAGE=${BASE_IMAGE} \
		-f tests/Dockerfile.test \
		-t ${IMAGE_NAME}:test-${IMAGE_TAG} .
	@docker run --rm \
		-e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
		-e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
		-e CPU_ONLY_TESTS=${CPU_ONLY_TESTS} \
		${IMAGE_NAME}:test-${IMAGE_TAG} \
		conda run -n app --no-capture-output pytest /app/tests || docker rm ${IMAGE_NAME}:test-${IMAGE_TAG}
	@docker rm ${IMAGE_NAME}:test-${IMAGE_TAG}


dist: dist/$(MLMODULE_WHEEL_NAME)	##@Release Builds MLModule wheel in dist/ folder

dist/$(MLMODULE_WHEEL_NAME): $(shell find src/mlmodule/ -name "*.py" -print)
	@python -m pip install build
	@MLMODULE_BUILD_VERSION=$(MLMODULE_BUILD_VERSION) python -m build --wheel

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
