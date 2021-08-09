MLMODULEKIT_VERSION := 1

.PHONY: build release

build: 
	@docker build . -t lsirepfl/mlmodulekit:${MLMODULEKIT_VERSION} -f Dockerfile.binaries
	@docker build . -t lsirepfl/mlmodulekit:${MLMODULEKIT_VERSION}-ca35 -f Dockerfile.sources

release: build login
	@docker push lsirepfl/mlmodulekit:${MLMODULEKIT_VERSION}
	@docker push lsirepfl/mlmodulekit:${MLMODULEKIT_VERSION}-ca35

login:
	@docker login
