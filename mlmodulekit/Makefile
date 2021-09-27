MLMODULEKIT_VERSION := 1

.PHONY: build release

build: 
	@docker build . -t lsirepfl/mlmodulekit:${MLMODULEKIT_VERSION} -f Dockerfile.binaries

release: build login
	@docker push lsirepfl/mlmodulekit:${MLMODULEKIT_VERSION}

login:
	@docker login
