# MLModule Kit

Docker images with a pre-configured environment to use MLModule.

```
docker pull lsirepfl/mlmodulekit:<version>
```

## Usage

In order to use the Docker Image and run a script called `main.py`,
you should create a new `Dockerfile` with the following content.

```Dockerfile
FROM lsirepfl/mlmodulekit:3

WORKDIR /app
RUN pip install git+https://github.com/LSIR/mlmodule

ADD main.py .

ENTRYPOINT ["conda", "run", "-n", "app", "--no-capture-output"]
```

Then build the new docker image:

```shell
docker build . -t my-mlmodule-job
```

And run the resulting container with:

```shell
docker run my-mlmodule-job python main.py
```

## Available versions

Description of available image tags.

### `3`

* Python 3.7
* CUDA 11.1
* PyTorch 1.9.1
* TorchVision 0.10.1
