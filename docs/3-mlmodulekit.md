# MLModule Kit

MLModuleKit is a collection of docker images with a pre-configured environment to use MLModule.

You can pull the image for a specific [`<version>`](#available-versions) with:

```
docker pull lsirepfl/mlmodulekit:<version>
```

## Usage

This example will go through the process to run a python script called `main.py`
in the MLModuleKit docker image.

As a first step, we need to create a `Dockerfile` that:

1. Imports `mlmodulekit` image
1. Installs the latest version of MLModule
1. Copies the script we want to run.

```Dockerfile
FROM lsirepfl/mlmodulekit:3

WORKDIR /app
RUN pip install git+https://github.com/LSIR/mlmodule

ADD main.py .

ENTRYPOINT ["conda", "run", "-n", "app", "--no-capture-output"]
```

Then, we need to build a docker image from the `Dockerfile`:

```shell
docker build . -t my-mlmodule-job
```

The previous command has created a docker container
that we can run with the following command:

```shell
docker run my-mlmodule-job python main.py
```

That's it ! You should see you script output in the terminal.

## Available versions

Description of available image tags.

### `3`

* Python 3.7
* CUDA 11.1
* PyTorch 1.9.1
* TorchVision 0.10.1
