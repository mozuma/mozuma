# Run with Docker

[MoZuMaKit](https://github.com/mozuma/mozumakit/pkgs/container/mozumakit) is a collection of docker images with a pre-configured environment to use MoZuMa.

You can pull the image for a specific [`<version>`](#available-versions) with:

```
docker pull ghcr.io/mozuma/mozumakit:<version>
```

## Usage

This example will go through the process to run a python script called `main.py`
in the MoZuMaKit docker image.

As a first step, we need to create a `Dockerfile` that:

1. Imports `mozumakit` image
1. Installs the latest version of MoZuMa
1. Copies the script we want to run.

```Dockerfile
FROM ghcr.io/mozuma/mozumakit:3

WORKDIR /app
RUN pip install git+https://github.com/mozuma/mozuma

ADD main.py .

ENTRYPOINT ["conda", "run", "-n", "app", "--no-capture-output"]
```

Then, we need to build a docker image from the `Dockerfile`:

```shell
docker build . -t my-job
```

The previous command has created a docker container
that we can run with the following command:

```shell
docker run my-job python main.py
```

That's it ! You should see you script output in the terminal.

## Available versions

See [MoZuMaKit releases](https://github.com/mozuma/mozumakit/releases).
