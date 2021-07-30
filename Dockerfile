FROM continuumio/miniconda3:4.10.3

RUN apt-get update -y && apt-get install -y build-essential libgl1-mesa-dev && apt-get clean

# Creating and making the environment usable by the data platform group
ARG PYTHON_VERSION=3.7.10
ARG CUDA_TOOLKIT_VERSION=11.0
RUN conda create -n app python=$PYTHON_VERSION cudatoolkit=$CUDA_TOOLKIT_VERSION mkl mkl-include && \
    conda install -n app -c conda-forge uwsgi && \
    addgroup data-platform && \
    chown :data-platform /opt/conda/envs/app && \
    chmod g+w /opt/conda/envs/app

# Make sure the shell commands run inside the environment
SHELL ["conda", "run", "-n", "app", "/bin/bash", "-c"]

# Install requirements
ADD requirements.txt /app/build/requirements.txt
RUN pip install -r /app/build/requirements.txt

# Install app
ARG MLMODULE_BUILD_VERSION
ADD dist /app/build/dist
RUN pip install -f /app/build/dist/ mlmodule==${MLMODULE_BUILD_VERSION}
