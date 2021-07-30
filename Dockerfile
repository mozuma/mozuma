ARG BASE_IMAGE

FROM ${BASE_IMAGE}

# Install requirements
ADD requirements.txt /app/build/requirements.txt
RUN pip install -r /app/build/requirements.txt

# Install app
ARG MLMODULE_BUILD_VERSION
ADD dist /app/build/dist
RUN pip install -f /app/build/dist/ mlmodule==${MLMODULE_BUILD_VERSION}
