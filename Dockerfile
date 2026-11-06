# usage: docker build -t yourusername/paynt .
# command-line arguments can be specified e.g. as follows: --build-arg no_threads=8

# base image
ARG BASE_IMAGE=movesrwth/stormpy:ci
FROM $BASE_IMAGE

# additional arguments for compiling payntbind
ARG setup_args=""
# number of threads to use for parallel compilation
ARG no_threads=2
# optional dependencies of paynt, e.g. [test]
ARG options=""

WORKDIR /opt/

# install dependencies
RUN apt-get update -qq
RUN apt-get install -y graphviz

# install paynt
WORKDIR /opt/paynt
COPY . .
RUN pip install -r build-requirements.txt
RUN pip install ".${options}" --no-build-isolation
