# syntax = docker/dockerfile:experimental
FROM jupyter/minimal-notebook
COPY requirements.txt requirements.txt
USER root
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:graphics-drivers/ppa
RUN apt-get install -y nvidia-driver-440
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt --cache-dir /root/.cache/pip
WORKDIR /notebooks
COPY . .
RUN pip install -e .
RUN chown 1000:100 .
USER 1000