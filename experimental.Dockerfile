# syntax = docker/dockerfile:experimental
FROM jupyter/minimal-notebook
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends graphviz-dev graphviz \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir pyparsing pydot
RUN pip install --install-option="--include-path=/usr/local/include/graphviz/" --install-option="--library-path=/usr/local/lib/graphviz" pygraphviz
COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt --cache-dir /root/.cache/pip
WORKDIR /notebooks
COPY . .
RUN pip install -e .
RUN chown 1000:100 .
USER 1000