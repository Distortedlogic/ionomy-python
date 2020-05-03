FROM jupyter/minimal-notebook
USER root
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
WORKDIR /notebooks
RUN chown 1000:100 .
USER 1000