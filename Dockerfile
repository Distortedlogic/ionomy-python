FROM jupyter/minimal-notebook
COPY requirements.txt requirements.txt
USER root
RUN pip install -r requirements.txt
COPY . .
RUN pip install -e .
WORKDIR /notebooks
RUN chown 1000:100 .
USER 1000