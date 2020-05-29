FROM jupyter/minimal-notebook
RUN add-apt-repository ppa:graphics-drivers
RUN apt-get update
RUN apt install -y nvidia-440 mpi
USER root
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install ionomy-python==0.0.6
WORKDIR /notebooks
COPY docs/source/notebooks /notebooks
RUN chown 1000:100 .
USER 1000