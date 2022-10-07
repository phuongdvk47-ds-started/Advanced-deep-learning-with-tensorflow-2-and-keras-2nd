FROM nvcr.io/nvidia/tensorflow:22.01-tf2-py3

RUN apt-get update && apt-get install -y git
RUN mkdir /init
COPY ./requirements-docker.txt /init/requirements.txt
RUN pip3 -q install pip --upgrade
RUN pip install -r /init/requirements.txt