FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0 \
        wget

COPY requirements.txt requirements.txt

RUN rm /usr/lib/python*/EXTERNALLY-MANAGED
RUN python3 -m pip install -r requirements.txt

WORKDIR /usr/src/app
