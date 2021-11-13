FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y curl python3-pip sudo
RUN apt-get install unzip

WORKDIR /usr/scr/app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip3 install --upgrade pip
COPY ./requrements.txt .
RUN pip3 install -r requirements.txt

COPY . .