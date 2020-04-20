FROM python:3.6-buster

### Dependencies ###
RUN apt-get update && apt-get install -y swig

ADD requirements.txt /root/requirements.txt
RUN cd /root && pip3 install -r requirements.txt

### Project files ###
ADD . /root/
WORKDIR /root
