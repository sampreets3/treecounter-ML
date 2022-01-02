FROM nvidia/cuda:11.0-base

RUN echo "Hello, building this image."
RUN mkdir -p /home/app && cd /home/app
RUN ls 
RUN ls /home

RUN apt-get update -y && apt-get upgrade -y 
RUN apt-get install python3-pip -y
RUN pip3 install opencv-python

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y


RUN echo "All done"