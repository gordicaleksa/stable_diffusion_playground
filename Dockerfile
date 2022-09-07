FROM ubuntu:latest
COPY . /
RUN apt update && apt install wget python3 git python3-pip ffmpeg libsm6 libxext6 -y 
RUN wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh && bash Anaconda3-2022.05-Linux-x86_64.sh -b
RUN /root/anaconda3/condabin/conda config --set remote_read_timeout_secs 1000.0
RUN ./root/anaconda3/condabin/conda env create
RUN pip3 install huggingface_hub
RUN git config --global credential.helper store
ENTRYPOINT /bin/bash entrypoint.sh


