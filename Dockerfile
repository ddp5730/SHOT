# Build Docker image specific to SHOT

FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

RUN apt update && apt install -y build-essential python3 python3-pip git
RUN apt clean

RUN pip3 install matplotlib
RUN pip3 install numpy
RUN pip3 install cupy
RUN pip3 install scipy
RUN pip3 install tqdm
RUN pip3 install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip3 install torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip3 install git+https://github.com/nottombrown/imagenet_stubs

RUN useradd --create-home --shell /bin/bash developer

WORKDIR /opt
RUN git clone https://github.com/NVIDIA/apex.git
RUN cd apex && pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && cd ..

RUN pip3 install timm
RUN pip3 install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8
RUN pip3 install google protobuf
RUN pip3 install tensorboard
RUN pip3 install sklearn