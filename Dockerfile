# Build Docker image specific to SHOT

FROM ml-base

WORKDIR /opt
RUN git clone https://github.com/NVIDIA/apex.git
RUN cd apex && pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && cd ..

RUN pip3 install timm
RUN pip3 install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8
RUN pip3 install google protobuf
RUN pip3 install tensorboard
RUN pip3 install sklearn