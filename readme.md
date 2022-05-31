# Research Code for Swapping **SHOT** Backbones

**For the original paper and source code please see:** [SHOT](https://github.com/tim-learn/SHOT)

This repository contains the research code used to test the SHOT domain
adaptation technique with using different backbone networks.
Specifically the backbone was swapped from a ResNet backbone to a 
[SWIN](https://github.com/microsoft/Swin-Transformer) and an
[HRNet-V2](https://github.com/HRNet/HRNet-Image-Classification) model.

## Clone Repository

1. Clone repository
	```
 	git clone git@github.com:ddp5730/SHOT.git
 	```

## Requirements
See Dockerfile for full list of install dependencies and packages

- CUDA == 11.3.1
- torch == 1.10.2+cu113
- torchvision=0.11.3+cu113
- [apex](https://github.com/NVIDIA/apex)

### Build Docker Container (Optional)

You may use the provided Dockerfile to build a container with all
of the necessary requirements required to run the provided code.
However, you must have some version of CUDA, Docker and the
NVIDIA container toolkit installed (see [link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)).

1. Update .dockerignore with any added directories as necessary
2. Build the Docker container
    ```
   $ docker build -t <desired-tag> -f Dockerfile .
    ```
3. Run Docker Container.  `<tag>` must be the same tag used in step 4.
   ```
   docker run -it --gpus all --shm-size=25G -e HOME=$HOME -e USER=$USER -v $HOME:$HOME -w $HOME --user developer <tag>
   ```
4. Navigate to code (Home directories will be linked) and run

## Download Datasets

## Download Pretrained models
- HRNET_W whatever...

### Contact

- [ddp5730@rit.edu](mailto:ddp5730@rit.edu)
