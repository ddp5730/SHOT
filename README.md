# Research Code for Swapping **SHOT** Backbones

**For the original paper and source code please see:** [SHOT](https://github.com/tim-learn/SHOT)

This repository contains the research code used to test the SHOT domain
adaptation technique with using different backbone networks.
Specifically the backbone was swapped from a ResNet backbone to a 
[SWIN](https://github.com/microsoft/Swin-Transformer) and an
[HRNet-V2](https://github.com/HRNet/HRNet-Image-Classification) model.

## Setup

### Clone Repository

1. Clone repository
   ```
     git clone git@github.com:ddp5730/SHOT.git
   ```
2. Install submodules
   ```
   git submodule update --init --recursive
   ```

### Requirements
See Dockerfile for full list of install dependencies and packages

- CUDA == 11.3.1
- torch == 1.10.2+cu113
- torchvision=0.11.3+cu113
- [apex](https://github.com/NVIDIA/apex)

#### Build Docker Container (Optional)

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

### Download Datasets
Datasets are loaded using the [DatasetFolder](https://pytorch.org/vision/stable/generated/torchvision.datasets.DatasetFolder.html#torchvision.datasets.DatasetFolder)
class.  Therefore all datasets should be downloaded in directories
as such:
```
directory/
├── train
    ├── class_x
    │   ├── xxx.ext
    │   ├── xxy.ext
    │   └── ...
    │       └── xxz.ext
    └── class_y
        ├── 123.ext
        ├── nsdf3.ext
        └── ...
        └── asd932_.ext
├── test
    ├── class_x
    │   ├── xxx.ext
    │   ├── xxy.ext
    │   └── ...
    │       └── xxz.ext
    └── class_y
        ├── 123.ext
        ├── nsdf3.ext
        └── ...
        └── asd932_.ext
```

The file `utils/partition_dota_xview.py` provides a helpful script
for partitioning a dataset into training/validation splits.


### Download Pretrained models
To recreate the results download the following pretrained Swin and
HRNet models.

[HRNet-W48-C](https://github.com/HRNet/HRNet-Image-Classification)

[swin_base_patch4_window12_384_22k](https://github.com/microsoft/Swin-Transformer)

## Run the Code
Sample config files and scripts are contained in 
[sample_configs](sample_configs) and [sample_scripts](sample_scripts)
respectively.

### Fine Tune Model on Target Dataset
The file [image_source.py](object/image_source.py) is used
to fine tune a model onto a source dataset.  To fine tune a 
Swin-B model pretrained on ImageNet-22k to DOTA you could run
the following command:

```
PYTHONPATH=.:./swin:$PYTHONPATH python3 -m torch.distributed.launch \
--nproc_per_node 1 \
--master_port 12345 \
object/image_source.py \
--trte val \
--da uda \
--output output \
--gpu_id 0 \
--cfg sample_configs/swin_base_patch4_window12_384_22ktodota_transfer.yaml \
--pretrained data/swin_base_patch4_window12_384_22k.pth \
--dset dota \
--data-path /home/poppfd/data/dota-xview/DOTA_ImageFolder \
--t-dset xview \
--t-data-path /home/poppfd/data/dota-xview/XVIEW_ImageFolder \
--evals-per-epoch 1 \
--batch_size=20 \
--net=swin-b \
--transfer-dataset \
--source 1 \
--target 0 \
--name=swin-dota-source-1
```

For this code, the `TOP_N` performing models on the target
domain will be saved in `output/<name>/T/` for further analysis.
The `TOP_N` value is a global variable in the `image_source.py`
script.

### Evaluate Model Generalization
The file [image_eval.py](object/image_eval.py) is used to 
evaluate the performance of a model on both a source and target
dataset.  This script does not perform any training and is for
evaluation only.

To evaluate the performance of the Swin-B model fine-tuned on
the DOTA dataset you could run the following command:

```
PYTHONPATH=.:./swin:$PYTHONPATH python3 -m torch.distributed.launch \
--nproc_per_node 1 \
--master_port 12345 \
object/image_eval.py \
--output output \
--gpu_id 0 \
--cfg /home/poppfd/College/Research/SHOT/configs/swin_base_patch4_window12_384_22ktodota_transfer.yaml \
--pretrained /home/poppfd/College/Research/SHOT/output/swin-dota-to-xview-target-3/X/ckpt_epoch_6_eval_10.pth \
--dset dota \
--data-path /home/poppfd/data/dota-xview/DOTA_ImageFolder \
--t-dset xview \
--t-data-path /home/poppfd/data/dota-xview/XVIEW_ImageFolder \
--batch_size=128 \
--net=swin-b \
--transfer-dataset \
--source 1 \
--target 0 \
--name=swin-dota-source-1
```

*Any script that loads netF, netB, and netC only needs to be
pointed to the saved netF path using the `--pretrained` argument*

Note that the saved output of this evaluation is placed in 
`output/eval/<name>`.

Also if the `--t-dset` and `--t-data-path` arguments are omitted
this script can simply evaluate the model on a given dataset.

### Adapt model using SHOT to Target Domain
The script [image_target.py](object/image_target.py) is used to 
adapt a given model onto a target domain using the unsupervised 
domain adaptation technique SHOT.

To perform this adaptation on a Swin-B model fine-tuned on DOTA
and adapt it to the XVIEW dataset, the following command
could be used:

```
PYTHONPATH=.:./swin:$PYTHONPATH python3 -m torch.distributed.launch \
--nproc_per_node 1 \
--master_port 12345 \
object/image_target.py \
--cls_par 0.3 \
--da uda \
--output output \
--gpu_id 0 \
--cfg sample_configs/swin_base_patch4_window12_384_dota_to_xview_adaptatiopn.yaml \
--pretrained output/swin-dota-source-1/V/ckpt_epoch_9_eval_8.pth \
--dset xview \
--data-path /home/poppfd/data/dota-xview/ \
--batch_size=20 \
--evals-per-epoch=2 \
--net=swin-b \
--transfer-dataset \
--source -1 \
--target 0 \
--name=swin-dota-to-xview-1
```

**VERY IMPORTANT: Due to a bug when generating pseudo-labels,
the `--batch-size`
argument must perfectly divide the target dataset**

i.e. dataset_size % batch_size == 0

### Tensorboard
Important training metrics for this project are logged using
[Tensorboard](https://www.tensorflow.org/tensorboard/get_started).
When training these metrics can be seen by:

1. `$ tensorboard --logdir='logs/<name>`
2. Open a web-browser and navigate to `localhost:6006`

### Contact

- [ddp5730@rit.edu](mailto:ddp5730@rit.edu)
