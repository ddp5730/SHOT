#!/bin/bash
# This will run the SHOT for training on the synth dataset as a source

if (($# < 1))
then
  tag='default'
else
  tag=$1
fi

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
--center-loss True \
--cent-lr=0.01 \
--cent-alpha=2 \
--name="$tag"
