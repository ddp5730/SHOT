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
--name="$tag"
