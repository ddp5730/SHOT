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
--name="$tag"
