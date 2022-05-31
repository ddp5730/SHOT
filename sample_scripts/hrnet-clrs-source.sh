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
--cfg sample_configs/hrnet_384_22ktoclrs_transfer.yaml \
--cfg-hr HRNet/experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml \
--pretrained data/HRNet_W48_C_ssld_pretrained.pth \
--dset clrs \
--data-path /home/poppfd/data/clrs-nwpu/CLRS \
--t-dset nwpu \
--t-data-path /home/poppfd/data/clrs-nwpu/NWPU \
--evals-per-epoch 1 \
--batch_size=24 \
--net=hrnet \
--transfer-dataset \
--source 1 \
--target 0 \
--name="$tag"
