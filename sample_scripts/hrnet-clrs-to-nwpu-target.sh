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
--cfg sample_configs/hrnet_384_clrs_to_nwpu_adaptation.yaml \
--cfg-hr HRNet/experiments/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml \
--pretrained output/hrnet-clrs-source-1/T/source_F_epoch_17_eval_16.pt \
--dset nwpu \
--data-path /home/poppfd/data/clrs-nwpu/ \
--batch_size=20 \
--evals-per-epoch=2 \
--net=hrnet \
--transfer-dataset \
--source -1 \
--target 0 \
--name="$tag"
