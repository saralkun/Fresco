#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

HEAD_CKPT_PATH="/app/data/CODE-file/LACE1/checkpoints/mega/j1j2/253/staged_253_phase5_seam_1106_0815_phase5/checkpoint_latest.pth"
MOTIONFILE="/app/data/nersemble/104/UNION10_104_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine/transforms_train.json"
NAME="cross-104-253"

# Refer to ./scripts/alter_hair.sh for detailed command line arguments
python funny_demo/render_funny.py \
    --checkpoint $HEAD_CKPT_PATH \
    --motionfile $MOTIONFILE \
    --name $NAME --with_pose