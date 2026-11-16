#!/bin/bash

# 阶段化训练脚本 - 5个阶段渐进式 J1/J2（HF=0.1，Seam 显式关闭）
# 日期: 2025-10-23

set -e

WORKSPACE="checkpoints/mega/j1j2/460"
VERSION="staged_460_J1J2_$(date +%m%d_%H%M)"
SESSION_NAME="stagedJ1J2_$(date +%H%M%S)"

echo "=== 阶段化训练启动 ==="
echo "创建screen会话: $SESSION_NAME"

mkdir -p $WORKSPACE/$VERSION
mkdir -p reports/j1j2_460/j1_vis/plan1
mkdir -p reports/j1j2_460/j2_vis/plan1

screen -dmS "$SESSION_NAME" bash -c "
source /home/wyq/416/anaconda3/etc/profile.d/conda.sh
conda activate mega
cd /home/wyq/zsk/work3/J1J2mega

echo '=== Phase 1: Epoch 0-10 (J1+HF；无J2/Seam) ==='
python train.py --config configs/nersemble/460/full.yaml --workspace $WORKSPACE --version ${VERSION}_phase1 --extra_config '{
  \"training.gpus\": \"1\",
  \"training.learning_rate\": 0.001,
  \"training.epochs\": 10,
  \"training.pretrained_checkpoint_path\": null,
  \"data.per_gpu_batch_size\": 16,
  \"data.num_workers\": 16,
  \"exp.lf_img.enable\": true,
  \"exp.lf_img.lambda\": 0.25,
  \"exp.hf_img.enable\": true,
  \"exp.hf_img.lambda\": 0.1,
  \"exp.lf_uv.enable\": false,
  \"exp.lf_uv.force_bake_now\": false,
  \"exp.lf_uv.loss.enable\": false,
  \"exp.lf_uv.loss.weight\": 0.0,
  \"exp.lf_uv.seam.enable\": false
}' 2>&1 | tee $WORKSPACE/$VERSION/phase1_training.log


echo '=== Phase 2: Epoch 10-30 (J1+J2+HF；Seam 关闭) ==='
python train.py --config configs/nersemble/460/full.yaml --workspace $WORKSPACE --version ${VERSION}_phase2 --extra_config '{
  \"training.gpus\": \"1\",
  \"training.learning_rate\": 0.001,
  \"training.epochs\": 30,
  \"training.pretrained_checkpoint_path\": \"$WORKSPACE/${VERSION}_phase1/checkpoint_latest.pth\",
  \"data.per_gpu_batch_size\": 16,
  \"data.num_workers\": 16,
  \"exp.lf_img.enable\": true,
  \"exp.lf_img.lambda\": 0.15,
  \"exp.hf_img.enable\": true,
  \"exp.hf_img.lambda\": 0.1,
  \"exp.lf_uv.enable\": true,
  \"exp.lf_uv.loss.enable\": true,
  \"exp.lf_uv.loss.weight\": 0.01,
  \"exp.lf_uv.seam.enable\": false
}' 2>&1 | tee $WORKSPACE/$VERSION/phase2_training.log


echo '=== Phase 3: Epoch 30-48 (J1+J2+HF；Seam 关闭) ==='
python train.py --config configs/nersemble/460/full.yaml --workspace $WORKSPACE --version ${VERSION}_phase3 --extra_config '{
  \"training.gpus\": \"1\",
  \"training.learning_rate\": 0.001,
  \"training.epochs\": 48,
  \"training.pretrained_checkpoint_path\": \"$WORKSPACE/${VERSION}_phase2/checkpoint_latest.pth\",
  \"data.per_gpu_batch_size\": 16,
  \"data.num_workers\": 16,
  \"exp.lf_img.enable\": true,
  \"exp.lf_img.lambda\": 0.10,
  \"exp.hf_img.enable\": true,
  \"exp.hf_img.lambda\": 0.1,
  \"exp.lf_uv.enable\": true,
  \"exp.lf_uv.loss.enable\": true,
  \"exp.lf_uv.loss.weight\": 0.02,
  \"exp.lf_uv.seam.enable\": false
}' 2>&1 | tee $WORKSPACE/$VERSION/phase3_training.log


echo '=== Phase 4: Epoch 49-70 (J1+J2+HF；Seam 关闭) ==='
python train.py --config configs/nersemble/460/full.yaml --workspace $WORKSPACE --version ${VERSION}_phase4 --extra_config '{
  \"training.gpus\": \"1\",
  \"training.learning_rate\": 0.0005,
  \"training.epochs\": 70,
  \"training.pretrained_checkpoint_path\": \"$WORKSPACE/${VERSION}_phase3/checkpoint_latest.pth\",
  \"data.per_gpu_batch_size\": 8,
  \"data.num_workers\": 16,
  \"exp.lf_img.enable\": true,
  \"exp.lf_img.lambda\": 0.05,
  \"exp.hf_img.enable\": true,
  \"exp.hf_img.lambda\": 0.1,
  \"exp.lf_uv.enable\": true,
  \"exp.lf_uv.loss.enable\": true,
  \"exp.lf_uv.loss.weight\": 0.03,
  \"exp.lf_uv.seam.enable\": false
}' 2>&1 | tee $WORKSPACE/$VERSION/phase4_training.log


echo '=== Phase 5: Epoch 70-80 (J2+HF；Seam 关闭) ==='
python train.py --config configs/nersemble/460/full.yaml --workspace $WORKSPACE --version ${VERSION}_phase5 --extra_config '{
  \"training.gpus\": \"1\",
  \"training.learning_rate\": 0.0005,
  \"training.epochs\": 80,
  \"training.pretrained_checkpoint_path\": \"$WORKSPACE/${VERSION}_phase4/checkpoint_latest.pth\",
  \"data.per_gpu_batch_size\": 8,
  \"data.num_workers\": 16,
  \"exp.lf_img.enable\": false,
  \"exp.lf_img.lambda\": 0.0,
  \"exp.hf_img.enable\": true,
  \"exp.hf_img.lambda\": 0.1,
  \"exp.lf_uv.enable\": true,
  \"exp.lf_uv.loss.enable\": true,
  \"exp.lf_uv.loss.weight\": 0.03,
  \"exp.lf_uv.seam.enable\": false
}' 2>&1 | tee $WORKSPACE/$VERSION/phase5_training.log

echo '=== 所有阶段完成 ==='
"

echo "训练已在 screen 会话中启动: $SESSION_NAME"
echo "使用以下命令查看训练状态:"
echo "  screen -r $SESSION_NAME"
echo "  screen -list"
echo ""
echo "训练日志保存在: $WORKSPACE/$VERSION/"
echo ""
echo "阶段摘要:"
echo "Phase 1: J1+HF (无J2/Seam)"
echo "Phase 2: J1+J2+HF, 无Seam"
echo "Phase 3: J1+J2+HF, 无Seam"
echo "Phase 4: J1+J2+HF, 无Seam"
echo "Phase 5: J2+HF, 无Seam"
