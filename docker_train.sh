#!/usr/bin/env bash
# Run Unified-Lift training inside the Docker image.
#   ./docker_train.sh <scene> <gpu_id> [extra args for train_unified_lift.py]
# Datasets live one level above the repo (mounted at /data), e.g. /data/figurines.
set -euo pipefail

SCENE="${1:?usage: docker_train.sh <scene> <gpu_id> [extra args]}"
GPU="${2:?usage: docker_train.sh <scene> <gpu_id> [extra args]}"
shift 2

REPO="$(cd "$(dirname "$0")" && pwd)"
DATA_PARENT="$(dirname "$REPO")"

docker run --rm --gpus "device=${GPU}" \
  -e WANDB_MODE=offline \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v "${REPO}:/workspace" \
  -v "${DATA_PARENT}:/data" \
  unified-lift:cu113 \
  python train_unified_lift.py \
    -s "/data/${SCENE}" \
    -m "/workspace/output/unifed_lift/${SCENE}" \
    --config_file config/gaussian_dataset/train.json \
    --train_split \
    --weight_loss 1e-0 \
    "$@"
