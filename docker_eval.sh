#!/usr/bin/env bash
# Render LERF text-query masks for all three scenes, then compute mIoU / bIoU.
# Usage: ./docker_eval.sh [gpu_id]
set -euo pipefail

GPU="${1:-0}"
REPO="$(cd "$(dirname "$0")" && pwd)"
PARENT="$(dirname "$REPO")"
IMG="unified-lift:eval"
SCENES=(figurines ramen teatime)

mkdir -p "${REPO}/results"

for s in "${SCENES[@]}"; do
  echo "=== rendering ${s} ==="
  docker run --rm --gpus "device=${GPU}" \
    -e CUDA_VISIBLE_DEVICES=0 -e HF_HOME=/hf \
    -v "${REPO}:/workspace" -v "${PARENT}:/data" -v "${REPO}/.hfcache:/hf" \
    "${IMG}" \
    python render_lerf_mask_unified_lift.py \
      -m "/workspace/output/unifed_lift/${s}" --skip_train --iteration 30000
done

echo "=== evaluating (mIoU / bIoU) ==="
docker run --rm \
  -e GT_ROOT=/data \
  -v "${REPO}:/workspace" -v "${PARENT}:/data" \
  "${IMG}" \
  python script/eval_lerf_mask_unified_lift.py \
    --excel_name /workspace/results/Unifed_Lift \
    --pred_path /workspace/output/unifed_lift

echo "=== results ==="
cat "${REPO}/results/Unifed_Lift.csv"
