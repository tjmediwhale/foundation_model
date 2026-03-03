#!/bin/bash
# LP with HuggingFace pretrained DINOv3 (no local .pth)
# HuggingFace에서 pretrained weight 다운로드 후 LP 실행
# pip install huggingface_hub 필요

export PYTHONWARNINGS="ignore"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="${1:-scripts/configs/eval_lp_pretrained_hf.yaml}"
RUN_NAME="${2:-lp_pretrained_dinov3_hf}"

cd "$PROJECT_ROOT"

python main.py \
  --config "$CONFIG" \
  --run_name "$RUN_NAME" \
  --mode eval_lp_pretrained_dinov3_hf
