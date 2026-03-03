#!/bin/bash
# foundation_model 실행 스크립트
# torchrun 또는 accelerate launch로 main.py 실행

export PYTHONWARNINGS="ignore"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="${1:-scripts/configs/default.yaml}"
RUN_NAME="${2:-dinov3_vitb16_finetune_2nd}"
MODE="${3:-train_dinov3_ssl_then_lp}"

cd "$PROJECT_ROOT"

# GPU 개수 (기본 1). 여러 GPU: NUM_PROCESSES=8 bash scripts/run.sh
NUM_PROCESSES="${NUM_PROCESSES:-8}"

# accelerate 있으면 사용, 없으면 torchrun 사용
if python -c "import accelerate" 2>/dev/null; then
  python -m accelerate launch --num_processes=$NUM_PROCESSES main.py \
    --config "$CONFIG" \
    --run_name "$RUN_NAME" \
    --mode "$MODE"
else
  torchrun --nproc_per_node=$NUM_PROCESSES main.py \
    --config "$CONFIG" \
    --run_name "$RUN_NAME" \
    --mode "$MODE"
fi
