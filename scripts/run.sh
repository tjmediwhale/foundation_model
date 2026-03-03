#!/bin/bash
# foundation_model 실행 스크립트
# torchrun 또는 accelerate launch로 main.py 실행

export PYTHONWARNINGS="ignore"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="${1:-scripts/configs/default.yaml}"
RUN_NAME="${2:-retfound_baseline_lp}" #dinov3_vits16_finetune_3rd, retfound_baseline_lp
MODE="${3:-eval_lp_retfound_baseline}" #train_dinov3_ssl_then_lp, eval_lp_retfound_baseline, eval_lp_pretrained_dinov3

cd "$PROJECT_ROOT"

# eval 모드는 main.py가 분산 학습 안 함 → 1 프로세스만. train 모드는 8 GPU 사용
if [[ "$MODE" == eval_* ]]; then
  NUM_PROCESSES="${NUM_PROCESSES:-1}"
else
  NUM_PROCESSES="${NUM_PROCESSES:-8}"
fi

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
