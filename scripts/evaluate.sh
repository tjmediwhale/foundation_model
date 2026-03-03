#!/bin/bash
# foundation_model 실행 스크립트
# torchrun 또는 accelerate launch로 main.py 실행

export PYTHONWARNINGS="ignore"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="${1:-scripts/configs/eval.yaml}"
RUN_NAME="${2:-eval_dinov3_baseline_lp}" #dinov3_vits16_finetune_3rd, retfound_baseline_lp
MODE="${3:-eval_lp_pretrained_dinov3}" #eval_dinov3_baseline_lp, eval_dinov3_3rd_lp

cd "$PROJECT_ROOT"

# eval 모드: 분산 미사용 → python 직접 실행 (포트 충돌 방지)
# train 모드: accelerate/torchrun 사용
if [[ "$MODE" == eval_* ]]; then
  python main.py \
    --config "$CONFIG" \
    --run_name "$RUN_NAME" \
    --mode "$MODE"
else
  NUM_PROCESSES="${NUM_PROCESSES:-8}"
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
fi
