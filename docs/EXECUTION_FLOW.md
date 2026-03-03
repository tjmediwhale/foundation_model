# foundation_model 실행 흐름 분석

`run.sh`, `evaluate.sh`, `evaluate_pretrained_hf.sh` 세 스크립트의 실행 흐름을 코드 단위로 분석한 문서입니다.

---

## 공통: main.py 진입 전 처리

### 1. 환경 변수 및 경로 설정

세 스크립트 모두 다음을 수행합니다:

```bash
export PYTHONWARNINGS="ignore"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
```


| 변수                   | 의미                                          |
| -------------------- | ------------------------------------------- |
| `SCRIPT_DIR`         | 스크립트 파일 위치 (예: `foundation_model/scripts/`) |
| `PROJECT_ROOT`       | 프로젝트 루트 (예: `foundation_model/`)            |
| `cd "$PROJECT_ROOT"` | 작업 디렉터리를 프로젝트 루트로 변경                        |


### 2. main.py 공통 초기화 (모든 모드)

```python
# main.py 47-69행
args = parse_args()
cfg = yaml.safe_load(config_path)

output_root = args.output_dir or os.path.join(PROJECT_ROOT, "output")
run_dir = os.path.join(output_root, args.run_name)  # output/{run_name}/
os.makedirs(run_dir, exist_ok=True)
os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(run_dir, "tmp"), exist_ok=True)

with open(os.path.join(run_dir, "config_snapshot.yaml"), "w") as f:
    yaml.dump(cfg, f)
```

- `run_dir`: `output/{run_name}/`
- `config_snapshot.yaml`: 현재 실행에 사용된 config 백업

### 3. eval 모드에서의 rank 체크

```python
# main.py 52-55행
if args.mode.startswith("eval_"):
    rank = int(os.environ.get("RANK", 0))
    if rank != 0:
        return  # rank 1 이상은 즉시 종료
```

- `eval_*` 모드: `RANK`가 0이 아니면 바로 종료 (분산 실행 시 rank 0만 실행)

---

## 1. run.sh

### 1.1 스크립트 구조

```bash
# run.sh 1-33행
CONFIG="${1:-scripts/configs/default.yaml}"
RUN_NAME="${2:-retfound_baseline_lp}"
MODE="${3:-eval_lp_retfound_baseline}"

if [[ "$MODE" == eval_* ]]; then
  NUM_PROCESSES="${NUM_PROCESSES:-1}"
else
  NUM_PROCESSES="${NUM_PROCESSES:-8}"
fi

if python -c "import accelerate" 2>/dev/null; then
  python -m accelerate launch --num_processes=$NUM_PROCESSES main.py \
    --config "$CONFIG" --run_name "$RUN_NAME" --mode "$MODE"
else
  torchrun --nproc_per_node=$NUM_PROCESSES main.py \
    --config "$CONFIG" --run_name "$RUN_NAME" --mode "$MODE"
fi
```

### 1.2 기본값


| 인자              | 기본값                            | 설명         |
| --------------- | ------------------------------ | ---------- |
| `$1` (CONFIG)   | `scripts/configs/default.yaml` | 설정 파일      |
| `$2` (RUN_NAME) | `retfound_baseline_lp`         | 출력 디렉터리 이름 |
| `$3` (MODE)     | `eval_lp_retfound_baseline`    | 실행 모드      |


### 1.3 실행 방식 분기


| MODE 패턴       | NUM_PROCESSES | 실행 방식                                                                  |
| ------------- | ------------- | ---------------------------------------------------------------------- |
| `eval_*`      | 1             | `accelerate launch --num_processes=1` 또는 `torchrun --nproc_per_node=1` |
| 그 외 (train 등) | 8             | `accelerate launch --num_processes=8` 또는 `torchrun --nproc_per_node=8` |


- `accelerate` 있으면 `accelerate launch` 사용, 없으면 `torchrun` 사용

### 1.4 run.sh로 가능한 모드


| MODE                        | 설명                           |
| --------------------------- | ---------------------------- |
| `train_dinov3_ssl_then_lp`  | SSL 학습 → LP                  |
| `eval_lp_retfound_baseline` | RETFound pretrained로 LP (기본) |
| `eval_lp_pretrained_dinov3` | 로컬 DINOv3 체크포인트로 LP          |


### 1.5 run.sh 기본 실행 시 (MODE=eval_lp_retfound_baseline)

```
run.sh (인자 없음)
  → CONFIG=default.yaml, RUN_NAME=retfound_baseline_lp, MODE=eval_lp_retfound_baseline
  → eval_* 이므로 NUM_PROCESSES=1
  → accelerate launch --num_processes=1 main.py --config default.yaml --run_name retfound_baseline_lp --mode eval_lp_retfound_baseline
```

### 1.6 main.py 분기: eval_lp_retfound_baseline

```python
# main.py 74-112행
elif args.mode == "eval_lp_retfound_baseline":
    lp_cfg = cfg.get("lp", {})
    retfound_ckpt = lp_cfg.get("retfound_pretrained", "RETFound_dinov2_meh")
    
    # retfound_pretrained 이름으로 모델 타입 결정
    if "dinov2" in retfound_ckpt.lower():
        model = "RETFound_dinov2"
        model_arch = lp_cfg.get("model_arch", "dinov2_vitb16")
    elif "mae" in retfound_ckpt.lower():
        model = "RETFound_mae"
        model_arch = lp_cfg.get("model_arch", "vit_base")
    else:
        model = "RETFound_dinov2"
        model_arch = lp_cfg.get("model_arch", "dinov2_vitb16")
    
    run_lp_retfound(
        run_dir, 0, retfound_ckpt,  # ckpt_path = "RETFound_dinov2_meh" (HuggingFace ID 또는 로컬)
        train_csv, val_csv, test_csv, tasks,
        model=model, model_arch=model_arch,
        ...
    )
```

- `retfound_ckpt`: RETFound pretrained ID (예: `RETFound_dinov2_meh`)
- `model`: `RETFound_dinov2` 또는 `RETFound_mae`
- `run_lp_retfound`에 `ckpt_path=retfound_ckpt` 전달

### 1.7 run_lp_retfound 호출 (RETFound baseline)

- `model="RETFound_dinov2"` 또는 `"RETFound_mae"`
- `ckpt_path="RETFound_dinov2_meh"` → main_finetune.py에서 HuggingFace 또는 로컬 .pth로 처리
- `main_finetune.py`는 `args.model in ["RETFound_dinov2", "RETFound_mae"]` 분기로 체크포인트 로드

---

## 2. evaluate.sh

### 2.1 스크립트 구조

```bash
# evaluate.sh 1-35행
CONFIG="${1:-scripts/configs/eval.yaml}"
RUN_NAME="${2:-eval_dinov3_baseline_lp}"
MODE="${3:-eval_lp_pretrained_dinov3}"

if [[ "$MODE" == eval_* ]]; then
  python main.py \
    --config "$CONFIG" \
    --run_name "$RUN_NAME" \
    --mode "$MODE"
else
  NUM_PROCESSES="${NUM_PROCESSES:-8}"
  if python -c "import accelerate" 2>/dev/null; then
    python -m accelerate launch --num_processes=$NUM_PROCESSES main.py ...
  else
    torchrun --nproc_per_node=$NUM_PROCESSES main.py ...
  fi
fi
```

### 2.2 run.sh와의 차이


| 항목          | run.sh                        | evaluate.sh                 |
| ----------- | ----------------------------- | --------------------------- |
| 기본 CONFIG   | `default.yaml`                | `eval.yaml`                 |
| 기본 RUN_NAME | `retfound_baseline_lp`        | `eval_dinov3_baseline_lp`   |
| 기본 MODE     | `eval_lp_retfound_baseline`   | `eval_lp_pretrained_dinov3` |
| eval 모드 실행  | `accelerate/torchrun` (1프로세스) | `python main.py` (직접 실행)    |
| 목적          | RETFound baseline LP 등        | 로컬 학습된 DINOv3로 LP           |


### 2.3 eval 모드 실행 방식

```bash
# evaluate.sh 16-19행
if [[ "$MODE" == eval_* ]]; then
  python main.py --config "$CONFIG" --run_name "$RUN_NAME" --mode "$MODE"
```

- `eval_*`이면 `accelerate`/`torchrun` 없이 `python main.py`만 실행
- 포트 29500 사용 없음 → EADDRINUSE 방지

### 2.4 evaluate.sh 기본 실행 시

```
evaluate.sh (인자 없음)
  → CONFIG=eval.yaml, RUN_NAME=eval_dinov3_baseline_lp, MODE=eval_lp_pretrained_dinov3
  → eval_* 이므로 python main.py 직접 실행
  → python main.py --config eval.yaml --run_name eval_dinov3_baseline_lp --mode eval_lp_pretrained_dinov3
```

### 2.5 main.py 분기: eval_lp_pretrained_dinov3

```python
# main.py 113-146행
elif args.mode == "eval_lp_pretrained_dinov3":
    lp_cfg = cfg.get("lp", {})
    ckpt = lp_cfg.get("dinov3_ckpt") or getattr(args, "ckpt", None)
    
    if not ckpt:
        print("체크포인트 경로 필요: ...")
        return
    if not os.path.isabs(ckpt):
        ckpt = str((PROJECT_ROOT / ckpt).resolve())
    if not os.path.isfile(ckpt):
        print(f"[오류] 체크포인트 없음: {ckpt}")
        return
    
    run_lp_retfound(
        run_dir, 0, ckpt,  # ckpt = 로컬 .pt/.pth 파일 경로
        ...,
        model="Dinov3",
        model_arch=lp_cfg.get("model_arch", "dinov3_vits16"),
        ...
    )
```

- `ckpt`: `lp.dinov3_ckpt` 또는 `--ckpt` (로컬 파일 경로)
- 상대 경로 → `PROJECT_ROOT` 기준 절대 경로로 변환
- 파일이 없으면 에러 후 종료
- `model="Dinov3"`, `model_arch`는 config에서 읽음

### 2.6 eval.yaml에서의 체크포인트

```yaml
# eval.yaml 56행
lp:
  dinov3_ckpt: "/home/tj/Research/foundation_model/output/dinov3_vits16_finetune_original/checkpoints/best.pt"
```

- SSL/finetune으로 학습한 `best.pt` 경로 지정

### 2.7 run_lp_retfound 호출 (로컬 DINOv3)

- `model="Dinov3"`
- `ckpt_path`: 로컬 `best.pt` 절대 경로
- main_finetune.py에서 `args.model in ["Dinov3", "Dinov2"]` 분기로 `student.backbone.*` 키 추출 후 로드

---

## 3. evaluate_pretrained_hf.sh

### 3.1 스크립트 구조

```bash
# evaluate_pretrained_hf.sh 1-18행
CONFIG="${1:-scripts/configs/eval_lp_pretrained_hf.yaml}"
RUN_NAME="${2:-lp_pretrained_dinov3_hf}"

python main.py \
  --config "$CONFIG" \
  --run_name "$RUN_NAME" \
  --mode eval_lp_pretrained_dinov3_hf
```

- MODE는 항상 `eval_lp_pretrained_dinov3_hf`로 고정
- `accelerate`/`torchrun` 없이 `python main.py`만 실행

### 3.2 기본값


| 인자       | 기본값                                          |
| -------- | -------------------------------------------- |
| CONFIG   | `scripts/configs/eval_lp_pretrained_hf.yaml` |
| RUN_NAME | `lp_pretrained_dinov3_hf`                    |


### 3.3 main.py 분기: eval_lp_pretrained_dinov3_hf

```python
# main.py 149-196행
elif args.mode == "eval_lp_pretrained_dinov3_hf":
    lp_cfg = cfg.get("lp", {})
    hf_id = lp_cfg.get("dinov3_ckpt") or getattr(args, "ckpt", None)
    
    if not hf_id:
        print("HuggingFace 모델 ID 필요: ...")
        return
    
    HF_PTH_MAP = {
        "facebook/dinov3-vits16-pretrain-lvd1689m": ("jaychempan/dinov3", "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),
        "facebook/dinov3-vitb16-pretrain-lvd1689m": ("jaychempan/dinov3", "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
        "facebook/dinov3-vitl16-pretrain-lvd1689m": ("jaychempan/dinov3", "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
    }
    if hf_id not in HF_PTH_MAP:
        print(f"지원 모델: {list(HF_PTH_MAP.keys())}")
        return
    
    from huggingface_hub import hf_hub_download
    repo_id, filename = HF_PTH_MAP[hf_id]
    ckpt = hf_hub_download(repo_id=repo_id, filename=filename)  # 로컬 캐시 경로 반환
    
    model_arch_map = {
        "facebook/dinov3-vits16-pretrain-lvd1689m": "dinov3_vits16",
        ...
    }
    model_arch = lp_cfg.get("model_arch") or model_arch_map[hf_id]
    
    run_lp_retfound(run_dir, 0, ckpt, ..., model="Dinov3", model_arch=model_arch, ...)
```

- `hf_id`: HuggingFace 모델 ID (예: `facebook/dinov3-vits16-pretrain-lvd1689m`)
- `HF_PTH_MAP`으로 `(repo_id, filename)` 조회
- `hf_hub_download`로 `.pth` 다운로드 → 로컬 경로 `ckpt` 반환
- `model_arch`는 config 또는 `model_arch_map`에서 결정
- 이후 `run_lp_retfound`는 로컬 `ckpt` 경로로 호출

### 3.4 eval_lp_pretrained_hf.yaml

```yaml
lp:
  dinov3_ckpt: "facebook/dinov3-vits16-pretrain-lvd1689m"  # HuggingFace ID (파일 경로 아님)
  model_arch: "dinov3_vits16"
```

- `dinov3_ckpt`는 HuggingFace ID만 허용 (로컬 경로 아님)

---

## 공통: run_lp_retfound (lp_engine.py)

세 모드 모두 최종적으로 `run_lp_retfound`를 호출합니다. `ckpt_path`와 `model`/`model_arch`만 다릅니다.

### run_lp_retfound 흐름

```python
# lp_engine.py 64-165행
for task in tasks:  # ["dr", "amd", "glaucoma"]
    # 1. 태스크별 CSV 생성
    t, v, te = create_filtered_csv_for_lp_task(train_csv, val_csv, test_csv, task, ...)
    
    # 2. ImageFolder 데이터셋 생성
    n_train = build_imagefolder_from_csv(t, ..., "train", ...)
    build_imagefolder_from_csv(v, ..., "val", ...)
    build_imagefolder_from_csv(te, ..., "test", ...)
    
    if n_train == 0:
        continue
    
    # 3. main_finetune.py 서브프로세스 실행
    base_args = [
        "--model", model,
        "--model_arch", model_arch,
        "--finetune", ckpt_path,
        "--adaptation", "lp",
        "--input_size", "224",
        ...
    ]
    cmd = ["python", main_script] + base_args  # num_processes=1
    # 또는 torchrun/accelerate + base_args  # num_processes>1
    
    env = os.environ.copy()
    for k in _DIST_ENV_KEYS:
        env.pop(k, None)
    if num_processes == 1 and gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    subprocess.run(cmd, cwd=RETFOUND_DIR, env=env, check=True)
    
    # 4. 결과 수집
    summary[task] = {"log": ..., "score": ...}

_maybe_save_all_sota_foundation(run_dir, ckpt_path, tasks, summary)
return summary, True
```

### 서브프로세스 실행 방식


| num_processes | 실행 명령                                                                                                 |
| ------------- | ----------------------------------------------------------------------------------------------------- |
| 1             | `python model/RETFound/main_finetune.py` + base_args                                                  |
| >1            | `torchrun --nproc_per_node=N` 또는 `accelerate launch --num_processes=N` + main_finetune.py + base_args |


### GPU 지정 (num_processes=1)

```python
if num_processes == 1 and gpu_id is not None:
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
```

- `gpu_id`가 있으면 해당 GPU만 사용

---

## run.sh 전용: train_dinov3_ssl_then_lp

`run.sh`에서 `MODE=train_dinov3_ssl_then_lp`로 실행할 때만 이 경로가 사용됩니다.

### main.py 분기

```python
# main.py 70-72행
if args.mode == "train_dinov3_ssl_then_lp":
    from engine.train_loop import run_train_dinov3_ssl_then_lp
    run_train_dinov3_ssl_then_lp(cfg, run_dir, args)
```

### train_loop 흐름 요약

```
1. DINOv3 config 로드 (retinal_csv_finetune.yaml)
2. SSLMetaArch 모델 생성
3. resume_from_teacher_chkpt가 HuggingFace ID면 hf_hub_download
4. for epoch in range(epochs):
     - train_one_epoch_dinov3_ssl (SSL 학습)
     - run_validation_dinov3_ssl (val_loss)
     - epoch >= lp_warmup 이고 val_loss 개선 시 best.pt 저장
     - early stopping 체크
5. torch.distributed.destroy_process_group()
6. rank 0만: run_lp_retfound(ckpt_path=best.pt, model="Dinov3", ...)
```

- SSL 학습 후 `best.pt` 저장
- LP는 `run_lp_retfound`로 `best.pt`를 `ckpt_path`로 전달

---

## 비교 요약


| 항목        | run.sh                         | evaluate.sh               | evaluate_pretrained_hf.sh    |
| --------- | ------------------------------ | ------------------------- | ---------------------------- |
| 기본 CONFIG | default.yaml                   | eval.yaml                 | eval_lp_pretrained_hf.yaml   |
| 기본 MODE   | eval_lp_retfound_baseline      | eval_lp_pretrained_dinov3 | eval_lp_pretrained_dinov3_hf |
| eval 시 실행 | accelerate/torchrun 1 proc     | python 직접                 | python 직접                    |
| 체크포인트     | RETFound ID 또는 train 후 best.pt | 로컬 best.pt                | HuggingFace ID → 다운로드        |
| model     | RETFound_dinov2/mae 또는 Dinov3  | Dinov3                    | Dinov3                       |


