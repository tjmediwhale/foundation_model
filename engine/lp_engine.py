"""
foundation_model engine/lp_engine.py
RETFound main_finetune.py 서브프로세스 래퍼.
accelerate/torchrun으로 분산 학습 지원.
"""
import os
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RETFOUND_DIR = PROJECT_ROOT / "model" / "RETFound"

# 분산 실행 시 부모의 RANK/WORLD_SIZE 상속 방지 (torchrun/accelerate가 자식 프로세스에 새로 설정)
# CUDA_VISIBLE_DEVICES: 부모(rank 0)가 GPU 0만 보도록 설정돼 있으면 LP도 1 GPU만 사용 → 제거하여 전체 GPU 노출
_DIST_ENV_KEYS = (
    "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
    "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID",
    "CUDA_VISIBLE_DEVICES",
)


def _has_accelerate() -> bool:
    try:
        import accelerate  # noqa: F401
        return True
    except ImportError:
        return False


def run_lp_retfound(
    run_dir: str,
    trigger_idx: int,
    ckpt_path: str,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    tasks: List[str],
    image_column: str = "jpg_h1024_path",
    batch_size: int = 24,
    epochs: int = 50,
    nb_classes_map: Dict[str, int] = None,
    retfound_pretrained: str = "RETFound_dinov2_meh",
    local_prefix: str = "/nas/mediwhale_processed_data/",
    warmup_epochs: int = 10,
    model: str = "Dinov3",
    model_arch: str = "dinov3_vitl16",
    num_processes: int = 1,
    use_drnoon_preprocess: bool = True,
    drnoon_precrop: float = 0.4,
    drnoon_circle_mask: bool = True,
) -> Tuple[Dict, bool]:
    """
    태스크별 LP 실행. ImageFolder 생성 후 RETFound main_finetune 서브프로세스 호출.
    Returns: (summary_dict, sota_ok)
    """
    from utils.data import create_filtered_csv_for_lp_task, build_imagefolder_from_csv
    from tqdm import tqdm

    print("[LP mode] Linear Probing 시작")
    nb_classes_map = nb_classes_map or {}
    summary = {}
    for task in tasks:
        pbar = tqdm(desc=f"[LP] task={task}", total=1, unit="task", leave=True)
        nb = nb_classes_map.get(task, 2)
        tmp_csv = os.path.join(run_dir, "tmp", "lp_csv", task)
        os.makedirs(tmp_csv, exist_ok=True)
        t, v, te = create_filtered_csv_for_lp_task(
            train_csv, val_csv, test_csv, task, tmp_csv, image_column, local_prefix
        )
        data_root = os.path.join(run_dir, "tmp", "lp_data", task)
        os.makedirs(data_root, exist_ok=True)
        n_train = build_imagefolder_from_csv(t, image_column, task, data_root, "train", local_prefix)
        build_imagefolder_from_csv(v, image_column, task, data_root, "val", local_prefix)
        build_imagefolder_from_csv(te, image_column, task, data_root, "test", local_prefix)

        if n_train == 0:
            tqdm.write(f"[LP] task={task}: train 이미지 0개 (경로 미존재?). 스킵.")
            pbar.update(1)
            pbar.close()
            continue

        # RETFound가 output_dir/task 경로 사용 → output_dir에 task 포함하면 dr/dr 중복됨
        out_dir_root = os.path.join(run_dir, "lp_results", f"trigger_{trigger_idx}")
        os.makedirs(out_dir_root, exist_ok=True)

        tqdm.write(f"[LP] task={task} (trigger_{trigger_idx}) num_processes={num_processes}")
        base_args = [
            "--model", model,
            "--model_arch", model_arch,
            "--data_path", data_root,
            "--nb_classes", str(nb),
            "--finetune", ckpt_path,
            "--adaptation", "lp",
            "--batch_size", str(batch_size),
            "--epochs", str(epochs),
            "--warmup_epochs", str(warmup_epochs),
            "--output_dir", out_dir_root,
            "--task", task,
        ]
        if use_drnoon_preprocess:
            base_args.extend([
                "--use_drnoon_preprocess",
                "--drnoon_precrop", str(drnoon_precrop),
                "--drnoon_circle_mask", str(drnoon_circle_mask),
            ])
        if num_processes > 1:
            base_args.append("--dist_eval")
        main_script = str(RETFOUND_DIR / "main_finetune.py")

        # 부모(SSL)가 29500 사용 중 → LP는 다른 포트 사용 (포트 충돌 방지)
        lp_port = 29600 + (trigger_idx * 10) + (abs(hash(task)) % 90) if num_processes > 1 else None

        if num_processes > 1:
            # accelerate 있으면 사용, 없으면 torchrun
            if _has_accelerate():
                cmd = [
                    "python", "-m", "accelerate", "launch",
                    "--num_processes", str(num_processes),
                    "--main_process_port", str(lp_port),
                    main_script,
                ] + base_args
            else:
                cmd = [
                    "torchrun", "--nproc_per_node", str(num_processes),
                    "--master_port", str(lp_port),
                    main_script,
                ] + base_args
        else:
            cmd = ["python", main_script] + base_args

        env = os.environ.copy()
        for k in _DIST_ENV_KEYS:
            env.pop(k, None)
        # 분산 실행 시 에러 traceback 파일로 저장 (ChildFailedError 원인 확인용)
        if num_processes > 1:
            err_file = os.path.join(run_dir, "tmp", f"lp_error_trigger{trigger_idx}_{task}.txt")
            os.makedirs(os.path.dirname(err_file), exist_ok=True)
            env["TORCH_ELASTIC_ERROR_FILE"] = err_file
        try:
            subprocess.run(cmd, cwd=str(RETFOUND_DIR), env=env, check=True)
        except subprocess.CalledProcessError as e:
            summary[task] = {"error": str(e)}
            pbar.update(1)
            pbar.close()
            continue

        pbar.update(1)
        pbar.close()
        log_file = os.path.join(out_dir_root, task, "log.txt")
        if os.path.exists(log_file):
            with open(log_file) as f:
                summary[task] = {"log": f.read()[-2000:]}
        score_file = os.path.join(out_dir_root, task, "best_score.json")
        if os.path.exists(score_file):
            with open(score_file) as f:
                summary[task]["score"] = json.load(f).get("score", 0.0)

    # 3개 task 모두 SOTA(새 best) 찍었으면 foundation model 별도 저장
    _maybe_save_all_sota_foundation(run_dir, ckpt_path, tasks, summary)

    return summary, True


def _maybe_save_all_sota_foundation(run_dir: str, ckpt_path: str, tasks: list, summary: dict) -> None:
    """3개 task 모두 이번 trigger에서 새 best 달성 시 foundation model(best.pt)을 best_all_sota.pt로 복사."""
    history_file = os.path.join(run_dir, "lp_best_scores.json")
    try:
        with open(history_file) as f:
            best_so_far = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        best_so_far = {t: -1.0 for t in tasks}

    all_improved = True
    for task in tasks:
        score = summary.get(task, {}).get("score")
        if score is None:
            all_improved = False
            break
        if score <= best_so_far.get(task, -1.0):
            all_improved = False
            break
        best_so_far[task] = float(score)

    if all_improved and os.path.isfile(ckpt_path):
        import shutil
        dest = os.path.join(os.path.dirname(ckpt_path), "best_all_sota.pt")
        shutil.copy2(ckpt_path, dest)
        print(f"[LP] 3 tasks 모두 SOTA → foundation model 저장: {dest}")
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    with open(history_file, "w") as f:
        json.dump(best_so_far, f, indent=2)
