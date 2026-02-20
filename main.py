"""
foundation_model main.py - 유일한 진입점.
모드 선택 및 전체 파이프라인 오케스트레이션.
accelerate 지원.
"""
import os
import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
DINOV3_PATH = PROJECT_ROOT / "model" / "dinov3"
sys.path.insert(0, str(DINOV3_PATH))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="train_dinov3_ssl_then_lp",
                   choices=["train_dinov3_ssl_then_lp", "eval_lp_retfound_baseline", "eval_lp_pretrained_dinov3"])
    p.add_argument("--config", default="scripts/configs/default.yaml", help="YAML config path")
    p.add_argument("--run_name", default="run_default", help="output/{run_name}/")
    p.add_argument("--output_dir", default=None, help="Override output root")
    return p.parse_args()


def main():
    args = parse_args()
    import yaml
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_root = args.output_dir or os.path.join(PROJECT_ROOT, "output")
    run_dir = os.path.join(output_root, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "tmp"), exist_ok=True)

    with open(os.path.join(run_dir, "config_snapshot.yaml"), "w") as f:
        yaml.dump(cfg, f)

    if args.mode == "train_dinov3_ssl_then_lp":
        from engine.train_loop import run_train_dinov3_ssl_then_lp
        run_train_dinov3_ssl_then_lp(cfg, run_dir, args)
    elif args.mode == "eval_lp_retfound_baseline":
        from engine.lp_engine import run_lp_retfound
        from utils.data import load_csv_with_path_replace, create_filtered_csv_for_lp_task
        train_csv = cfg["data"]["train_csv"]
        val_csv = cfg["data"]["val_csv"]
        test_csv = cfg["data"]["test_csv"]
        tasks = cfg.get("lp", {}).get("tasks", ["dr", "amd", "glaucoma"])
        tmp_dir = os.path.join(run_dir, "tmp", "baseline_csv")
        os.makedirs(tmp_dir, exist_ok=True)
        for task in tasks:
            t, v, te = create_filtered_csv_for_lp_task(
                train_csv, val_csv, test_csv, task, tmp_dir,
                cfg["data"].get("image_column", "jpg_h1024_path"),
            )
        # RETFound pretrained weight로 baseline
        retfound_ckpt = cfg.get("lp", {}).get("retfound_pretrained", "RETFound_dinov2_meh")
        # TODO: implement baseline run with RETFound pretrained
        print("eval_lp_retfound_baseline: placeholder")
    elif args.mode == "eval_lp_pretrained_dinov3":
        ckpt = cfg.get("lp", {}).get("dinov3_ckpt")
        if not ckpt:
            print("--lp.dinov3_ckpt required")
            return
        from engine.lp_engine import run_lp_retfound
        from utils.data import create_filtered_csv_for_lp_task
        train_csv = cfg["data"]["train_csv"]
        val_csv = cfg["data"]["val_csv"]
        test_csv = cfg["data"]["test_csv"]
        tasks = cfg.get("lp", {}).get("tasks", ["dr", "amd", "glaucoma"])
        tmp_dir = os.path.join(run_dir, "tmp", "eval_csv")
        os.makedirs(tmp_dir, exist_ok=True)
        for task in tasks:
            t, v, te = create_filtered_csv_for_lp_task(
                train_csv, val_csv, test_csv, task, tmp_dir,
                cfg["data"].get("image_column", "jpg_h1024_path"),
            )
        summary, ok = run_lp_retfound(
            run_dir, 0, ckpt, t, v, te, tasks,
            cfg["data"].get("image_column", "jpg_h1024_path"),
            cfg.get("lp", {}).get("batch_size", 24),
            cfg.get("lp", {}).get("epochs", 50),
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
