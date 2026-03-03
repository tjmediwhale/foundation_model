"""
foundation_model main.py - 유일한 진입점.
모드 선택 및 전체 파이프라인 오케스트레이션.
accelerate 지원.
"""
import os
import sys

# torchrun 분산 실행 시 에러 traceback 캡처용
try:
    from torch.distributed.elastic.multiprocessing.errors import record
except ImportError:
    def record(fn):
        return fn
import warnings
import logging

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*DTensor.*RNG.*", category=DeprecationWarning)
logging.captureWarnings(True)
# DTensor RNG 동기화 확인: _random은 WARNING 허용 (동기화 OK면 경고 안 뜸)
for _name in ("torch", "torch.distributed", "torch.distributed.elastic", "torch.distributed.tensor", "dinov3"):
    logging.getLogger(_name).setLevel(logging.ERROR)
logging.getLogger("torch.distributed.tensor._random").setLevel(logging.WARNING)
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
                   choices=["train_dinov3_ssl_then_lp", "eval_lp_retfound_baseline", "eval_lp_pretrained_dinov3", "eval_lp_pretrained_dinov3_hf"])
    p.add_argument("--config", default="scripts/configs/default.yaml", help="YAML config path")
    p.add_argument("--run_name", default="run_default", help="output/{run_name}/")
    p.add_argument("--output_dir", default=None, help="Override output root")
    p.add_argument("--ckpt", default=None, help="eval_lp_pretrained_dinov3: 체크포인트 경로 (config 우선)")
    return p.parse_args()


@record
def main():
    args = parse_args()
    import yaml

    # eval 모드: 분산 미사용 → rank 0만 실행 (run.sh가 1 프로세스로 실행하면 문제없음)
    if args.mode.startswith("eval_"):
        rank = int(os.environ.get("RANK", 0))
        if rank != 0:
            return
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
        train_csv = cfg["data"]["train_csv"]
        val_csv = cfg["data"]["val_csv"]
        test_csv = cfg["data"]["test_csv"]
        tasks = cfg.get("lp", {}).get("tasks", ["dr", "amd", "glaucoma"])
        # run_lp_retfound 내부에서 create_filtered_csv_for_lp_task 호출함
        # RETFound pretrained weight로 baseline
        lp_cfg = cfg.get("lp", {})
        retfound_ckpt = lp_cfg.get("retfound_pretrained", "RETFound_dinov2_meh")
        # RETFound 모델 타입 결정 (retfound_pretrained 이름에서 추론)
        if "dinov2" in retfound_ckpt.lower():
            model = "RETFound_dinov2"
            model_arch = lp_cfg.get("model_arch", "dinov2_vitb16")
        elif "mae" in retfound_ckpt.lower():
            model = "RETFound_mae"
            model_arch = lp_cfg.get("model_arch", "vit_base")
        else:
            model = "RETFound_dinov2"
            model_arch = lp_cfg.get("model_arch", "dinov2_vitb16")
        
        summary, ok = run_lp_retfound(
            run_dir, 0, retfound_ckpt, train_csv, val_csv, test_csv, tasks,
            image_column=cfg["data"].get("image_column", "jpg_h1024_path"),
            batch_size=lp_cfg.get("batch_size", 24),
            epochs=lp_cfg.get("epochs", 50),
            num_processes=lp_cfg.get("num_processes") or 1,
            model=model,
            model_arch=model_arch,
            retfound_pretrained=retfound_ckpt,
            warmup_epochs=lp_cfg.get("warmup_epochs", 10),
            nb_classes_map=lp_cfg.get("nb_classes", {}),
            local_prefix=cfg["data"].get("local_prefix", "/nas/mediwhale_processed_data/"),
            use_drnoon_preprocess=cfg.get("data", {}).get("preprocessing", {}).get("use_drnoon_preprocess", True),
            drnoon_precrop=cfg.get("data", {}).get("preprocessing", {}).get("drnoon_precrop", 0.4),
            drnoon_circle_mask=cfg.get("data", {}).get("preprocessing", {}).get("drnoon_circle_mask", True),
            gpu_id=lp_cfg.get("gpu_id"),
        )
        print(json.dumps(summary, indent=2))
    elif args.mode == "eval_lp_pretrained_dinov3":
        lp_cfg = cfg.get("lp", {})
        ckpt = lp_cfg.get("dinov3_ckpt") or getattr(args, "ckpt", None)
        if not ckpt:
            print("체크포인트 경로 필요: --ckpt path/to/best.pt 또는 config에 lp.dinov3_ckpt")
            return
        if not os.path.isabs(ckpt):
            ckpt = str((PROJECT_ROOT / ckpt).resolve())
        if not os.path.isfile(ckpt):
            print(f"[오류] 체크포인트 없음: {ckpt}")
            return
        from engine.lp_engine import run_lp_retfound
        train_csv = cfg["data"]["train_csv"]
        val_csv = cfg["data"]["val_csv"]
        test_csv = cfg["data"]["test_csv"]
        tasks = cfg.get("lp", {}).get("tasks", ["dr", "amd", "glaucoma"])
        # run_lp_retfound 내부에서 create_filtered_csv_for_lp_task 호출함
        lp_cfg = cfg.get("lp", {})
        summary, ok = run_lp_retfound(
            run_dir, 0, ckpt, train_csv, val_csv, test_csv, tasks,
            image_column=cfg["data"].get("image_column", "jpg_h1024_path"),
            batch_size=lp_cfg.get("batch_size", 24),
            epochs=lp_cfg.get("epochs", 50),
            num_processes=lp_cfg.get("num_processes") or 1,
            model="Dinov3",
            model_arch=lp_cfg.get("model_arch", "dinov3_vits16"),
            nb_classes_map=lp_cfg.get("nb_classes", {}),
            warmup_epochs=lp_cfg.get("warmup_epochs", 10),
            local_prefix=cfg["data"].get("local_prefix", "/nas/mediwhale_processed_data/"),
            use_drnoon_preprocess=cfg.get("data", {}).get("preprocessing", {}).get("use_drnoon_preprocess", True),
            drnoon_precrop=cfg.get("data", {}).get("preprocessing", {}).get("drnoon_precrop", 0.4),
            drnoon_circle_mask=cfg.get("data", {}).get("preprocessing", {}).get("drnoon_circle_mask", True),
            gpu_id=lp_cfg.get("gpu_id"),
        )
        print(json.dumps(summary, indent=2))
    elif args.mode == "eval_lp_pretrained_dinov3_hf":
        lp_cfg = cfg.get("lp", {})
        hf_id = lp_cfg.get("dinov3_ckpt") or getattr(args, "ckpt", None)
        if not hf_id:
            print("HuggingFace 모델 ID 필요: config에 lp.dinov3_ckpt (예: facebook/dinov3-vits16-pretrain-lvd1689m)")
            return
        HF_PTH_MAP = {
            "facebook/dinov3-vits16-pretrain-lvd1689m": ("jaychempan/dinov3", "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),
            "facebook/dinov3-vitb16-pretrain-lvd1689m": ("jaychempan/dinov3", "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
            "facebook/dinov3-vitl16-pretrain-lvd1689m": ("jaychempan/dinov3", "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
        }
        if hf_id not in HF_PTH_MAP:
            print(f"지원 모델: {list(HF_PTH_MAP.keys())}")
            return
        try:
            from huggingface_hub import hf_hub_download
            repo_id, filename = HF_PTH_MAP[hf_id]
            ckpt = hf_hub_download(repo_id=repo_id, filename=filename)
            print(f"[LP] HuggingFace에서 다운로드: {repo_id}/{filename}")
        except ImportError:
            raise ImportError("huggingface_hub 필요: pip install huggingface_hub")
        model_arch_map = {
            "facebook/dinov3-vits16-pretrain-lvd1689m": "dinov3_vits16",
            "facebook/dinov3-vitb16-pretrain-lvd1689m": "dinov3_vitb16",
            "facebook/dinov3-vitl16-pretrain-lvd1689m": "dinov3_vitl16",
        }
        model_arch = lp_cfg.get("model_arch") or model_arch_map[hf_id]
        from engine.lp_engine import run_lp_retfound
        train_csv = cfg["data"]["train_csv"]
        val_csv = cfg["data"]["val_csv"]
        test_csv = cfg["data"]["test_csv"]
        tasks = lp_cfg.get("tasks", ["dr", "amd", "glaucoma"])
        summary, ok = run_lp_retfound(
            run_dir, 0, ckpt, train_csv, val_csv, test_csv, tasks,
            image_column=cfg["data"].get("image_column", "jpg_h1024_path"),
            batch_size=lp_cfg.get("batch_size", 24),
            epochs=lp_cfg.get("epochs", 50),
            num_processes=lp_cfg.get("num_processes") or 1,
            model="Dinov3",
            model_arch=model_arch,
            nb_classes_map=lp_cfg.get("nb_classes", {}),
            warmup_epochs=lp_cfg.get("warmup_epochs", 10),
            local_prefix=cfg["data"].get("local_prefix", "/nas/mediwhale_processed_data/"),
            use_drnoon_preprocess=cfg.get("data", {}).get("preprocessing", {}).get("use_drnoon_preprocess", True),
            drnoon_precrop=cfg.get("data", {}).get("preprocessing", {}).get("drnoon_precrop", 0.4),
            drnoon_circle_mask=cfg.get("data", {}).get("preprocessing", {}).get("drnoon_circle_mask", True),
            gpu_id=lp_cfg.get("gpu_id"),
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
