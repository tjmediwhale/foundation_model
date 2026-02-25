"""
foundation_model engine/train_loop.py
train_dinov3_ssl_then_lp 모드 전체 루프.
"""
import os
import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DINOV3_PATH = PROJECT_ROOT / "model" / "dinov3"
sys.path.insert(0, str(DINOV3_PATH))


def _setup_mlflow(cfg: dict, run_dir: str, run_name: str):
    """MLflow 초기화 (rank 0만). cfg.mlflow 없거나 비활성화 시 None 반환."""
    try:
        import mlflow
    except ImportError:
        print("[MLflow] skipped: pip install mlflow 필요", flush=True)
        return None
    mlflow_cfg = cfg.get("mlflow") or {}
    if not mlflow_cfg:
        print("[MLflow] skipped: config에 mlflow 섹션 없음", flush=True)
        return None
    tracking_uri = mlflow_cfg.get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    experiment_name = mlflow_cfg.get("experiment_name", "foundation_model")
    mlflow.set_experiment(experiment_name)
    run_name_ = mlflow_cfg.get("run_name") or run_name
    try:
        mlflow.start_run(run_name=run_name_)
    except Exception as e:
        print(f"[MLflow] start_run 실패: {e}", flush=True)
        return None
    print(f"[MLflow] OK | uri={mlflow.get_tracking_uri()} | experiment={experiment_name} | run={run_name_}", flush=True)
    opts = cfg.get("dinov3", {}).get("opts") or []
    arch = "vit_base"
    for o in opts:
        if isinstance(o, str) and "student.arch=" in o:
            arch = o.split("=")[-1]
            break
    mlflow.log_params({
        "run_dir": run_dir,
        "student_arch": arch,
        "batch_size": str(cfg.get("training", {}).get("batch_size", "")),
        "epochs": str(cfg.get("training", {}).get("epochs", "")),
        "resume_from": str(cfg.get("dinov3", {}).get("resume_from_teacher_chkpt", ""))[:64],
    })
    return mlflow


def run_train_dinov3_ssl_then_lp(cfg: dict, run_dir: str, args):
    """DINOv3 SSL 학습 + val_loss 개선 시 LP 트리거."""
    from dinov3.configs.config import setup_job, get_cfg_from_args, apply_scaling_rules_to_cfg
    from dinov3.configs import DinoV3SetupArgs
    from dinov3.train.ssl_meta_arch import SSLMetaArch
    from utils.data import load_csv_with_path_replace
    from engine.train_engine import train_one_epoch_dinov3_ssl
    from engine.val_engine import run_validation_dinov3_ssl
    from engine.lp_engine import run_lp_retfound

    setup_job(output_dir=run_dir, seed=cfg.get("seed", 0))

    import logging
    logging.getLogger("dinov3").setLevel(logging.WARNING)

    import dinov3.distributed as distributed
    rank = distributed.get_rank() if distributed.is_enabled() else 0

    run_name = getattr(args, "run_name", "run_default")
    mlflow_active = False
    if rank == 0:
        _mlflow = _setup_mlflow(cfg, run_dir, run_name)
        mlflow_active = _mlflow is not None

    local_prefix = cfg.get("data", {}).get("local_prefix", "/nas/mediwhale_processed_data/")
    if rank == 0:
        print(f"[Data] local_prefix: {local_prefix}")

    default_config = str(DINOV3_PATH / "dinov3" / "configs" / "ssl_default_config.yaml")
    config_path = cfg.get("dinov3", {}).get("config") or default_config
    if config_path and not os.path.isabs(config_path):
        config_path = str((PROJECT_ROOT / config_path).resolve())
    if not config_path or not os.path.exists(config_path):
        config_path = default_config

    setup_args = DinoV3SetupArgs(
        config_file=config_path,
        output_dir=run_dir,
        opts=cfg.get("dinov3", {}).get("opts", []),
    )
    dinov3_cfg = get_cfg_from_args(setup_args, strict=False)
    dinov3_cfg.train.output_dir = run_dir
    resume_ckpt = cfg.get("dinov3", {}).get("resume_from_teacher_chkpt")
    if resume_ckpt:
        # Hugging Face 모델 ID (org/model) → hf_hub_download으로 자동 다운로드
        if "/" in resume_ckpt and not os.path.isabs(resume_ckpt) and not (PROJECT_ROOT / resume_ckpt).exists():
            HF_PTH_MAP = {
                "facebook/dinov3-vitl16-pretrain-lvd1689m": ("jaychempan/dinov3", "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
                "facebook/dinov3-vitb16-pretrain-lvd1689m": ("jaychempan/dinov3", "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
                "facebook/dinov3-vits16-pretrain-lvd1689m": ("jaychempan/dinov3", "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),
            }
            if resume_ckpt in HF_PTH_MAP:
                try:
                    from huggingface_hub import hf_hub_download
                    repo_id, filename = HF_PTH_MAP[resume_ckpt]
                    resume_ckpt = hf_hub_download(repo_id=repo_id, filename=filename)
                    if rank == 0:
                        print(f"[DINOv3] Hugging Face에서 다운로드: {repo_id}/{filename}")
                except ImportError:
                    raise ImportError("Hugging Face 모델 ID 사용 시 huggingface_hub 필요: pip install huggingface_hub")
        elif not os.path.isabs(resume_ckpt):
            resume_ckpt = str((PROJECT_ROOT / resume_ckpt).resolve())
        dinov3_cfg.student.resume_from_teacher_chkpt = resume_ckpt
        if rank == 0:
            print(f"[DINOv3] resume_from_teacher_chkpt: {resume_ckpt}")
    if distributed.is_enabled():
        apply_scaling_rules_to_cfg(dinov3_cfg)
    dinov3_cfg.train.batch_size_per_gpu = cfg.get("training", {}).get("batch_size", 64)
    dinov3_cfg.train.num_workers = cfg.get("training", {}).get("num_workers", 4)
    dinov3_cfg.optim.epochs = cfg.get("training", {}).get("epochs", 100)

    with torch.device("meta"):
        model = SSLMetaArch(dinov3_cfg)
    model.prepare_for_distributed_training()
    model._apply(
        lambda t: torch.full_like(
            t,
            fill_value=float("nan") if t.dtype.is_floating_point else 0,
            device="cuda",
        ),
        recurse=True,
    )
    model.init_weights()

    train_csv = cfg["data"]["train_csv"]
    val_csv = cfg["data"]["val_csv"]
    image_col = cfg.get("data", {}).get("image_column", "jpg_h1024_path")

    df_train = load_csv_with_path_replace(train_csv, image_col, local_prefix)
    df_val = load_csv_with_path_replace(val_csv, image_col, local_prefix)
    train_paths = df_train[image_col].astype(str).tolist()
    val_paths = df_val[image_col].astype(str).tolist()

    train_paths = [p for p in train_paths if p and str(p).strip() not in ("", "nan")]
    val_paths = [p for p in val_paths if p and str(p).strip() not in ("", "nan")]

    if rank == 0:
        print(f"[Data] train={len(train_paths)}, val={len(val_paths)}")
    if len(train_paths) == 0:
        raise RuntimeError(
            f"유효한 학습 이미지가 없습니다. CSV: {train_csv}, image_col={image_col}. "
            f"gs:// -> {local_prefix} 치환 후 파일이 존재해야 합니다."
        )

    # drnoon-image-transform 전처리 (retinal crop). augmentation 전에 적용.
    preproc_cfg = cfg.get("data", {}).get("preprocessing", {})
    use_drnoon_preprocess = preproc_cfg.get("use_drnoon_preprocess", True)
    preprocess_fn = None
    if use_drnoon_preprocess:
        from utils.preprocessing import get_fundus_preprocess_fn
        preprocess_fn = get_fundus_preprocess_fn(
            precrop=preproc_cfg.get("drnoon_precrop", 0.4),
            circle_mask=preproc_cfg.get("drnoon_circle_mask", True),
        )
        if rank == 0:
            print(f"[Data] drnoon 전처리: precrop={preproc_cfg.get('drnoon_precrop', 0.4)}, circle_mask={preproc_cfg.get('drnoon_circle_mask', True)}")

    lp_cfg = cfg.get("lp", {})
    lp_warmup = lp_cfg.get("lp_warmup_epochs", 5)
    min_val_delta = lp_cfg.get("min_val_loss_delta", 0.001)
    tasks = lp_cfg.get("tasks", ["dr", "amd", "glaucoma"])
    nb_classes = lp_cfg.get("nb_classes", {})

    best_val_loss = float("inf")
    delta = 0.0  # val_loss 개선폭 (LP 트리거 조건). 초기화 필수.
    lp_trigger_count = 0
    improvement_count = 0  # val_loss 갱신 횟수 (lp_trigger_every_k_improvements용)
    max_triggers = lp_cfg.get("lp_max_triggers", 10)
    epochs = cfg.get("training", {}).get("epochs", 100)
    # epochs <= lp_warmup이면 LP가 절대 실행 안 됨 → 마지막 epoch은 항상 LP 블록 진입
    lp_warmup = min(lp_warmup, max(0, epochs - 1))
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    try:
        for epoch in range(epochs):
            train_metrics = train_one_epoch_dinov3_ssl(dinov3_cfg, model, train_paths, epoch, preprocess_fn=preprocess_fn)
            val_loss = run_validation_dinov3_ssl(dinov3_cfg, model, val_paths, epoch, preprocess_fn=preprocess_fn)

            if rank == 0:
                print(f"Epoch {epoch} train_loss={train_metrics.get('loss', 0):.4f} val_loss={val_loss:.4f}")
                if mlflow_active:
                    try:
                        import mlflow
                        mlflow.log_metrics({"train_loss": train_metrics.get("loss", 0), "val_loss": val_loss}, step=epoch)
                    except Exception:
                        pass

            if epoch < lp_warmup:
                continue

            if val_loss < best_val_loss or epoch == epochs - 1:
                # FSDP DTensor → plain Tensor 변환 (LP에서 load 시 mixed Tensor/DTensor 에러 방지)
                state_dict = model.state_dict()
                state_dict_plain = {}
                for k, v in state_dict.items():
                    if hasattr(v, "full_tensor"):
                        state_dict_plain[k] = v.full_tensor()
                    else:
                        state_dict_plain[k] = v

            if val_loss < best_val_loss:
                delta = best_val_loss - val_loss
                best_val_loss = val_loss
                improvement_count += 1
                if rank == 0:
                    ckpt_path = os.path.join(ckpt_dir, "best.pt")
                    torch.save({"model": state_dict_plain, "epoch": epoch, "val_loss": val_loss}, ckpt_path)
                # LP는 SSL 중간에 실행 시 GPU 충돌 → SSL 완료 후에만 8 GPU로 실행

            if epoch == epochs - 1 and rank == 0:
                ckpt_path = os.path.join(ckpt_dir, "last.pt")
                torch.save({"model": state_dict_plain, "epoch": epoch, "val_loss": val_loss}, ckpt_path)

        # SSL 완료 → GPU 해제 후 LP를 8 GPU로 실행 (SSL 중 실행 시 GPU 충돌)
        world_size = distributed.get_world_size() if distributed.is_enabled() else 1
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if rank != 0:
            return  # rank 1~7 종료 (GPU 해제)
        # rank 0: 모델 해제 후 LP 실행
        del model
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        bridge_ckpt = os.path.join(ckpt_dir, "best.pt")
        if os.path.exists(bridge_ckpt):
            num_proc = lp_cfg.get("num_processes")
            if num_proc is None:
                num_proc = world_size
            print(f"[LP] SSL 완료. best.pt로 LP 실행 (num_processes={num_proc})")
            lp_summary, _ = run_lp_retfound(
                run_dir=run_dir,
                trigger_idx=0,
                ckpt_path=bridge_ckpt,
                train_csv=train_csv,
                val_csv=val_csv,
                test_csv=cfg["data"]["test_csv"],
                tasks=tasks,
                image_column=image_col,
                batch_size=lp_cfg.get("batch_size", 24),
                epochs=lp_cfg.get("epochs", 50),
                nb_classes_map=nb_classes,
                retfound_pretrained=lp_cfg.get("retfound_pretrained", "RETFound_dinov2_meh"),
                local_prefix=local_prefix,
                warmup_epochs=lp_cfg.get("warmup_epochs", 10),
                model="Dinov3",
                model_arch=lp_cfg.get("model_arch", "dinov3_vitl16"),
                num_processes=num_proc,
                use_drnoon_preprocess=use_drnoon_preprocess,
                drnoon_precrop=preproc_cfg.get("drnoon_precrop", 0.4),
                drnoon_circle_mask=preproc_cfg.get("drnoon_circle_mask", True),
            )
            if mlflow_active and lp_summary:
                try:
                    import mlflow
                    for task, data in lp_summary.items():
                        if isinstance(data, dict) and "score" in data:
                            mlflow.log_metric(f"lp_{task}_score", data["score"])
                except Exception:
                    pass
    finally:
        if rank == 0 and mlflow_active:
            try:
                import mlflow
                mlflow.end_run()
            except Exception:
                pass
