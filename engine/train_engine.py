"""
foundation_model engine/train_engine.py
accelerate 기반 DINOv3 SSL 학습 1 epoch 수행.
"""
import os
import sys
import math
from pathlib import Path
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# dinov3 경로
DINOV3_PATH = PROJECT_ROOT / "model" / "dinov3"
sys.path.insert(0, str(DINOV3_PATH))


def _build_fundus_dataloader(
    image_paths: list,
    aug_fn,
    batch_size: int,
    num_workers: int,
    collate_fn,
    rank: int,
    world_size: int,
    epoch: int,
) -> DataLoader:
    """Fundus 이미지 경로로 DINOv3 SSL용 DataLoader 생성."""
    from utils.dataset import FundusSSLDataset

    def transform(img):
        return aug_fn(img)

    dataset = FundusSSLDataset(image_paths, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    sampler.set_epoch(epoch)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )


def train_one_epoch_dinov3_ssl(
    cfg,
    model,
    image_paths: list,
    epoch: int,
    accelerator=None,
) -> dict:
    """
    DINOv3 SSL 1 epoch 학습.
    cfg: dinov3 config (OmegaConf)
    model: SSLMetaArch
    image_paths: 유효 이미지 경로 리스트 (pre-scan 완료)
    """
    import dinov3.distributed as distributed
    from dinov3.data import DataAugmentationDINO, MaskingGenerator, collate_data_and_cast
    from dinov3.train.train import (
        build_optimizer,
        build_schedulers,
        apply_optim_scheduler,
        MetricLogger,
    )

    rank = distributed.get_rank() if distributed.is_enabled() else 0
    world_size = distributed.get_world_size() if distributed.is_enabled() else 1

    # Augmentation
    aug = DataAugmentationDINO(
        global_crops_scale=tuple(cfg.crops.global_crops_scale),
        local_crops_scale=tuple(cfg.crops.local_crops_scale),
        local_crops_number=cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
        gram_teacher_crops_size=cfg.crops.gram_teacher_crops_size,
        gram_teacher_no_distortions=cfg.crops.gram_teacher_no_distortions,
        teacher_no_color_jitter=cfg.crops.teacher_to_student_resolution_scale != 1.0,
        local_crops_subset_of_global_crops=cfg.crops.localcrops_subset_of_globalcrops,
        patch_size=cfg.student.patch_size,
        share_color_jitter=cfg.crops.share_color_jitter,
        horizontal_flips=cfg.crops.horizontal_flips,
        mean=cfg.crops.rgb_mean,
        std=cfg.crops.rgb_std,
    )

    img_size = cfg.crops.global_crops_size
    patch_size = int(cfg.student.patch_size * cfg.crops.teacher_to_student_resolution_scale)
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=tuple(cfg.ibot.mask_ratio_min_max),
        mask_probability=cfg.ibot.mask_sample_probability,
        dtype={"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
            cfg.compute_precision.param_dtype
        ],
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        random_circular_shift=cfg.ibot.mask_random_circular_shift,
    )

    data_loader = _build_fundus_dataloader(
        image_paths=image_paths,
        aug_fn=aug,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_fn,
        rank=rank,
        world_size=world_size,
        epoch=epoch,
    )

    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)

    model.train()
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter_this_epoch = min(len(data_loader), OFFICIAL_EPOCH_LENGTH)
    global_batch_size = cfg.train.batch_size_per_gpu * world_size

    metric_logger = MetricLogger(delimiter="  ")
    student = model.student

    for batch_idx, data in enumerate(data_loader):
        if batch_idx >= max_iter_this_epoch:
            break
        it = epoch * OFFICIAL_EPOCH_LENGTH + batch_idx

        data["global_batch_size"] = global_batch_size

        lr = lr_schedule[it]
        wd = wd_schedule[it]
        mom = momentum_schedule[it]
        teacher_temp = teacher_temp_schedule[it]
        last_layer_lr = last_layer_lr_schedule[it]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        optimizer.zero_grad(set_to_none=True)
        total_loss, metrics_dict = model.forward_backward(
            data, teacher_temp=teacher_temp, iteration=it
        )

        if cfg.optim.clip_grad:
            for k, v in student.items():
                torch.nn.utils.clip_grad_norm_(v.parameters(), max_norm=cfg.optim.clip_grad)

        optimizer.step()
        model.update_ema(mom)

        metric_logger.update(loss=total_loss.item())
        metric_logger.update(lr=lr)

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
