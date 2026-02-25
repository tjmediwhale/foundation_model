"""
foundation_model engine/val_engine.py
DINOv3 SSL validation loss 계산.
"""
import sys
from pathlib import Path
from functools import partial

import torch
from torch.utils.data import DataLoader, DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DINOV3_PATH = PROJECT_ROOT / "model" / "dinov3"
sys.path.insert(0, str(DINOV3_PATH))


def run_validation_dinov3_ssl(
    cfg,
    model,
    image_paths: list,
    epoch: int,
    preprocess_fn=None,
) -> float:
    """
    DINOv3 SSL validation loss 계산.
    Returns: val_loss (float)
    """
    from utils.dataset import FundusSSLDataset
    from dinov3.data import DataAugmentationDINO, MaskingGenerator, collate_data_and_cast
    import dinov3.distributed as distributed

    rank = distributed.get_rank() if distributed.is_enabled() else 0
    world_size = distributed.get_world_size() if distributed.is_enabled() else 1

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

    def transform(img):
        if preprocess_fn is not None:
            img = preprocess_fn(img)
        return aug(img)

    dataset = FundusSSLDataset(image_paths, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    sampler.set_epoch(epoch)

    data_loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        sampler=sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    # SSLMetaArch는 train(mode)를 오버라이드해서 eval() 호출 시 에러남. validation loss만 계산하므로 eval 생략
    total_loss = 0.0
    n_batches = 0
    global_batch_size = cfg.train.batch_size_per_gpu * world_size

    for data in data_loader:
            data["global_batch_size"] = global_batch_size
            teacher_temp = cfg.teacher.teacher_temp
            loss, _ = model.forward_backward(
                data, teacher_temp=teacher_temp, iteration=epoch * 1000
            )
            total_loss += loss.item()
            n_batches += 1

    if distributed.is_enabled():
        total_loss_t = torch.tensor([total_loss, float(n_batches)], device="cuda")
        torch.distributed.all_reduce(total_loss_t)
        total_loss = total_loss_t[0].item()
        n_batches = int(total_loss_t[1].item())

    return total_loss / max(n_batches, 1)
