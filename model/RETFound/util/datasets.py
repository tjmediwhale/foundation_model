import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def _get_fundus_preprocess_fn(args):
    """drnoon-image-transform 전처리 (retinal crop). foundation_model utils 사용."""
    if not getattr(args, "use_drnoon_preprocess", False):
        return None
    _fm = Path(__file__).resolve().parent.parent.parent.parent  # util->RETFound->model->foundation_model
    if str(_fm) not in sys.path:
        sys.path.insert(0, str(_fm))
    from utils.preprocessing import get_fundus_preprocess_fn
    return get_fundus_preprocess_fn(
        precrop=getattr(args, "drnoon_precrop", 0.4),
        circle_mask=getattr(args, "drnoon_circle_mask", True),
    )


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, is_train)
    dataset = datasets.ImageFolder(root, transform=transform)

    if is_train == 'train':
        ratio = float(getattr(args, "dataratio", 1.0))
        seed = int(getattr(args, "seed", 0))
        stratified = bool(getattr(args, "stratified", False))

        if 0.0 < ratio < 1.0:
            if stratified:
                idx = _stratified_indices(dataset.targets, ratio, seed)
            else:
                # simple uniform subsample with torch.Generator for reproducibility
                g = torch.Generator().manual_seed(seed)
                n = len(dataset)
                k = max(1, int(n * ratio))
                idx = torch.randperm(n, generator=g)[:k].tolist()
            dataset = Subset(dataset, idx)

    return dataset

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    preprocess_fn = _get_fundus_preprocess_fn(args)

    if is_train == 'train':
        base_transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if preprocess_fn is not None:
            def _compose_train(img):
                img = preprocess_fn(img)
                return base_transform(img)
            return _compose_train
        return base_transform

    # eval transform
    crop_pct = 224 / 256 if args.input_size <= 224 else 1.0
    size = int(args.input_size / crop_pct)
    t = [
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    base = transforms.Compose(t)
    if preprocess_fn is not None:
        def _compose_eval(img):
            img = preprocess_fn(img)
            return base(img)
        return _compose_eval
    return base

# ---- helpers ----

def _stratified_indices(targets, ratio: float, seed: int):
    """Maintain class proportions. Ensures at least 1 sample per class when possible."""
    t = torch.as_tensor(targets)
    classes = torch.unique(t)
    g = torch.Generator().manual_seed(seed)

    keep = []
    for c in classes.tolist():
        cls_idx = torch.nonzero(t == c, as_tuple=False).view(-1)
        if len(cls_idx) == 0:
            continue
        k = max(1, int(round(len(cls_idx) * ratio)))
        sel = cls_idx[torch.randperm(len(cls_idx), generator=g)[:k]]
        keep.extend(sel.tolist())

    # shuffle final indices (stable across seed)
    g2 = torch.Generator().manual_seed(seed + 1)
    keep = torch.tensor(keep)[torch.randperm(len(keep), generator=g2)].tolist()
    return keep

