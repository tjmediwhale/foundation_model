"""
foundation_model utils/dataset.py
Fundus SSL용 Dataset. DINOv3 augment 적용, 경로는 data.py에서 치환된 상태로 전달.
DataLoader worker에서 import 되므로 .data 의존 제거 (worker sys.path 이슈 회피).
"""
from typing import Callable, List, Optional

from PIL import Image


def _to_local_path(path: str) -> str:
    """gs:// 있으면 /nas/mediwhale_processed_data/ 로 치환. worker 전용 (data.py import 없음)."""
    if path.startswith("gs://"):
        return path.replace("gs://", "/nas/mediwhale_processed_data/")
    return path


class FundusSSLDataset:
    """Fundus 이미지 경로 리스트로 DINOv3 SSL용 Dataset. transform은 DataAugmentationDINO 등."""

    def __init__(
        self,
        image_paths: List[str],
        transform: Optional[Callable] = None,
    ):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = str(self.image_paths[idx])
        local_path = _to_local_path(path)
        img = Image.open(local_path).convert("RGB")
        if self.transform is not None:
            return self.transform(img)
        return img
