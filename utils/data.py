"""
foundation_model utils/data.py
CSV 로드, gs:// 경로 치환, LP용 filtered CSV 생성.
"""
import os
import pandas as pd
from typing import Optional, Tuple

_DEFAULT_LOCAL_PREFIX = "/nas/mediwhale_processed_data/"


def replace_gs_path(path: str, local_prefix: Optional[str] = None) -> str:
    """gs:// -> local_prefix 치환."""
    prefix = (local_prefix or _DEFAULT_LOCAL_PREFIX).rstrip("/") + "/"
    if isinstance(path, str) and path.startswith("gs://"):
        return path.replace("gs://", prefix)
    return path


def load_csv_with_path_replace(
    csv_path: str,
    image_column: str = "jpg_h1024_path",
    local_prefix: Optional[str] = None,
) -> pd.DataFrame:
    """CSV 로드 후 image_column 경로 치환."""
    df = pd.read_csv(csv_path)
    if image_column in df.columns:
        p = local_prefix or _DEFAULT_LOCAL_PREFIX
        df[image_column] = df[image_column].astype(str).apply(lambda x: replace_gs_path(x, p))
    return df


def create_filtered_csv_for_lp_task(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    task_column: str,
    output_dir: str,
    image_column: str = "jpg_h1024_path",
    local_prefix: Optional[str] = None,
) -> Tuple[str, str, str]:
    """LP용 태스크별 filtered CSV 생성 (blank/NaN 라벨 제외). Returns (train_path, val_path, test_path)."""
    os.makedirs(output_dir, exist_ok=True)
    out_train = os.path.join(output_dir, f"train_{task_column}.csv")
    out_val = os.path.join(output_dir, f"val_{task_column}.csv")
    out_test = os.path.join(output_dir, f"test_{task_column}.csv")

    for src, dst in [(train_csv, out_train), (val_csv, out_val), (test_csv, out_test)]:
        df = load_csv_with_path_replace(src, image_column, local_prefix)
        if task_column not in df.columns:
            df.to_csv(dst, index=False)
            continue
        mask = df[task_column].notna() & (df[task_column].astype(str).str.strip() != "")
        df_filtered = df[mask].copy()
        df_filtered.to_csv(dst, index=False)

    return out_train, out_val, out_test


def build_imagefolder_from_csv(
    csv_path: str,
    image_column: str,
    label_column: str,
    output_root: str,
    split: str,
    local_prefix: Optional[str] = None,
) -> int:
    """CSV 기반 ImageFolder 구조 생성 (symlink, 실패 시 copy). output_root/{split}/class0/, class1/ ...
    Returns: 생성된 이미지 개수."""
    import shutil

    df = load_csv_with_path_replace(csv_path, image_column, local_prefix)
    if label_column not in df.columns:
        return 0

    split_dir = os.path.join(output_root, split)
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)
    os.makedirs(split_dir, exist_ok=True)

    n_created = 0
    first_path_logged = False
    for _, row in df.iterrows():
        path = row[image_column]
        label = row[label_column]
        if pd.isna(path) or pd.isna(label) or str(label).strip() == "":
            continue
        local_path = replace_gs_path(str(path), local_prefix)
        if not first_path_logged and n_created == 0:
            first_path_logged = True
            print(f"[LP build_imagefolder] {split} 첫 경로: {local_path[:90]}... exists={os.path.exists(local_path)}")
        class_dir = os.path.join(split_dir, str(int(label) if isinstance(label, (int, float)) else label))
        os.makedirs(class_dir, exist_ok=True)
        base = os.path.basename(local_path)
        dst_path = os.path.join(class_dir, base)
        if os.path.exists(dst_path):
            n_created += 1
            continue
        try:
            os.symlink(local_path, dst_path)
            n_created += 1
        except OSError:
            if os.path.exists(local_path):
                try:
                    shutil.copy2(local_path, dst_path)
                    n_created += 1
                except OSError:
                    pass
    return n_created
