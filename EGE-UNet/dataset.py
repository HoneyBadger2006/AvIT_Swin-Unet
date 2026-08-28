# Dataset loader for the official-split + hard-image-augmentation training run.
#
# Reads this project's existing precomputed .npy arrays (data/isic2017/Image_clahe/
# for CLAHE-enhanced images, data/isic2017/Label/ for masks) by ID, driven by a meta
# CSV with an ID column -- matching AViT/Datasets/create_dataset.py's SkinDataset_csv
# on-disk format exactly (uint8 HxWx3 RGB images, uint8 HxWx1 masks thresholded
# >0.5), but using EGE-UNet's own transform pipeline (egeunet-style scalar
# normalize + resize-before-augment) instead of Swin/AViT's ImageNet normalization.

from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def load_meta(csv_path):
    """Returns a list of image IDs (str, zero-padded as stored in the CSV/on disk)."""
    df = pd.read_csv(csv_path, dtype={"ID": str})
    return df["ID"].tolist()


def compute_mean_std(ids, image_dir, size=(256, 256)):
    """Scalar mean/std over [0,255] pixel values for the given IDs, matching
    egeunet/dataset.py:compute_mean_std's convention (see EGE-UNET verification
    repo). Images are resized to `size` first for consistency with training
    resolution and to keep this fast."""
    import cv2

    image_dir = Path(image_dir)
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0
    for img_id in ids:
        arr = np.load(image_dir / f"{img_id}.npy").astype(np.float64)
        if arr.shape[:2] != size:
            arr = cv2.resize(arr, size, interpolation=cv2.INTER_LINEAR)
        total_sum += arr.sum()
        total_sq_sum += (arr ** 2).sum()
        total_count += arr.size
    mean = total_sum / total_count
    var = total_sq_sum / total_count - mean ** 2
    return float(mean), float(np.sqrt(max(var, 0.0)))


class ISICOfficialDataset(Dataset):
    """Dataset over a fixed list of image IDs, loading pre-existing .npy arrays
    (already CLAHE-enhanced for images, already including the 185 hard-image
    augmentation variants when the training ID list includes them -- both are
    physically materialized in Image_clahe/ and Label/ by this project's
    existing AViT/build_shift_augmentation.py, not regenerated here)."""

    def __init__(self, ids, image_dir, label_dir, transform):
        self.ids = ids
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img = np.load(self.image_dir / f"{img_id}.npy").astype(np.uint8)
        msk = np.load(self.label_dir / f"{img_id}.npy")
        msk = (msk > 0.5).astype(np.float64)
        if msk.ndim == 2:
            msk = np.expand_dims(msk, axis=2)
        img, msk = self.transform((img, msk))
        return img, msk
