# Evaluates an EGE-UNet checkpoint's per-image Dice against an arbitrary
# image-ID list (the 2000 official training images, or the 600 official test
# images), single forward pass, no TTA, no Method A postprocessing --
# matching the convention used to originally identify SwinUnet's/AViT's own
# hard-image sets (a quick per-image scan to find Dice<0.7 images, not a
# final eval number). Reused for two purposes in the EGE-UNet hard-image
# pipeline:
#   1. Baseline checkpoint (no augmentation, 2000 images, seed 42) evaluated
#      against its own 2000 training images -> identifies EGE-UNet's own
#      hard TRAINING images (mirrors AViT/eval_train_full_recovery.py, which
#      is Dataset_wrap_csv/build_model-specific and can't be reused directly
#      for EGE-UNet's standalone model/loader).
#   2. That same baseline checkpoint evaluated against the 600 official test
#      images -> identifies EGE-UNet's own hard TEST images, for methodological
#      consistency with how SwinUnet's/AViT's "test73"/"test145" sets were
#      each derived from that network's own pre-augmentation CLAHE baseline,
#      not reused from an already-augmented checkpoint.

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import medpy.metric.binary as medpy_metrics

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import EGEUNet  # noqa: E402
from dataset import load_meta, ISICOfficialDataset  # noqa: E402
from transforms import build_test_transform  # noqa: E402

DATA_ROOT = Path(r"C:\Users\quanp\Downloads\ISIC 2017\data\isic2017")
IMAGE_DIR = DATA_ROOT / "Image_clahe"
LABEL_DIR = DATA_ROOT / "Label"
IMG_SIZE = 256


def dice_iou(pred01, gt01):
    if pred01.sum() == 0 and gt01.sum() == 0:
        return 1.0, 1.0
    return medpy_metrics.dc(pred01, gt01), medpy_metrics.jc(pred01, gt01)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--norm-stats", required=True, help="path to the run's norm_stats.json")
    ap.add_argument("--meta-csv-name", required=True,
                     help="filename under data/isic2017/, e.g. meta_isic2017_train2000.csv")
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    stats = json.loads(Path(args.norm_stats).read_text())
    mean, std = stats["mean"], stats["std"]

    ids = load_meta(DATA_ROOT / args.meta_csv_name)
    print(f"Evaluating {len(ids)} images from {args.meta_csv_name}", flush=True)

    test_tf = build_test_transform(mean, std, size_h=IMG_SIZE, size_w=IMG_SIZE)
    ds = ISICOfficialDataset(ids, IMAGE_DIR, LABEL_DIR, test_tf)

    model = EGEUNet(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64],
                     bridge=True, gt_ds=True).cuda()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cuda", weights_only=True))
    model.eval()

    rows = []
    with torch.no_grad():
        for idx, img_id in enumerate(ids):
            img_t, msk_t = ds[idx]
            img_t = img_t.unsqueeze(0).cuda().float()
            gt = (msk_t[0].numpy() > 0.5).astype(np.uint8)

            _, out = model(img_t)
            pred = (out[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
            dice, iou = dice_iou(pred, gt)
            rows.append({"image_id": img_id, "dice": dice, "iou": iou})

            if idx % 200 == 0:
                print(f"  {idx}/{len(ids)}", flush=True)

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    n_bad = int((df["dice"] < 0.7).sum())
    print(f"n={len(df)}  mean_dice={df['dice'].mean():.4f}  n_bad(dice<0.7)={n_bad}")


if __name__ == "__main__":
    main()
