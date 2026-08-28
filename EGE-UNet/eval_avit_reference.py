# Evaluates the AViT checkpoint trained on the official split + 185 hard-image
# augmentation + CLAHE (results/isic2017_avit_clahe_seed42_dice37_shift5_
# SwinSeg_CNNprompt_adapt_<timestamp>/final.pth) under the SAME protocol used
# for the SwinUnet and EGE-UNet numbers: 5-view TTA + Method A postprocessing,
# 600 official test images, per-image mean Dice/IoU as primary metric.
#
# Reuses AViT/tta_inference.py's build_model/forward_logits (so model
# construction -- Swin-Large backbone + ResNet34 CNN prompt + adapters --
# exactly matches how this checkpoint was trained and how every other AViT
# number in this project was produced) and AViT/postprocess_pipeline.py's
# Method A functions, both unmodified. Pure inference; no training here.

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
import medpy.metric.binary as medpy_metrics

AVIT_DIR = Path(r"C:\Users\quanp\Downloads\ISIC 2017\AViT")
sys.path.insert(0, str(AVIT_DIR))

from Utils.pieces import DotDict  # noqa: E402
from Utils.tta import tta_forward  # noqa: E402
from tta_inference import build_model, forward_logits  # noqa: E402
from postprocess_pipeline import morph_open_close, keep_largest_component  # noqa: E402

DATA_ROOT = Path(r"C:\Users\quanp\Downloads\ISIC 2017\data\isic2017")
IMAGE_DIR = DATA_ROOT / "Image_clahe"
LABEL_DIR = DATA_ROOT / "Label"
TEST_META = DATA_ROOT / "meta_isic2017_test600.csv"
MODEL_NAME = "SwinSeg_CNNprompt_adapt"
IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PROB_DIR = Path(__file__).resolve().parent / "results" / "prob_maps" / "avit"


def find_checkpoint(results_dir, prefix):
    candidates = sorted(p for p in results_dir.iterdir()
                         if p.is_dir() and p.name.startswith(prefix) and (p / "final.pth").exists())
    if not candidates:
        raise FileNotFoundError(f"no completed run found with prefix {prefix!r} under {results_dir}")
    return candidates[-1] / "final.pth"


def load_image_tensor(img_id):
    img = np.load(IMAGE_DIR / f"{img_id}.npy").astype(np.uint8)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img).permute(2, 0, 1).float()


def load_mask(img_id):
    msk = np.load(LABEL_DIR / f"{img_id}.npy")
    msk = (msk > 0.5).astype(np.uint8)
    if msk.ndim == 3:
        msk = msk[..., 0]
    return cv2.resize(msk, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)


def apply_method_a(prob01_thresholded):
    m = morph_open_close(prob01_thresholded, kernel_size=5)
    m = keep_largest_component(m)
    return m


def dice_iou(pred01, gt01):
    if pred01.sum() == 0 and gt01.sum() == 0:
        return 1.0, 1.0
    return medpy_metrics.dc(pred01, gt01), medpy_metrics.jc(pred01, gt01)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                     help="defaults to the newest isic2017_avit_clahe_seed42_dice37_shift5_* run's final.pth")
    ap.add_argument("--config_yml", default=str(AVIT_DIR / "Configs" / "multi_train_local.yml"))
    args = ap.parse_args()

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
    else:
        results_dir = Path(r"C:\Users\quanp\Downloads\ISIC 2017\results")
        ckpt = find_checkpoint(results_dir, f"isic2017_avit_clahe_seed42_dice37_shift5_{MODEL_NAME}_")

    ids = pd.read_csv(TEST_META, dtype={"ID": str})["ID"].tolist()
    assert len(ids) == 600, f"expected 600 test IDs, got {len(ids)}"

    config = DotDict(yaml.load(open(args.config_yml), Loader=yaml.FullLoader))
    model = build_model(MODEL_NAME, config)
    model.load_state_dict(torch.load(ckpt, map_location="cuda", weights_only=True))
    model.eval()

    def forward_fn(x):
        return forward_logits(model, MODEL_NAME, x)

    PROB_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.no_grad():
        for img_id in ids:
            img_t = load_image_tensor(img_id).unsqueeze(0).cuda()
            gt = load_mask(img_id)

            avg_prob, orig_prob = tta_forward(forward_fn, img_t)
            avg_prob_np = avg_prob[0, 0].cpu().numpy().astype(np.float32)
            np.save(PROB_DIR / f"{img_id}.npy", avg_prob_np)

            avg_pred = (avg_prob_np > 0.5).astype(np.uint8)
            notta_pred = (orig_prob[0, 0].cpu().numpy() > 0.5).astype(np.uint8)

            avg_pred_pp = apply_method_a(avg_pred)
            notta_pred_pp = apply_method_a(notta_pred)

            d_tta, i_tta = dice_iou(avg_pred, gt)
            d_tta_pp, i_tta_pp = dice_iou(avg_pred_pp, gt)
            d_notta, i_notta = dice_iou(notta_pred, gt)
            d_notta_pp, i_notta_pp = dice_iou(notta_pred_pp, gt)

            rows.append({
                "image_id": img_id,
                "dice_notta": d_notta, "iou_notta": i_notta,
                "dice_notta_methodA": d_notta_pp, "iou_notta_methodA": i_notta_pp,
                "dice_tta": d_tta, "iou_tta": i_tta,
                "dice_tta_methodA": d_tta_pp, "iou_tta_methodA": i_tta_pp,
            })

    df = pd.DataFrame(rows)
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "avit_reference_per_image.csv", index=False)

    summary = {
        "checkpoint": str(ckpt),
        "n_test": len(ids),
        "protocol": "official split + 185 hard-image aug + CLAHE + FTL, per-image mean Dice/IoU",
    }
    for col in ["dice_notta", "iou_notta", "dice_notta_methodA", "iou_notta_methodA",
                "dice_tta", "iou_tta", "dice_tta_methodA", "iou_tta_methodA"]:
        summary[f"{col}_mean"] = float(df[col].mean())
        summary[f"{col}_std"] = float(df[col].std(ddof=1))

    with open(out_dir / "avit_reference_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
