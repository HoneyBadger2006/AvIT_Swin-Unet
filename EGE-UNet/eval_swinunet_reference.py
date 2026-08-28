# Like-for-like comparison number for the existing SwinUnet reference checkpoint
# (results/isic2017_swinunet_clahe_seed42_dice37_shift5_SwinUnet_20260824_0145/
# final.pth), computed under the SAME protocol EGE-UNet will be evaluated with:
# official split + 185 hard-image augmentation + CLAHE + 5-view TTA + Method A
# postprocessing. This checkpoint's own test_results.csv (0.8392, batch-pooled
# no-TTA) and per_image_analysis_v2's shift37_pilot_summary.json (0.8500,
# per-image-mean no-TTA) both predate TTA/postprocessing, so neither is a fair
# comparison point for the ensemble's actual eval protocol -- this script fills
# that gap. Pure inference on an already-trained checkpoint; no training here.

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import medpy.metric.binary as medpy_metrics

AVIT_DIR = Path(r"C:\Users\quanp\Downloads\ISIC 2017\AViT")
sys.path.insert(0, str(AVIT_DIR))

from Models.Transformer.SwinUnet import SwinUnet  # noqa: E402
from Utils.tta import tta_forward  # noqa: E402
from postprocess_pipeline import morph_open_close, keep_largest_component  # noqa: E402

DATA_ROOT = Path(r"C:\Users\quanp\Downloads\ISIC 2017\data\isic2017")
IMAGE_DIR = DATA_ROOT / "Image_clahe"
LABEL_DIR = DATA_ROOT / "Label"
TEST_META = DATA_ROOT / "meta_isic2017_test600.csv"
CKPT = Path(
    r"C:\Users\quanp\Downloads\ISIC 2017\results"
    r"\isic2017_swinunet_clahe_seed42_dice37_shift5_SwinUnet_20260824_0145\final.pth"
)
IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PROB_DIR = Path(__file__).resolve().parent / "results" / "prob_maps" / "swinunet"


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
    """prob01_thresholded: uint8 HxW mask, already thresholded at 0.5."""
    m = morph_open_close(prob01_thresholded, kernel_size=5)
    m = keep_largest_component(m)
    return m


def dice_iou(pred01, gt01):
    if pred01.sum() == 0 and gt01.sum() == 0:
        return 1.0, 1.0
    dice = medpy_metrics.dc(pred01, gt01)
    iou = medpy_metrics.jc(pred01, gt01)
    return dice, iou


def main():
    ids = pd.read_csv(TEST_META, dtype={"ID": str})["ID"].tolist()
    assert len(ids) == 600, f"expected 600 test IDs, got {len(ids)}"

    model = SwinUnet(img_size=IMG_SIZE).cuda().eval()
    model.load_state_dict(torch.load(CKPT, map_location="cuda", weights_only=True))

    def forward_fn(x):
        return model(x)

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
    df.to_csv(out_dir / "swinunet_reference_per_image.csv", index=False)

    summary = {
        "checkpoint": str(CKPT),
        "n_test": len(ids),
        "protocol": "official split + 185 hard-image aug + CLAHE, per-image mean Dice/IoU",
    }
    for col in ["dice_notta", "iou_notta", "dice_notta_methodA", "iou_notta_methodA",
                "dice_tta", "iou_tta", "dice_tta_methodA", "iou_tta_methodA"]:
        summary[f"{col}_mean"] = float(df[col].mean())
        summary[f"{col}_std"] = float(df[col].std(ddof=1))

    with open(out_dir / "swinunet_reference_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
