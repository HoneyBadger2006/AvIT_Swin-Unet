# Soft-voting ensemble of SwinUnet + EGE-UNet + AViT: averages each model's
# raw TTA-averaged probability map (post-sigmoid, pre-threshold), saved by
# eval_swinunet_reference.py / evaluate.py / eval_avit_reference.py at a
# common 224x224 resolution (results/prob_maps/{model}/{id}.npy), then
# thresholds at 0.5 and applies the same Method A postprocessing used for
# every individual model, so the ensemble is evaluated under the identical
# pipeline -- not a separately-tuned one.
#
# Two variants, both probability-averaging (soft voting), not hard/majority
# voting on binarized masks:
#   A. Equal weight:       p = (p_swin + p_ege + p_avit) / 3
#   B. Dice-weighted:      p = w_swin*p_swin + w_ege*p_ege + w_avit*p_avit
#      weights = each model's own TTA+Method A Dice score (0.8521 / 0.8481 /
#      0.7946), normalized to sum to 1. These are the exact values from the
#      task spec, not re-derived from a different metric.
#
# Pure inference on already-saved probability maps; no training, no new
# forward passes.

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import medpy.metric.binary as medpy_metrics

import sys
AVIT_DIR = Path(r"C:\Users\quanp\Downloads\ISIC 2017\AViT")
sys.path.insert(0, str(AVIT_DIR))
from postprocess_pipeline import morph_open_close, keep_largest_component  # noqa: E402

DATA_ROOT = Path(r"C:\Users\quanp\Downloads\ISIC 2017\data\isic2017")
LABEL_DIR = DATA_ROOT / "Label"
TEST_META = DATA_ROOT / "meta_isic2017_test600.csv"
PROB_ROOT = Path(__file__).resolve().parent / "results" / "prob_maps"
ENSEMBLE_SIZE = 224

INDIVIDUAL_DICE = {"swinunet": 0.8521, "egeunet": 0.8481, "avit": 0.7946}
DICE_WEIGHT_SUM = sum(INDIVIDUAL_DICE.values())
DICE_WEIGHTS = {k: v / DICE_WEIGHT_SUM for k, v in INDIVIDUAL_DICE.items()}


def load_mask(img_id):
    msk = np.load(LABEL_DIR / f"{img_id}.npy")
    msk = (msk > 0.5).astype(np.uint8)
    if msk.ndim == 3:
        msk = msk[..., 0]
    return cv2.resize(msk, (ENSEMBLE_SIZE, ENSEMBLE_SIZE), interpolation=cv2.INTER_NEAREST)


def apply_method_a(pred01):
    m = morph_open_close(pred01, kernel_size=5)
    m = keep_largest_component(m)
    return m


def dice_iou(pred01, gt01):
    if pred01.sum() == 0 and gt01.sum() == 0:
        return 1.0, 1.0
    return medpy_metrics.dc(pred01, gt01), medpy_metrics.jc(pred01, gt01)


def main():
    ids = pd.read_csv(TEST_META, dtype={"ID": str})["ID"].tolist()
    assert len(ids) == 600, f"expected 600 test IDs, got {len(ids)}"

    for name in ("swinunet", "egeunet", "avit"):
        n_files = len(list((PROB_ROOT / name).glob("*.npy")))
        assert n_files == 600, f"{name}: expected 600 probability maps, found {n_files}"

    print(f"Dice-normalized weights: {json.dumps(DICE_WEIGHTS, indent=2)}")

    rows = []
    for img_id in ids:
        p_swin = np.load(PROB_ROOT / "swinunet" / f"{img_id}.npy")
        p_ege = np.load(PROB_ROOT / "egeunet" / f"{img_id}.npy")
        p_avit = np.load(PROB_ROOT / "avit" / f"{img_id}.npy")
        gt = load_mask(img_id)

        p_equal = (p_swin + p_ege + p_avit) / 3.0
        p_weighted = (
            DICE_WEIGHTS["swinunet"] * p_swin
            + DICE_WEIGHTS["egeunet"] * p_ege
            + DICE_WEIGHTS["avit"] * p_avit
        )

        pred_equal = (p_equal > 0.5).astype(np.uint8)
        pred_weighted = (p_weighted > 0.5).astype(np.uint8)
        pred_equal_pp = apply_method_a(pred_equal)
        pred_weighted_pp = apply_method_a(pred_weighted)

        d_eq, i_eq = dice_iou(pred_equal, gt)
        d_eq_pp, i_eq_pp = dice_iou(pred_equal_pp, gt)
        d_w, i_w = dice_iou(pred_weighted, gt)
        d_w_pp, i_w_pp = dice_iou(pred_weighted_pp, gt)

        rows.append({
            "image_id": img_id,
            "dice_equal": d_eq, "iou_equal": i_eq,
            "dice_equal_methodA": d_eq_pp, "iou_equal_methodA": i_eq_pp,
            "dice_weighted": d_w, "iou_weighted": i_w,
            "dice_weighted_methodA": d_w_pp, "iou_weighted_methodA": i_w_pp,
        })

    df = pd.DataFrame(rows)
    out_dir = Path(__file__).resolve().parent / "results"
    df.to_csv(out_dir / "ensemble_per_image.csv", index=False)

    summary = {"n_test": len(ids), "dice_weights": DICE_WEIGHTS}
    for col in ["dice_equal", "iou_equal", "dice_equal_methodA", "iou_equal_methodA",
                "dice_weighted", "iou_weighted", "dice_weighted_methodA", "iou_weighted_methodA"]:
        summary[f"{col}_mean"] = float(df[col].mean())
        summary[f"{col}_std"] = float(df[col].std(ddof=1))

    with open(out_dir / "ensemble_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    best_individual_dice = max(INDIVIDUAL_DICE.values())
    print(json.dumps(summary, indent=2))
    print(f"\nBest individual model (SwinUnet): {best_individual_dice:.4f} Dice")
    print(f"Variant A (equal, +Method A):     {summary['dice_equal_methodA_mean']:.4f} Dice "
          f"({'BEATS' if summary['dice_equal_methodA_mean'] > best_individual_dice else 'DOES NOT BEAT'} SwinUnet alone)")
    print(f"Variant B (weighted, +Method A):  {summary['dice_weighted_methodA_mean']:.4f} Dice "
          f"({'BEATS' if summary['dice_weighted_methodA_mean'] > best_individual_dice else 'DOES NOT BEAT'} SwinUnet alone)")


if __name__ == "__main__":
    main()
