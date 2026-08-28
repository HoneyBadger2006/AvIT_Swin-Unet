# Follow-up to ensemble.py: (1) a 2-network SwinUnet+EGE-UNet ensemble (no
# AViT), and (2) a sweep of AViT's weight from 0% to 32% to trace out whether
# ANY nonzero AViT weight beats SwinUnet alone, or whether AViT never helps.
# Reuses the exact probability maps saved by eval_swinunet_reference.py /
# evaluate.py / eval_avit_reference.py -- no new inference, no retraining.

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
SWINUNET_ALONE = 0.8521

SWEEP_AVIT_WEIGHTS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.32]


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


def load_all_probs_and_gt(ids):
    """Loads every image's three probability maps + gt once, since both the
    2-network ensemble and the 7-point sweep reuse the same underlying data
    with only the combination weights changing."""
    data = {}
    for img_id in ids:
        data[img_id] = {
            "swinunet": np.load(PROB_ROOT / "swinunet" / f"{img_id}.npy"),
            "egeunet": np.load(PROB_ROOT / "egeunet" / f"{img_id}.npy"),
            "avit": np.load(PROB_ROOT / "avit" / f"{img_id}.npy"),
            "gt": load_mask(img_id),
        }
    return data


def evaluate_combo(data, ids, w_swin, w_ege, w_avit):
    """w_swin + w_ege + w_avit should sum to ~1. Returns a dict of per-image
    Dice/IoU lists (post-threshold, post-Method-A) plus mean/std summary."""
    dices, ious = [], []
    for img_id in ids:
        d = data[img_id]
        p = w_swin * d["swinunet"] + w_ege * d["egeunet"] + w_avit * d["avit"]
        pred = (p > 0.5).astype(np.uint8)
        pred_pp = apply_method_a(pred)
        dice, iou = dice_iou(pred_pp, d["gt"])
        dices.append(dice)
        ious.append(iou)
    return {
        "dice_mean": float(np.mean(dices)), "dice_std": float(np.std(dices, ddof=1)),
        "iou_mean": float(np.mean(ious)), "iou_std": float(np.std(ious, ddof=1)),
    }


def main():
    ids = pd.read_csv(TEST_META, dtype={"ID": str})["ID"].tolist()
    assert len(ids) == 600, f"expected 600 test IDs, got {len(ids)}"

    for name in ("swinunet", "egeunet", "avit"):
        n_files = len(list((PROB_ROOT / name).glob("*.npy")))
        assert n_files == 600, f"{name}: expected 600 probability maps, found {n_files}"

    print("Loading all probability maps + ground truth (once, reused across all combos)...")
    data = load_all_probs_and_gt(ids)

    # --- Step 1: 2-network ensemble (SwinUnet + EGE-UNet, no AViT) ---
    two_net_equal = evaluate_combo(data, ids, 0.5, 0.5, 0.0)
    w_swin_2n = INDIVIDUAL_DICE["swinunet"] / (INDIVIDUAL_DICE["swinunet"] + INDIVIDUAL_DICE["egeunet"])
    w_ege_2n = INDIVIDUAL_DICE["egeunet"] / (INDIVIDUAL_DICE["swinunet"] + INDIVIDUAL_DICE["egeunet"])
    two_net_weighted = evaluate_combo(data, ids, w_swin_2n, w_ege_2n, 0.0)

    print("\n=== Step 1: 2-network ensemble (SwinUnet + EGE-UNet, no AViT) ===")
    print(f"Equal-weight    (0.500/0.500): Dice={two_net_equal['dice_mean']:.4f} "
          f"IoU={two_net_equal['iou_mean']:.4f} std={two_net_equal['dice_std']:.4f}")
    print(f"Dice-weighted   ({w_swin_2n:.3f}/{w_ege_2n:.3f}): Dice={two_net_weighted['dice_mean']:.4f} "
          f"IoU={two_net_weighted['iou_mean']:.4f} std={two_net_weighted['dice_std']:.4f}")
    print(f"vs. SwinUnet alone: {SWINUNET_ALONE:.4f}")

    # --- Step 2: AViT weight sensitivity sweep, 0% to 32% ---
    # At each AViT weight w_avit, the remainder (1 - w_avit) is split between
    # SwinUnet and EGE-UNet in their own relative Dice-weighted proportion
    # (0.8521:0.8481), not 50/50 -- so at w_avit=0 this sweep's endpoint
    # exactly reproduces the Step-1 Dice-weighted 2-network result, and at
    # w_avit=0.32 it exactly reproduces ensemble.py's Variant B.
    print("\n=== Step 2: AViT weight sensitivity sweep ===")
    sweep_rows = []
    for w_avit in SWEEP_AVIT_WEIGHTS:
        remainder = 1.0 - w_avit
        w_swin = remainder * w_swin_2n
        w_ege = remainder * w_ege_2n
        result = evaluate_combo(data, ids, w_swin, w_ege, w_avit)
        beats = result["dice_mean"] > SWINUNET_ALONE
        sweep_rows.append({
            "avit_weight": w_avit, "swinunet_weight": w_swin, "egeunet_weight": w_ege,
            "dice_mean": result["dice_mean"], "dice_std": result["dice_std"],
            "iou_mean": result["iou_mean"], "iou_std": result["iou_std"],
            "beats_swinunet_alone": beats,
        })
        print(f"avit_w={w_avit:.2f} (swin={w_swin:.3f} ege={w_ege:.3f}): "
              f"Dice={result['dice_mean']:.4f} IoU={result['iou_mean']:.4f} "
              f"std={result['dice_std']:.4f} {'BEATS' if beats else 'below'} SwinUnet alone")

    sweep_df = pd.DataFrame(sweep_rows)
    is_monotonic_decreasing = all(
        sweep_df["dice_mean"].iloc[i] >= sweep_df["dice_mean"].iloc[i + 1]
        for i in range(len(sweep_df) - 1)
    )
    best_row = sweep_df.loc[sweep_df["dice_mean"].idxmax()]

    out_dir = Path(__file__).resolve().parent / "results"
    sweep_df.to_csv(out_dir / "avit_weight_sweep.csv", index=False)

    summary = {
        "two_network_equal": two_net_equal,
        "two_network_weighted": {**two_net_weighted, "w_swinunet": w_swin_2n, "w_egeunet": w_ege_2n},
        "swinunet_alone_dice": SWINUNET_ALONE,
        "two_network_beats_swinunet_alone": two_net_weighted["dice_mean"] > SWINUNET_ALONE,
        "avit_sweep_monotonically_decreasing": bool(is_monotonic_decreasing),
        "avit_sweep_best_weight": float(best_row["avit_weight"]),
        "avit_sweep_best_dice": float(best_row["dice_mean"]),
        "any_avit_weight_beats_swinunet_alone": bool(sweep_df["beats_swinunet_alone"].any()),
    }
    with open(out_dir / "avit_weight_sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nMonotonically decreasing in AViT weight: {is_monotonic_decreasing}")
    print(f"Best AViT weight in sweep: {best_row['avit_weight']:.2f} (Dice={best_row['dice_mean']:.4f})")
    print(f"Any AViT weight beats SwinUnet alone: {bool(sweep_df['beats_swinunet_alone'].any())}")


if __name__ == "__main__":
    main()
