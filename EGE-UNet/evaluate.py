# Evaluates a trained EGE-UNet checkpoint on the 600 official ISIC 2017 test
# images, with 5-view TTA and Method A postprocessing (morph open/close +
# largest-connected-component), reusing this project's existing AViT/Utils/tta.py
# and AViT/postprocess_pipeline.py modules unmodified -- same protocol used for
# Swin-UNet/AViT, and for the like-for-like SwinUnet reference number computed
# by eval_swinunet_reference.py, so all three networks' numbers are directly
# comparable.
#
# Reports BOTH per-image mean Dice/IoU (macro-average, matches this project's
# per_image_analysis_v2 convention and the official ISIC challenge metric) and
# pooled/global Dice/IoU (matches multi_train_adapt.py's own batch-pooled
# convention) -- see EGE-UNet training log discussion of the two conventions
# used elsewhere in this project for why both are reported rather than one.
#
# EGE-UNet's forward() returns already-sigmoid-activated probabilities (see
# model.py, official architecture behavior), but AViT/Utils/tta.py's
# tta_forward expects pre-sigmoid logits (it applies its own torch.sigmoid per
# view). We invert via a clamped logit() before handing off to tta_forward, so
# the exact same shared TTA code path is used for all three networks rather
# than a reimplementation.

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import medpy.metric.binary as medpy_metrics

sys.path.insert(0, str(Path(__file__).resolve().parent))
AVIT_DIR = Path(r"C:\Users\quanp\Downloads\ISIC 2017\AViT")
sys.path.insert(0, str(AVIT_DIR))

from model import EGEUNet  # noqa: E402
from transforms import build_test_transform  # noqa: E402
from Utils.tta import tta_forward  # noqa: E402
from postprocess_pipeline import morph_open_close, keep_largest_component  # noqa: E402

DATA_ROOT = Path(r"C:\Users\quanp\Downloads\ISIC 2017\data\isic2017")
IMAGE_DIR = DATA_ROOT / "Image_clahe"
LABEL_DIR = DATA_ROOT / "Label"
TEST_META = DATA_ROOT / "meta_isic2017_test600.csv"
IMG_SIZE = 256
# Ensemble probability maps are saved at 224x224 -- SwinUnet/AViT's native
# resolution -- rather than EGE-UNet's own 256x256, since 2 of 3 models are
# already at 224 and downsampling one map is preferable to upsampling two.
ENSEMBLE_SIZE = 224
PROB_DIR = Path(__file__).resolve().parent / "results" / "prob_maps" / "egeunet"


def load_image_and_mask(img_id, test_tf):
    """Applies the exact same test_tf used during training (Normalize on the
    raw-resolution image -> ToTensor -> Resize to 256x256) so eval-time
    preprocessing matches train-time val-loss evaluation exactly."""
    img = np.load(IMAGE_DIR / f"{img_id}.npy").astype(np.uint8)
    msk = np.load(LABEL_DIR / f"{img_id}.npy")
    msk = (msk > 0.5).astype(np.float64)
    if msk.ndim == 2:
        msk = np.expand_dims(msk, axis=2)
    img_t, msk_t = test_tf((img, msk))
    gt = (msk_t[0].numpy() > 0.5).astype(np.uint8)
    return img_t.float(), gt


def apply_method_a(pred01):
    m = morph_open_close(pred01, kernel_size=5)
    m = keep_largest_component(m)
    return m


def dice_iou(pred01, gt01):
    if pred01.sum() == 0 and gt01.sum() == 0:
        return 1.0, 1.0
    return medpy_metrics.dc(pred01, gt01), medpy_metrics.jc(pred01, gt01)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                     help="defaults to results/official_seed<seed>/checkpoints/best.pth")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    run_dir = root / "results" / f"official_seed{args.seed}"
    ckpt = Path(args.checkpoint) if args.checkpoint else run_dir / "checkpoints" / "best.pth"
    out_dir = Path(args.out_dir) if args.out_dir else run_dir

    stats = json.loads((run_dir / "norm_stats.json").read_text())
    mean, std = stats["mean"], stats["std"]

    ids = pd.read_csv(TEST_META, dtype={"ID": str})["ID"].tolist()
    assert len(ids) == 600, f"expected 600 test IDs, got {len(ids)}"

    test_tf = build_test_transform(mean, std, size_h=IMG_SIZE, size_w=IMG_SIZE)

    model = EGEUNet(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64],
                     bridge=True, gt_ds=True).cuda()
    model.load_state_dict(torch.load(ckpt, map_location="cuda", weights_only=True))
    model.eval()

    def forward_fn(x):
        # EGEUNet.forward (gt_ds=True) returns (gt_pre_tuple, out); out is
        # already post-sigmoid. Invert to logits so tta_forward's own
        # torch.sigmoid per view is mathematically a no-op round-trip.
        _, out = model(x)
        prob = out.clamp(1e-6, 1 - 1e-6)
        return torch.logit(prob)

    PROB_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    global_tp = global_fp = global_fn = 0
    with torch.no_grad():
        for img_id in ids:
            img_t, gt = load_image_and_mask(img_id, test_tf)
            img_t = img_t.unsqueeze(0).cuda()

            avg_prob, orig_prob = tta_forward(forward_fn, img_t)
            avg_prob_np = avg_prob[0, 0].cpu().numpy().astype(np.float32)
            avg_prob_224 = cv2.resize(avg_prob_np, (ENSEMBLE_SIZE, ENSEMBLE_SIZE), interpolation=cv2.INTER_LINEAR)
            np.save(PROB_DIR / f"{img_id}.npy", avg_prob_224)

            tta_pred = (avg_prob_np > 0.5).astype(np.uint8)
            notta_pred = (orig_prob[0, 0].cpu().numpy() > 0.5).astype(np.uint8)

            tta_pred_pp = apply_method_a(tta_pred)
            notta_pred_pp = apply_method_a(notta_pred)

            d_tta, i_tta = dice_iou(tta_pred, gt)
            d_tta_pp, i_tta_pp = dice_iou(tta_pred_pp, gt)
            d_notta, i_notta = dice_iou(notta_pred, gt)
            d_notta_pp, i_notta_pp = dice_iou(notta_pred_pp, gt)

            rows.append({
                "image_id": img_id,
                "dice_notta": d_notta, "iou_notta": i_notta,
                "dice_notta_methodA": d_notta_pp, "iou_notta_methodA": i_notta_pp,
                "dice_tta": d_tta, "iou_tta": i_tta,
                "dice_tta_methodA": d_tta_pp, "iou_tta_methodA": i_tta_pp,
            })

            # pooled/global confusion-matrix accumulation, using the final
            # (TTA + Method A) prediction -- this project's other pooled-dice
            # convention, computed here for cross-reference alongside the
            # per-image mean above.
            global_tp += int(((tta_pred_pp == 1) & (gt == 1)).sum())
            global_fp += int(((tta_pred_pp == 1) & (gt == 0)).sum())
            global_fn += int(((tta_pred_pp == 0) & (gt == 1)).sum())

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "test_per_image.csv", index=False)

    summary = {
        "checkpoint": str(ckpt),
        "n_test": len(ids),
        "protocol": "official split + 185 hard-image aug + CLAHE",
    }
    for col in ["dice_notta", "iou_notta", "dice_notta_methodA", "iou_notta_methodA",
                "dice_tta", "iou_tta", "dice_tta_methodA", "iou_tta_methodA"]:
        summary[f"{col}_mean_per_image"] = float(df[col].mean())
        summary[f"{col}_std_per_image"] = float(df[col].std(ddof=1))

    denom_dice = 2 * global_tp + global_fp + global_fn
    denom_iou = global_tp + global_fp + global_fn
    summary["dice_tta_methodA_pooled_global"] = (2 * global_tp / denom_dice) if denom_dice else 0.0
    summary["iou_tta_methodA_pooled_global"] = (global_tp / denom_iou) if denom_iou else 0.0

    with open(out_dir / "test_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
