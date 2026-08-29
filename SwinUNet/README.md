# SwinUNet — orchestration scripts & hard-image pilot

SwinUnet has no model code of its own in this repo — it trains entirely
through the shared `AViT/multi_train_adapt.py --model SwinUnet` (same
training script AViT itself uses, just a different `--model` value). This
folder holds SwinUnet-specific orchestration scripts and this project's
SwinUnet-specific documentation; there is no `model.py`/`train.py` to
vendor here the way `EGE-UNet/` has its own.

## What's here

| File | Origin |
|---|---|
| `run_swinunet_bceLogits_dice.ps1` | SwinUnet-only 5-fold driver (BCE-with-logits + Dice loss sweep, k0–k4). Moved here unchanged from the repo root — every path inside it is already absolute, so the move required no edits. |

## Scripts still at repo root — not moved, and why

Several other root-level scripts mention SwinUnet but are **not**
SwinUnet-exclusive — they train or monitor SwinUnet *and* AViT together in
the same driver run (e.g. `run_clahe_pilot.ps1` launches SwinUnet k0 then
AViT k0 in one script; `watchdog_clahe_focal.ps1` watches both networks'
5-fold sweeps and relaunches a shared driver). Moving those here would
misrepresent them as SwinUnet-only when AViT's own workflow depends on them
too, and `watchdog_clahe_focal.ps1`/`watchdog_clahe_ftl.ps1` reference their
driver scripts by hardcoded absolute root path — moving the drivers without
updating the watchdogs would silently break the relaunch-on-crash behavior.
Left at root pending a decision on how (or whether) to split joint
SwinUnet+AViT scripts by network.

## Training and evaluating

```sh
cd ../AViT
python -u multi_train_adapt.py --exp_name test --config_yml Configs/multi_train_local.yml --model SwinUnet --batch_size 16 --dataset isic2017 --k_fold 0
```

See `AViT/README.md` for the full set of `--model` options this shared
script supports (SwinUnet is one of several comparison models it can run).

## SwinUnet's own hard-image augmentation pilot — the original pilot this project's method is based on

The first run of this project's hard-image method, on SwinUnet itself: a
clean baseline (2000 official training images, `--k_fold allin`, seed 42,
CLAHE, SwinUnet's own recipe) evaluated against those same 2000 training
images, single pass, no TTA, threshold 0.5.

- **37 hard training images** (Dice<0.7 of 2000, baseline mean 0.9137) →
  `per_image_analysis_v2/bad_image_augmentation/bad_images_seed42_dice37.csv`
- **73 hard test images** (Dice<0.7 of the 600 official test images, on the
  separately-trained SwinUnet/CLAHE/fold0 5-fold-CV checkpoint — the
  established baseline for this network at the time, not the allin
  identification checkpoint) → not a standalone CSV; derived inline from
  `per_image_analysis_v2/final_pipeline/per_image_final_pipeline.csv`
  (network=SwinUnet, stage=clahe, fold=0) by `AViT/shift37_pilot_runner.py`.

Augmented via `AViT/build_shift_augmentation.py` (the same script every
later network's pilot reuses unmodified) — 37×5 = 185 images, all 5
techniques, 0 skips — merged into the canonical `Image_clahe`/`Label` dirs.
Pilot trained on the combined 2185-image set, `allin`, 30 epochs
(`results/isic2017_swinunet_clahe_seed42_dice37_shift5_SwinUnet_20260824_0145/final.pth`).

| | n | Baseline mean Dice | Pilot mean Dice | Still bad (Dice<0.7) |
|---|---|---|---|---|
| Hard training images | 37 | 0.5668 | **0.8560** | 2/37 (was 37/37) |
| Hard test images | 73 | 0.4846 | 0.5361 | 53/73 (was 73/73 by construction) |
| **Full test600** | 600 | **0.8460** | **0.8500** | 74/600 (was 73/600) |

| Paired one-sided t-test | n | mean_diff (pilot − baseline) | p (one-sided) | Gate (diff>0.003 & p<0.05) |
|---|---|---|---|---|
| **Full test600** | 600 | **+0.0040** | **0.153** | **diff clears the bar, p does not — fails: decision: stop** |
| Hard subset (73) | 73 | +0.0515 | 0.0095 | passes significance on its own, but this isn't the gating subset |

**Training-side recovery is strong** — 35 of 37 targeted images recovered
above Dice 0.7 (mean 0.57→0.86). **The hard-test subset shows a real,
statistically significant gain** (+5.15pp, p=0.0095) — unlike EGE-UNet's
own pilot (see `EGE-UNet/README.md`), this one *is* directionally positive
and clears significance on the subset it was aimed at. But the full test600
result is the gating metric, and there it's a narrow miss: the mean_diff
(+0.40pp) clears the +0.3pp threshold, but at p=0.153 it isn't statistically
significant — the full-set failure count actually ticked up by one (73→74)
even as the mean rose, meaning the gain is concentrated on the images it
helped rather than a uniform lift, with enough of a wash elsewhere that
the aggregate effect isn't distinguishable from noise.

**Decision: stop, per the same bar used for every other technique in this
project (`mean_diff > 0.003 and p_one_sided < 0.05` on the full test600).**
No 5-fold sweep committed for SwinUnet on this technique — this is the
reference result AViT's and EGE-UNet's own pilots are compared against in
`EGE-UNet/README.md` ("just short of the significance bar").

Full summary: `per_image_analysis_v2/overnight_run/shift37_pilot_summary.json`.
Per-image CSVs: `per_image_analysis_v2/bad_image_augmentation/shift37_train_recovery.csv`
(training-side), `per_image_analysis_v2/overnight_run/shift37_test600_per_image.csv`
(full test600).
