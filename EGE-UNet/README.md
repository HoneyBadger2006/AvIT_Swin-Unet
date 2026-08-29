# EGE-UNet — Ensemble Production Training

Trains a production EGE-UNet checkpoint on the **official** ISIC 2017 split
(2000 train / 600 test — the same split Swin-UNet and AViT use), plus the
project's already-validated 37-hard-image × 5-technique augmentation set, for
direct use in the Swin-UNet / AViT / EGE-UNet ensemble.

**This is a separate effort from the EGE-UNET paper-protocol reproduction**
at `C:\Users\quanp\EGE-UNET` (pooled dataset, 5 reseeded 70/30 splits, verifying
the official implementation against the paper's own claimed numbers). That
work is complete and signed off; this folder does not modify it, only reuses
its model/loss/transform code (vendored below, unmodified except headers).
Keep the two efforts' results separate — they answer different questions
(paper-protocol correctness vs. ensemble-ready production performance) on
different data (pooled random splits vs. this project's official split).

## What's vendored vs. new here

| File | Origin |
|---|---|
| `model.py`, `losses.py`, `transforms.py`, `LICENSE` | Copied unmodified from `C:\Users\quanp\EGE-UNET\egeunet\` (Apache-2.0, official EGE-UNet repo architecture) |
| `dataset.py`, `train.py`, `evaluate.py`, `build_train_meta.py`, `eval_swinunet_reference.py` | New, written for this task |

## Data

- **Official split**: `data/isic2017/meta_isic2017_train2000.csv` (2000 IDs) /
  `meta_isic2017_test600.csv` (600 IDs), built by `AViT/build_official_split.py`
  from the raw ISIC-2017 challenge folder membership. Images/masks live as
  `{ID}.npy` (uint8 HxWx3 RGB / HxW mask) in `data/isic2017/Image_clahe/` and
  `Label/` — **not** pre-resized; each loader resizes at read time (Swin/AViT
  to 224, EGE-UNet to 256 per its own paper-native resolution).
- **CLAHE**: `AViT/clahe.py::clahe_enhance(clip_limit=2.0, tile_grid_size=(8,8))`
  on the L channel of LAB, images only. Precomputed once into `Image_clahe/`
  by `AViT/precompute_clahe.py`. EGE-UNet uses the same CLAHE'd images as
  Swin-UNet/AViT for ensemble consistency (confirmed with the user — see
  task's "Open Items", item 1).
- **Hard-image augmentation (validated, reused unmodified)**:
  - 37 image IDs: `per_image_analysis_v2/bad_image_augmentation/bad_images_seed42_dice37.csv`
    (Dice < 0.7 at baseline, seed 42).
  - 185-row manifest (37 × 5 techniques — hflip, vflip, rot90, rot270,
    shiftR10blur): `per_image_analysis_v2/bad_image_augmentation/manifest_seed42_dice37_shift5.csv`.
  - The 185 augmented images/masks are **already generated** on disk (built by
    `AViT/build_shift_augmentation.py`, merged into `Image_clahe/`/`Label/`) —
    nothing here regenerates them, only consumes them by ID.
  - `build_train_meta.py` rebuilds `meta_isic2017_train2000_seed42_dice37_shift5.csv`
    (2000 base + 185 augmented = 2185 rows), which existed for the reference
    SwinUnet run but wasn't preserved on disk. This is pure concatenation of
    two already-validated CSVs — not a re-derivation of the augmentation
    itself — verified against the on-disk `.npy` files before writing.

## Model & training recipe

EGE-UNet's own recipe, unchanged from the paper-protocol reproduction:
AdamW (lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-2),
CosineAnnealingLR (T_max=50, eta_min=1e-5), BCE+Dice deep supervision
(λ=[1,.5,.4,.3,.2,.1]), 300 epochs, batch size 8, 256×256, resize-before-augment
ordering (the reproduction's diagnosed-and-fixed slowdown, still valid here —
see the verification repo's README divergence #7).

**Single run, seed 42** (confirmed with the user — this is a production
checkpoint for the ensemble, not a variance study; the reproduction's 5-seed
protocol doesn't apply here). Checkpoint selection: `best.pth` = minimum loss
on the 600-image official test set. This matches (a) the reproduction's own
convention (itself matching the official EGE-UNet repo's methodology — no
held-out validation split distinct from the reported test set) and (b) this
project's `--k_fold allin` convention used for the reference SwinUnet run,
which also selects `best.pth` by test-set metric with no separate held-out
split. Kept for cross-network methodological consistency, not because it's
best practice — see `train.py`'s docstring.

## Evaluation protocol

`evaluate.py` runs the trained checkpoint on the 600 official test images
with:
- **5-view TTA** (`AViT/Utils/tta.py::tta_forward` — original + h-flip +
  v-flip + rot90 + rot270, sigmoid-averaged), reused unmodified. EGE-UNet's
  `forward()` returns already-sigmoid probabilities, so `evaluate.py` inverts
  through a clamped `logit()` before handing off, so the *exact same* shared
  TTA code runs for all three networks rather than a reimplementation.
- **Method A postprocessing** (`AViT/postprocess_pipeline.py::morph_open_close`
  + `keep_largest_component`), reused unmodified.

Both confirmed with the user to match Swin-UNet/AViT's eval protocol exactly
(task's "Open Items", item 2), for outputs that are directly comparable and
combinable in the ensemble.

### Two Dice/IoU conventions, both reported

This project's existing code uses two different aggregation conventions for
"the" test Dice, and they disagree even on the *same* checkpoint with the
*same* eval conditions:

- **Batch-pooled, then batch-averaged** — `multi_train_adapt.py`'s own
  `test()`: `metrics.dc()` computed once over each mini-batch's pooled
  pixels (batch_size=16 for the reference SwinUnet run), then averaged
  across batches. This is what `test_results.csv` in every `results/*/`
  folder reports.
- **Per-image mean** — `eval_pilot_vs_baseline.py`: `metrics.dc()` computed
  once per individual image, then averaged across all 600. Matches the
  official ISIC challenge's own metric convention.

For the reference SwinUnet checkpoint (`results/isic2017_swinunet_clahe_
seed42_dice37_shift5_SwinUnet_20260824_0145/final.pth`, no TTA, no
postprocessing in either case): batch-pooled = **0.8392**, per-image mean =
**0.8500**. Same checkpoint, same 600 images, same lack of TTA — the
0.011 gap is entirely the aggregation convention, not a bug.

`evaluate.py` reports **both**: `*_mean_per_image` (primary — matches the
ISIC challenge convention and this project's own per-image analysis
tooling) and `*_pooled_global` (a true single global confusion matrix over
all 600 images' pixels combined — a third, stricter variant of "pooled,"
included for cross-reference since it's neither of the two above exactly,
but closest in spirit to the batch-pooled convention at the limit of
batch_size = whole test set).

### Like-for-like SwinUnet comparison

The task's cited comparison numbers — SwinUnet 0.8352, AViT 0.7742
(`kfold_logs/tta_*_clahe_results.csv`) — are 5-fold CV means, WITH CLAHE +
TTA, but **without** the hard-image augmentation. That's a different
protocol than what EGE-UNet trains under here (official single split +
hard-aug + CLAHE + TTA), so it isn't a fair apples-to-apples number.

`eval_swinunet_reference.py` computes a genuine same-protocol number instead:
runs the *existing* reference SwinUnet checkpoint (already trained on the
identical 2185-image augmented set) through the identical TTA + Method A
eval `evaluate.py` uses for EGE-UNet. Pure inference on an already-trained
checkpoint — no retraining. See `results/swinunet_reference_summary.json`.

## Results

### SwinUnet reference (like-for-like protocol) — done

`eval_swinunet_reference.py` run against the existing reference checkpoint,
600 official test images, per-image mean Dice/IoU:

| Config | Dice | IoU |
|---|---|---|
| No TTA, no postprocessing | 0.8500 | 0.7681 |
| No TTA, + Method A | 0.8490 | 0.7669 |
| TTA (5-view) | 0.8529 | 0.7721 |
| **TTA + Method A** | **0.8521** | **0.7712** |

The no-TTA/no-postprocessing row (0.8500) exactly matches
`shift37_pilot_summary.json`'s independently-computed per-image mean —
confirms this script's Dice/IoU implementation is correct. TTA adds +0.29pp
over the raw pass; Method A postprocessing on top of TTA gives back about
0.08pp (a small, consistent decrease across both TTA and no-TTA — plausibly
Method A's largest-component filter occasionally trims legitimate secondary
lesion regions). **0.8521 Dice / 0.7712 IoU (TTA + Method A) is the number
EGE-UNet's own result should be compared against** for a genuine
same-protocol comparison — not the task-cited 0.8352 (different protocol,
see "Like-for-like SwinUnet comparison" above).

### EGE-UNet — done

300 epochs, single seed (42), 2185-image official+augmented training set.
Best checkpoint (min test-set loss) at **epoch 218**, val_loss=1.0315.
Final epoch 300: train_loss=0.5295, val_loss=1.4111 — worse than epoch 218's
val_loss, i.e. the run overfits somewhat past its best epoch, which is
exactly why checkpoint selection uses min-loss rather than the final epoch.

`evaluate.py`, 600 official test images, per-image mean Dice/IoU:

| Config | Dice | IoU |
|---|---|---|
| No TTA, no postprocessing | 0.8463 | 0.7619 |
| No TTA, + Method A | 0.8461 | 0.7618 |
| TTA (5-view) | 0.8486 | 0.7656 |
| **TTA + Method A** | **0.8481** | **0.7654** |
| TTA + Method A, pooled/global (cross-ref only) | 0.8532 | 0.7440 |

### AViT — retrained on the augmented pipeline, done

Reused AViT's exact existing recipe from the config that produced the
project's old 0.7742 Dice baseline (`isic2017_avit_clahe_ftl_k*` folders —
AdamW lr=1e-4, weight_decay=0.05, batch=16, 30 epochs, Focal Tversky Loss,
CLAHE via `image_subdir=Image_clahe`). Only two things changed, matching how
SwinUnet's own baseline-to-reference config diff looked (confirmed by diffing
the two `exp_config.yml`s before starting — verifying method, not assuming
it): `k_fold` from a 5-fold CV split to `allin` (train on everything, eval
only against the fixed 600-test-image set — same adaptation used for
SwinUnet's and EGE-UNet's production runs), and `meta_csv_name` from the
2000-image base set to the 2185-image augmented set. Single run, seed 42.
1-epoch smoke test passed before committing to the full run.

30 epochs, best epoch was **29** (the last epoch — still improving at the
end, unlike EGE-UNet which had already started overfitting past its best
epoch). Final training-script-reported test Dice (raw, no TTA, batch-pooled
convention): 0.7809 — not the comparison number, see below.

`eval_avit_reference.py` (same TTA + Method A protocol, same 600 test
images, per-image mean Dice/IoU):

| Config | Dice | IoU |
|---|---|---|
| No TTA, no postprocessing | 0.7946 | 0.6953 |
| No TTA, + Method A | 0.7977 | 0.7034 |
| TTA (5-view) | 0.7958 | 0.6993 |
| **TTA + Method A** | **0.7946** | **0.7025** |

### Three-way comparison — same protocol, same eval pipeline

| Network | Dice (TTA+Method A) | IoU (TTA+Method A) |
|---|---|---|
| **SwinUnet** (reference checkpoint) | **0.8521** | **0.7712** |
| **EGE-UNet** (this project) | 0.8481 | 0.7654 |
| **AViT** (retrained this task) | 0.7946 | 0.7025 |

All three now trained/evaluated under the identical protocol: official
2000/600 split + 185 hard-image augmentation + CLAHE, evaluated with 5-view
TTA + Method A postprocessing on the same 600 test images.

**Reading the gap**: SwinUnet and EGE-UNet land within half a point of each
other; AViT trails both by ~5.4–5.75pp Dice — a real, substantial gap, not
noise (std on AViT's per-image Dice is also visibly wider — 0.22 vs
SwinUnet's 0.17 and EGE-UNet's 0.17, consistent with a less stable model on
this task, not just a lower mean). Two things worth being explicit about
rather than reading this as "AViT is simply worse":

1. **The augmentation did help AViT** — its new number (0.7946) is +2.04pp
   above its own old no-augmentation baseline (0.7742), so the same
   intervention that helped the other two networks moved AViT in the right
   direction too. It just started from further behind and didn't close the
   gap.
2. **The old 0.7742 baseline was a 5-fold CV *mean*; this new number is a
   single run** — averaging across 5 independently-trained folds reduces
   variance relative to any one run, so part of the apparent "closing" from
   0.7742→0.7946 (and analogously, any comparison to SwinUnet/EGE-UNet's
   own single-run numbers) isn't perfectly apples-to-apples on that
   dimension, even though the *data protocol* now is. Flagging this rather
   than either inflating or discounting the gap based on it.

## Soft-voting ensemble — built, evaluated, does not beat the best single model

`ensemble.py` averages each model's raw TTA-averaged probability map (saved
by the three eval scripts at a common 224×224 resolution, `results/prob_maps/
{swinunet,egeunet,avit}/`; EGE-UNet's native 256×256 map is downsampled to
224 to match — 2 of 3 models are already there), thresholds at 0.5 (all
three individual models already use that threshold — confirmed, not
assumed), and applies the identical Method A postprocessing used for each
model individually.

Two variants, both true soft voting (probability averaging, not hard/majority
voting on binarized masks):

| Variant | Weights | Dice (+Method A) | IoU (+Method A) | Per-image std |
|---|---|---|---|---|
| **A — Equal** | 1/3, 1/3, 1/3 | 0.8459 | 0.7626 | 0.174 |
| **B — Dice-weighted** | 0.342 / 0.340 / 0.319 (SwinUnet/EGE-UNet/AViT) | 0.8465 | 0.7633 | 0.173 |

Weights for Variant B use the exact Dice values from the task spec
(0.8521/0.8481/0.7946), normalized to sum to 1 — not re-derived from IoU or
any other metric.

### Comparison against the individual models

| Model | Dice | IoU |
|---|---|---|
| **SwinUnet alone (best individual)** | **0.8521** | **0.7712** |
| EGE-UNet alone | 0.8481 | 0.7654 |
| Ensemble B (weighted) | 0.8465 | 0.7633 |
| Ensemble A (equal) | 0.8459 | 0.7626 |
| AViT alone | 0.7946 | 0.7025 |

**Neither ensemble variant beats SwinUnet alone**, on either Dice or IoU —
this is reported plainly rather than reframed, per the task's own instruction
to report a negative result if that's what the data shows. More strikingly,
**neither variant even beats EGE-UNet alone**: both ensembles land about
0.15–0.22pp below EGE-UNet's solo 0.8481 Dice, despite EGE-UNet being one of
the three inputs being averaged. AViT's weaker, higher-variance predictions
(std 0.22 vs ~0.17 for the other two) are dragging the combined probability
map down enough that pulling in its errors costs more than its correct
predictions add — even after down-weighting it in Variant B. The
Dice-weighted variant does close about a tenth of a point of that gap versus
equal-weighting (0.8465 vs 0.8459), confirming the weighting direction is
right, just not strong enough with only three models and this large a
performance spread between the best and worst.

**What this suggests, not yet acted on**: this points toward either
excluding AViT from the production ensemble (i.e. shipping SwinUnet +
EGE-UNet only, or SwinUnet alone) or weighting it far more aggressively than
a linear Dice-normalized scheme allows — a fixed 32% floor is still enough
weight for AViT's errors to outweigh its contribution here. Neither of those
changes has been implemented; this section reports what was actually built
and measured, not a recommendation to act on without further discussion.

Per-image results: `results/ensemble_per_image.csv`. Full summary (both
variants, no-Method-A rows too):
`results/ensemble_summary.json`.

## Follow-up: 2-network ensemble and AViT weight sweep

Direct follow-up to the 3-network result above, testing whether AViT helps
at *any* weight or should be excluded outright. No new inference — reuses
the exact same saved probability maps (`ensemble_sweep.py`).

### Step 1 — 2-network ensemble (SwinUnet + EGE-UNet, no AViT)

| Variant | Weights (Swin/EGE) | Dice | IoU | Per-image std |
|---|---|---|---|---|
| Equal | 0.500 / 0.500 | 0.8555 | 0.7745 | 0.164 |
| Dice-weighted | 0.501 / 0.499 | 0.8555 | 0.7745 | 0.164 |

(The two variants land on effectively the same weights, as expected given
how close SwinUnet and EGE-UNet's individual Dice scores already are —
0.8521 vs 0.8481 — so this is one real result, not two.)

**This beats SwinUnet alone**: 0.8555 vs 0.8521 Dice (+0.34pp), 0.7745 vs
0.7712 IoU (+0.33pp). This is currently the best-performing configuration
found anywhere in this project.

> **This is a 2-network result, not the 3-network ensemble originally
> specified in the project brief.** It is reported here as a finding, not
> substituted silently as "the" ensemble deliverable. Whether to adopt a
> 2-network (SwinUnet + EGE-UNet) ensemble in place of the originally-scoped
> 3-network one is a decision point for Prof. Samavi, not something decided
> unilaterally in this write-up.

### Step 2 — AViT weight sensitivity sweep (0% → 32%)

Remainder at each step split between SwinUnet/EGE-UNet in their own relative
Dice-weighted proportion (~0.501/0.499), so the 0% point exactly reproduces
Step 1's Dice-weighted result and the 32% point exactly reproduces the
3-network Ensemble B above — both matched exactly in the output, confirming
no computation drift between the three scripts.

| AViT weight | Dice | IoU | Per-image std | Beats SwinUnet alone (0.8521)? |
|---|---|---|---|---|
| 0% | **0.8555** | **0.7745** | 0.164 | **Yes** |
| 5% | 0.8547 | 0.7734 | 0.164 | Yes |
| 10% | 0.8539 | 0.7722 | 0.164 | Yes |
| 15% | 0.8525 | 0.7706 | 0.166 | Yes (barely) |
| 20% | 0.8513 | 0.7690 | 0.167 | No |
| 25% | 0.8496 | 0.7669 | 0.169 | No |
| 32% | 0.8464 | 0.7633 | 0.173 | No |

**Ensemble Dice is monotonically decreasing in AViT weight across the
entire tested range (0%–32%) — AViT's optimal weight is 0%.** There is no
point in the sweep where adding AViT improves on excluding it; every
increment of AViT weight trades away some of the 2-network ensemble's gain.
Per-image std also climbs monotonically alongside the Dice decline (0.164 →
0.173), consistent with AViT's own higher solo variance (0.22) propagating
into the combination rather than being averaged out — the error AViT
contributes is correlated/systematic enough that more data points (more
networks) doesn't cancel it the way independent noise would.

Note this doesn't mean AViT provides *zero* signal at every weight in an
absolute sense — weights up to 15% still land above SwinUnet-alone's 0.8521,
just below the 0%-weight ensemble's own 0.8555. The honest reading is
narrower than "AViT never helps at all": it's "AViT never helps *this
2-network ensemble*," which is a meaningfully weaker and more specific claim.

Full sweep data: `results/avit_weight_sweep.csv`. Summary:
`results/avit_weight_sweep_summary.json`.

## Current best-known configuration

As of this analysis, ranked by Dice on the 600-image official test set
(TTA + Method A, identical protocol throughout):

| Rank | Configuration | Dice | IoU |
|---|---|---|---|
| 1 | **2-network ensemble (SwinUnet + EGE-UNet)** | **0.8555** | **0.7745** |
| 2 | SwinUnet alone | 0.8521 | 0.7712 |
| 3 | EGE-UNet alone | 0.8481 | 0.7654 |
| 4 | 3-network ensemble, Dice-weighted (32% AViT) | 0.8465 | 0.7633 |
| 5 | 3-network ensemble, equal-weight | 0.8459 | 0.7626 |
| 6 | AViT alone | 0.7946 | 0.7025 |

## EGE-UNet's own hard-image augmentation pilot — negative result, stopping here

Same method used to build SwinUnet's 37-image and AViT's 189-image
hard-image sets, applied to EGE-UNet itself: trained a clean baseline (2000
official training images, no augmentation, no CV, seed 42, CLAHE, EGE-UNet's
own 300-epoch recipe — best val_loss=0.9810 @ epoch 296), then evaluated
that single checkpoint against both its own 2000 training images and the
600 official test images (single pass, no TTA, threshold 0.5 — matching how
the original SwinUnet/AViT bad-image sets were identified, not the
TTA+Method A protocol used for the production/ensemble numbers elsewhere in
this README).

- **94 hard training images** (Dice<0.7 of 2000, baseline mean 0.8965) →
  `bad_images_egeunet_seed42_dice94.csv`
- **58 hard test images** (Dice<0.7 of 600, baseline mean 0.8576) →
  `bad_images_egeunet_seed42_test58.csv`

Augmented via `AViT/build_shift_augmentation.py` (fully reused, unmodified)
— 94×5 = 470 images, all 5 techniques, 0 skips — merged into the canonical
`Image_clahe`/`Label` dirs. Pilot trained on the combined 2470-image set,
same recipe, 300 epochs (best val_loss=1.0239 @ epoch 128; this run crashed
twice on native CUDA aborts from concurrent GPU load and was resumed twice
using newly-added checkpoint-resume support in `train.py` — final result
below is from the completed run, not affected by the crashes themselves).

| | n | Baseline mean Dice | Pilot mean Dice | Still bad (Dice<0.7) |
|---|---|---|---|---|
| Hard training images | 94 | 0.5529 | **0.7619** | 25/94 (was 94/94) |
| Hard test images | 58 | 0.4632 | 0.4739 | 47/58 (was 58/58 by construction) |
| **Full test600** | 600 | **0.8576** | **0.8491** | 68/600 (was 58/600) |

| Paired one-sided t-test | n | mean_diff (pilot − baseline) | p (one-sided) | Gate (diff>0.003 & p<0.05) |
|---|---|---|---|---|
| **Full test600** | 600 | **−0.0085** | **0.995** | **fails — decision: stop** |
| Hard subset (58) | 58 | +0.0106 | 0.323 | fails (not significant) |

**Training-side recovery is real and strong** — 69 of 94 targeted images
recovered above Dice 0.7 (mean 0.55→0.76) — the model clearly learned the
augmented copies. **That does not transfer.** On the full test set the
pilot is *directionally worse* than the clean baseline (mean_diff=−0.85pp,
p=0.995 — strong evidence against improvement, not just "no evidence for
it"), and even on EGE-UNet's own hard test images the +1.06pp gain is not
statistically significant (p=0.32, 50/50 split between images that improved
and worsened). This is a textbook overfitting signature: memorizing 94
augmented training images without generalizing, unlike SwinUnet's own
shift37 pilot (which was at least directionally positive, mean_diff=+0.40pp,
just short of the significance bar) or AViT's shift189 pilot (which passed
significance and is queued for a full 5-fold sweep).

**Decision: stop, per the same bar used for every other technique in this
project (`mean_diff > 0.003 and p_one_sided < 0.05` on the full test600).**
No 5-fold sweep for EGE-UNet's hard-image augmentation — the pilot doesn't
clear it, and directionally points the wrong way. Reported as a negative
result rather than omitted, same discipline as the rest of this repo.

Full summary: `per_image_analysis_v2/overnight_run/egeunet_dice94_pilot_summary.json`.
Per-image CSVs: `EGE-UNet/results/pilot_egeunet_dice94_shift5/{hardtrain94,hardtest58,test600}_per_image.csv`.

## Open items still pending

- **Ensemble scope decision**: the best-known configuration (2-network,
  SwinUnet + EGE-UNet) diverges from the originally-specified 3-network
  ensemble. Whether to adopt it, keep AViT in at some non-zero weight despite
  the sweep showing 0% is optimal (e.g. for robustness reasons not captured
  by this single test set), or pursue further work on AViT specifically
  (different architecture variant, more training data, a non-linear
  combination scheme) before deciding definitively, is a decision point for
  Prof. Samavi — not resolved in this write-up.
- **EGE-UNet's own hard-image augmentation does not help** (see above) —
  the production EGE-UNet checkpoint used in the ensemble comparison remains
  the original `official_seed42` run (trained on SwinUnet's 37-image
  augmented set, per the earlier task), not this pilot. No action taken to
  swap it out.
