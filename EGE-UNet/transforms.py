# Data augmentation transforms, adapted from the official implementation's
# utils.py (myToTensor / myResize / myRandomHorizontalFlip / myRandomVerticalFlip
# / myRandomRotation / myNormalize).
#
# Source: https://github.com/JCruan519/EGE-UNet (Apache-2.0)
# Commit: f52ba30c6bf7d0ca479c2c9d4a3cbda999f49d3a
#
# DOCUMENTED DIVERGENCES from official code (flagged per task instructions):
#
# 1. The official myRandomRotation samples its rotation angle ONCE in
#    __init__, when the transforms.Compose pipeline object is built (a
#    single time for the whole training run) -- not per __call__. Since
#    torchvision's Compose is constructed once and reused for every
#    __getitem__, every image in every epoch that gets rotated (p=0.5 coin
#    flip, checked per-call) is rotated by the *same* fixed angle for the
#    entire run. The paper just says "random rotation" as an augmentation,
#    which we read as intending a fresh angle per sample. This file
#    resamples the angle on every __call__.
#
# 2. The official pipeline resizes to 256x256 LAST, after flip/rotation, so
#    every augmentation runs on the full-resolution source image (some ISIC
#    source photos are several thousand pixels per side). This is faithful
#    to the official code but pathologically slow (measured ~13 min/epoch
#    on an RTX 4060, i.e. ~27 GPU-days for the full 5-seed x 2-dataset x
#    300-epoch protocol) since the model itself is tiny (~0.05M params) and
#    trains almost instantly -- the bottleneck is entirely CPU-bound image
#    decode/augment. We move Resize to run right after loading, before
#    flip/rotation. For horizontal/vertical flip this is an EXACT no-op
#    change: flip and resize commute pixel-for-pixel (resize(flip(x)) ==
#    flip(resize(x))), so output is identical either way. For rotation,
#    resize-then-rotate vs rotate-then-resize differ slightly in
#    interpolation/anti-aliasing (a minor image-quality effect, not a
#    change to what "random rotation augmentation" means statistically).
#    Confirmed with the user before applying, given the ~10x runtime
#    difference this makes to the whole reproduction.

import random
import numpy as np
import torch
import torchvision.transforms.functional as TF

# Official repo's hardcoded per-dataset TRAIN mean/std (utils.py myNormalize,
# commit f52ba30), used only for the norm-mode="official_fixed" ablation run
# (see scripts/run_diagnostics.py). The official code also uses separate
# (different) constants for its val split (isic17: 148.429/25.748; isic18:
# 149.034/32.022) -- we deliberately do NOT reproduce that train/test split
# here, since this ablation isolates one factor at a time (which mean/std
# *value* is used, applied uniformly to train+test transforms, same as the
# persplit mode already does) rather than also introducing a second new
# factor (different stats for train vs test).
OFFICIAL_FIXED_NORM_STATS = {
    "isic17": (159.922, 28.871),
    "isic18": (157.561, 26.706),
}


class ToTensor:
    def __call__(self, data):
        image, mask = data
        return torch.tensor(image).permute(2, 0, 1), torch.tensor(mask).permute(2, 0, 1)


class Resize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w

    def __call__(self, data):
        image, mask = data
        return TF.resize(image, [self.size_h, self.size_w]), TF.resize(mask, [self.size_h, self.size_w])


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.hflip(image), TF.hflip(mask)
        return image, mask


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.vflip(image), TF.vflip(mask)
        return image, mask


class RandomRotation:
    """Random rotation. By default (resample_each_call=True) samples a fresh
    angle on every __call__ (divergence #1 fix, see module docstring).
    Setting resample_each_call=False reproduces the official code's bug
    exactly: one angle sampled at __init__ time and reused for every rotated
    sample for the whole training run (used only by the rotation ablation in
    scripts/run_diagnostics.py to measure that fix's contribution)."""

    def __init__(self, p=0.5, degree=(0, 360), resample_each_call=True):
        self.p = p
        self.degree = degree
        self.resample_each_call = resample_each_call
        if not resample_each_call:
            self.angle = random.uniform(degree[0], degree[1])

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            angle = random.uniform(self.degree[0], self.degree[1]) if self.resample_each_call else self.angle
            return TF.rotate(image, angle), TF.rotate(mask, angle)
        return image, mask


class Normalize:
    """
    Z-score normalize using per-split train-set mean/std, then min-max
    stretch to [0, 255] -- matches the official myNormalize's math exactly,
    but takes mean/std as constructor args instead of hardcoding them per
    dataset.

    DOCUMENTED DIVERGENCE: the official code hardcodes mean/std constants
    (e.g. isic17 train: mean=159.922, std=28.871) that were precomputed on
    ONE specific fixed official split. Our protocol draws 5 different random
    70/30 splits per dataset, so those constants don't correspond to our
    train subsets. We recompute mean/std from each split's own training
    images (see scripts/compute_stats.py), which is the statistically
    correct thing to do for a resampled-split protocol, at the cost of
    exact numeric fidelity to the official hardcoded constants.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        img, msk = data
        img_normalized = (img - self.mean) / self.std
        img_normalized = (
            (img_normalized - np.min(img_normalized))
            / (np.max(img_normalized) - np.min(img_normalized))
        ) * 255.0
        return img_normalized, msk


def build_train_transform(mean, std, size_h=256, size_w=256, resample_rotation_each_call=True):
    from torchvision import transforms

    return transforms.Compose([
        Normalize(mean, std),
        ToTensor(),
        Resize(size_h, size_w),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation(p=0.5, degree=(0, 360), resample_each_call=resample_rotation_each_call),
    ])


def build_test_transform(mean, std, size_h=256, size_w=256):
    from torchvision import transforms

    return transforms.Compose([
        Normalize(mean, std),
        ToTensor(),
        Resize(size_h, size_w),
    ])
