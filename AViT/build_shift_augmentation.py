"""
Builds the 5-technique augmentation set Prof. Samavi specified, for the 37 seed42
Dice<0.7 training images:
  a-d. horizontal flip, vertical flip, 90-degree rotation, 270-degree rotation
       (identical logic to build_augmentations_generic.py's make_hflip/vflip/
       rot90/rot270 -- reproduced here rather than imported so this script is
       self-contained and its skip-logic/report is easy to audit standalone)
  e-f. rightward horizontal shift by 10% of image width (revised down from 20%
       per Prof. Samavi's fix, after the original 20% version produced a visibly
       mirrored-duplicate seam for images with rulers/ink marks near the edge).
       Before shifting, the LEFT shift_px-wide strip of the ORIGINAL image (the
       exact region that ends up as cv2.copyMakeBorder's reflection source once
       shifted -- see make_shift_right) is Gaussian-blurred, so the mirrored
       copy that fills the gap is a soft, low-detail echo rather than a sharp
       duplicate of any ruler/ink marks. The GT mask is never blurred (binary,
       blurring would corrupt it) -- it shifts by the same amount with a simple
       zero-fill (background) on the newly-empty side. Whatever would clip off
       the right edge is dropped, for both image and mask.

       Skip condition: if the GT mask's rightmost lesion pixel would land at or
       past the right edge after the shift (i.e. any lesion pixels would be
       clipped, which for a real circular/wrap operation would reappear split
       off on the left -- visually splitting the lesion), the shift is skipped
       for that image entirely. All 4 other techniques still apply.
"""
import argparse
import os
import numpy as np
import pandas as pd
import cv2

DATA_ROOT = '../data/isic2017'
SRC_IMAGE_SUBDIR = 'Image_clahe'
SRC_LABEL_SUBDIR = 'Label'
OUT_IMAGE_SUBDIR = 'Image_clahe_aug'
OUT_LABEL_SUBDIR = 'Label_aug'
DEFAULT_BAD_CSV = '../per_image_analysis_v2/bad_image_augmentation/bad_images_seed42_dice37.csv'
DEFAULT_OUT_MANIFEST = '../per_image_analysis_v2/bad_image_augmentation/manifest_seed42_dice37_shift5.csv'
DEFAULT_OUT_SKIP_REPORT = '../per_image_analysis_v2/bad_image_augmentation/shift_skip_report.csv'
SHIFT_FRAC = 0.1
BLUR_KERNEL = 21  # must be odd; ~15-25px per Prof. Samavi's spec


def make_hflip(img, lbl):
    return img[:, ::-1, :].copy(), lbl[:, ::-1].copy()


def make_vflip(img, lbl):
    return img[::-1, :, :].copy(), lbl[::-1, :].copy()


def make_rot90(img, lbl):
    return np.rot90(img, k=1, axes=(0, 1)).copy(), np.rot90(lbl, k=1, axes=(0, 1)).copy()


def make_rot270(img, lbl):
    return np.rot90(img, k=3, axes=(0, 1)).copy(), np.rot90(lbl, k=3, axes=(0, 1)).copy()


def rightmost_lesion_col(lbl):
    ys, xs = np.where(lbl > 0.5)
    if len(xs) == 0:
        return -1
    return int(xs.max())


def make_shift_right(img, lbl, shift_px, blur_kernel):
    """img: (H,W,3) uint8. lbl: (H,W) float32 in {0,1}. Returns (shifted_img, shifted_lbl).

    The strip img[:, :shift_px] is exactly what cv2.copyMakeBorder's REFLECT_101
    padding mirrors from (kept_img's own left edge = the original image's first
    shift_px columns), so blurring that strip BEFORE the shift is what actually
    softens the mirrored copy that fills the gap.
    """
    h, w = lbl.shape
    img_prepped = img.copy()
    img_prepped[:, :shift_px, :] = cv2.GaussianBlur(img[:, :shift_px, :], (blur_kernel, blur_kernel), 0)

    kept_img = img_prepped[:, :w - shift_px, :]    # will end up at columns [shift_px, w)
    shifted_img = cv2.copyMakeBorder(kept_img, 0, 0, shift_px, 0, borderType=cv2.BORDER_REFLECT_101)

    shifted_lbl = np.zeros_like(lbl)
    shifted_lbl[:, shift_px:w] = lbl[:, :w - shift_px]
    return shifted_img, shifted_lbl


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bad_csv', default=DEFAULT_BAD_CSV)
    p.add_argument('--out_manifest', default=DEFAULT_OUT_MANIFEST)
    p.add_argument('--out_skip_report', default=DEFAULT_OUT_SKIP_REPORT)
    args = p.parse_args()
    BAD_CSV, OUT_MANIFEST, OUT_SKIP_REPORT = args.bad_csv, args.out_manifest, args.out_skip_report

    bad_df = pd.read_csv(BAD_CSV, dtype={'image_id': str})
    print('Images to augment: {}'.format(len(bad_df)))

    src_img_dir = os.path.join(DATA_ROOT, SRC_IMAGE_SUBDIR)
    src_lbl_dir = os.path.join(DATA_ROOT, SRC_LABEL_SUBDIR)
    out_img_dir = os.path.join(DATA_ROOT, OUT_IMAGE_SUBDIR)
    out_lbl_dir = os.path.join(DATA_ROOT, OUT_LABEL_SUBDIR)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    manifest_rows = []
    skip_rows = []
    n_shift_applied = 0
    n_shift_skipped = 0

    for _, row in bad_df.iterrows():
        img_id = row['image_id']
        orig_dice = row['dice']
        img = np.load(os.path.join(src_img_dir, img_id + '.npy'))
        lbl = (np.load(os.path.join(src_lbl_dir, img_id + '.npy')) > 0.5).astype(np.float32)
        h, w = lbl.shape
        shift_px = int(round(SHIFT_FRAC * w))

        variants = {
            'hflip': make_hflip(img, lbl),
            'vflip': make_vflip(img, lbl),
            'rot90': make_rot90(img, lbl),
            'rot270': make_rot270(img, lbl),
        }

        x_max = rightmost_lesion_col(lbl)
        would_clip = (x_max + shift_px) >= w
        if would_clip:
            n_shift_skipped += 1
            skip_rows.append({'image_id': img_id, 'x_max': x_max, 'shift_px': shift_px,
                               'width': w, 'reason': 'lesion rightmost col {} + shift {} >= width {}'.format(
                                   x_max, shift_px, w)})
        else:
            n_shift_applied += 1
            variants['shiftR10blur'] = make_shift_right(img, lbl, shift_px, BLUR_KERNEL)

        for suffix, (v_img, v_lbl) in variants.items():
            new_id = '{}_{}'.format(img_id, suffix)
            img_path = os.path.join(out_img_dir, new_id + '.npy')
            lbl_path = os.path.join(out_lbl_dir, new_id + '.npy')
            if not os.path.exists(img_path):
                np.save(img_path, v_img.astype(np.uint8))
            if not os.path.exists(lbl_path):
                np.save(lbl_path, v_lbl.astype(np.float32))
            manifest_rows.append({'ID': new_id, 'source_id': img_id, 'augmentation': suffix,
                                   'source_dice': orig_dice, 'dataset': 'isic2017',
                                   'diagnosis': 'unknown', 'diagnosis_id': 0})

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(OUT_MANIFEST, index=False)
    skip_df = pd.DataFrame(skip_rows, columns=['image_id', 'x_max', 'shift_px', 'width', 'reason'])
    skip_df.to_csv(OUT_SKIP_REPORT, index=False)

    print('Shift applied: {} / {}'.format(n_shift_applied, len(bad_df)))
    print('Shift SKIPPED (would clip lesion off right edge): {} / {}'.format(n_shift_skipped, len(bad_df)))
    if n_shift_skipped:
        print(skip_df.to_string(index=False))
    print('Manifest rows: {} ({} images x 4 always + {} shifted)'.format(
        len(manifest_df), len(bad_df), n_shift_applied))
    print('Saved manifest: {}'.format(OUT_MANIFEST))
    print('Saved skip report: {}'.format(OUT_SKIP_REPORT))
    print('SHIFT_AUG_DONE')


if __name__ == '__main__':
    main()
