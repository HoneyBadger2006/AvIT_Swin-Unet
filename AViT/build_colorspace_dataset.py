"""
Full-dataset color-space conversion for Prof. Samavi's HSV vs. YCbCr vs. RGB pilot.
Converts every image in meta_isic2017_train2000.csv (2000) + meta_isic2017_test600.csv
(600) -- 2600 unique images, no overlap -- from the RAW (no-CLAHE) 'Image' subdir into
two new sibling subdirs: 'Image_hsv' and 'Image_ycbcr'. Label/ and all existing
Image*/ subdirs are untouched; this only adds two new directories.

Uses OpenCV's conventions: cv2.COLOR_RGB2HSV for HSV, cv2.COLOR_RGB2YCrCb for
"YCbCr" (channel order Y, Cr, Cb) -- same conversion functions already used and
visually sanity-checked in the earlier build_colorspace_variants.py scaffold.

Source images are the raw uint8 (H,W,3) arrays as stored (not yet resized -- resize
happens at dataset-load time via SkinDataset_csv, same as every other subdir), so
this is a straight per-file conversion with generate-if-missing semantics.
"""
import os
import numpy as np
import pandas as pd
import cv2

DATA_ROOT = '../data/isic2017'
SRC_SUBDIR = 'Image'
HSV_SUBDIR = 'Image_hsv'
YCBCR_SUBDIR = 'Image_ycbcr'


def main():
    train_df = pd.read_csv(os.path.join(DATA_ROOT, 'meta_isic2017_train2000.csv'), dtype={'ID': str})
    test_df = pd.read_csv(os.path.join(DATA_ROOT, 'meta_isic2017_test600.csv'), dtype={'ID': str})
    all_ids = pd.concat([train_df['ID'], test_df['ID']], ignore_index=True).unique()
    print('Total unique images to convert: {} (train={}, test={})'.format(len(all_ids), len(train_df), len(test_df)))
    assert len(all_ids) == 2600

    src_dir = os.path.join(DATA_ROOT, SRC_SUBDIR)
    hsv_dir = os.path.join(DATA_ROOT, HSV_SUBDIR)
    ycbcr_dir = os.path.join(DATA_ROOT, YCBCR_SUBDIR)
    os.makedirs(hsv_dir, exist_ok=True)
    os.makedirs(ycbcr_dir, exist_ok=True)

    n_hsv_written = 0
    n_ycbcr_written = 0
    for i, image_id in enumerate(all_ids):
        src_path = os.path.join(src_dir, image_id + '.npy')
        img = np.load(src_path).astype(np.uint8)

        hsv_path = os.path.join(hsv_dir, image_id + '.npy')
        if not os.path.exists(hsv_path):
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            np.save(hsv_path, hsv)
            n_hsv_written += 1

        ycbcr_path = os.path.join(ycbcr_dir, image_id + '.npy')
        if not os.path.exists(ycbcr_path):
            ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            np.save(ycbcr_path, ycbcr)
            n_ycbcr_written += 1

        if (i + 1) % 500 == 0:
            print('  {}/{} converted'.format(i + 1, len(all_ids)))

    print('HSV files written: {} (skipped {} already present)'.format(n_hsv_written, len(all_ids) - n_hsv_written))
    print('YCbCr files written: {} (skipped {} already present)'.format(n_ycbcr_written, len(all_ids) - n_ycbcr_written))

    hsv_count = len([f for f in os.listdir(hsv_dir) if f.endswith('.npy')])
    ycbcr_count = len([f for f in os.listdir(ycbcr_dir) if f.endswith('.npy')])
    print('Final Image_hsv/ file count: {}'.format(hsv_count))
    print('Final Image_ycbcr/ file count: {}'.format(ycbcr_count))
    assert hsv_count == 2600
    assert ycbcr_count == 2600
    print('COLORSPACE_DATASET_DONE')


if __name__ == '__main__':
    main()
