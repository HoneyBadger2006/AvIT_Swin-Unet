"""
Scaffold for the HSV/YCbCr color-space experiment (Priority 3 stretch goal).
Converts a handful of sample CLAHE images to HSV and YCbCr and saves side-by-side
visual sanity-check panels, so the conversion logic can be reviewed before any
full-dataset generation or retraining is committed to. Not wired into training.
"""
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

DATA_ROOT = '../data/isic2017'
OUT_DIR = '../per_image_analysis_v2/colorspace_experiment/sanity_check'
N_SAMPLES = 8

os.makedirs(OUT_DIR, exist_ok=True)


def to_hsv(img_uint8):
    return cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)


def to_ycbcr(img_uint8):
    return cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YCrCb)


def main():
    meta = pd.read_csv(os.path.join(DATA_ROOT, 'meta_isic2017_train2000.csv'), dtype={'ID': str})
    sample_ids = meta['ID'].sample(N_SAMPLES, random_state=42).tolist()

    for image_id in sample_ids:
        img = np.load(os.path.join(DATA_ROOT, 'Image_clahe', image_id + '.npy')).astype(np.uint8)
        hsv = to_hsv(img)
        ycbcr = to_ycbcr(img)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
        axes[0].imshow(img); axes[0].set_title('CLAHE RGB'); axes[0].axis('off')
        axes[1].imshow(hsv); axes[1].set_title('HSV (raw channels as RGB)'); axes[1].axis('off')
        axes[2].imshow(ycbcr); axes[2].set_title('YCbCr (raw channels as RGB)'); axes[2].axis('off')
        fig.suptitle(image_id, fontsize=11)
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, '{}_colorspace_sanity.png'.format(image_id))
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        print('Saved:', out_path)

    print('COLORSPACE_SANITY_DONE')


if __name__ == '__main__':
    main()
