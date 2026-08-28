"""
Visual verification (a few examples, per Prof. Samavi's pilot spec -- not exhaustive
like the bbox verification) that the full HSV and YCbCr dataset conversion produced
correct, sensible-looking data before committing to training. For each of 6 sample
images (from the raw 'Image' subdir), shows:
  RGB original | H | S | V | Y | Cr | Cb
so each individual channel's content can be sanity-checked directly, not just a
raw-channels-as-RGB composite.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_ROOT = '../data/isic2017'
OUT_PATH = '../per_image_analysis_v2/colorspace_experiment/full_dataset_verification.png'
N_SAMPLES = 6

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)


def main():
    meta = pd.read_csv(os.path.join(DATA_ROOT, 'meta_isic2017_train2000.csv'), dtype={'ID': str})
    sample_ids = meta['ID'].sample(N_SAMPLES, random_state=7).tolist()

    fig, axes = plt.subplots(N_SAMPLES, 7, figsize=(21, 3 * N_SAMPLES))
    col_titles = ['RGB', 'H', 'S', 'V', 'Y', 'Cr', 'Cb']

    for row, image_id in enumerate(sample_ids):
        rgb = np.load(os.path.join(DATA_ROOT, 'Image', image_id + '.npy'))
        hsv = np.load(os.path.join(DATA_ROOT, 'Image_hsv', image_id + '.npy'))
        ycrcb = np.load(os.path.join(DATA_ROOT, 'Image_ycbcr', image_id + '.npy'))

        panels = [rgb, hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2],
                  ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]]
        for col, (panel, title) in enumerate(zip(panels, col_titles)):
            ax = axes[row, col]
            if col == 0:
                ax.imshow(panel)
            else:
                ax.imshow(panel, cmap='gray', vmin=0, vmax=255)
            if row == 0:
                ax.set_title(title, fontsize=12)
            if col == 0:
                ax.set_ylabel(image_id, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=100)
    plt.close(fig)
    print('Saved:', OUT_PATH)
    print('VERIFY_DONE')


if __name__ == '__main__':
    main()
