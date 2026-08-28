"""
Visual verification for ONLY the 14 images that are new to the corrected (CLAHE-only
checkpoint) bad-image list, i.e. not part of the original 39-image FTL-based list that
was already fully visually verified and approved. Same 3-panel layout as before.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DATA_ROOT = '../data/isic2017'
OLD_BAD_CSV = '../per_image_analysis_v2/bad_image_augmentation/bad_images_fold0.csv'
NEW_BAD_CSV = '../per_image_analysis_v2/bad_image_augmentation/bad_images_fold0_clahe_only.csv'
BBOX_CSV = '../per_image_analysis_v2/bad_image_augmentation/bbox_coords.csv'
OUT_DIR = '../per_image_analysis_v2/bad_image_augmentation/visual_verification_new'

os.makedirs(OUT_DIR, exist_ok=True)


def overlay_mask(ax, img, lbl, title, bbox=None):
    ax.imshow(img)
    mask_rgba = np.zeros((*lbl.shape, 4))
    mask_rgba[lbl > 0.5] = [0.15, 1.0, 0.15, 0.35]
    ax.imshow(mask_rgba)
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                  linewidth=2, edgecolor='yellow', facecolor='none')
        ax.add_patch(rect)
    ax.set_title(title, fontsize=10)
    ax.axis('off')


def main():
    old_ids = set(pd.read_csv(OLD_BAD_CSV, dtype={'image_id': str})['image_id'])
    new_df = pd.read_csv(NEW_BAD_CSV, dtype={'image_id': str})
    bbox_df = pd.read_csv(BBOX_CSV, dtype={'image_id': str}).set_index('image_id')

    delta_df = new_df[~new_df['image_id'].isin(old_ids)].reset_index(drop=True)
    print('New (never-before-seen) images to verify: {}'.format(len(delta_df)))

    n = len(delta_df)
    fig, axes = plt.subplots(n, 3, figsize=(13.5, 4.3 * n))
    if n == 1:
        axes = axes.reshape(1, 3)

    for i, row in delta_df.iterrows():
        img_id = row['image_id']
        dice = row['dice']
        bbox = tuple(bbox_df.loc[img_id, ['x_min', 'y_min', 'x_max', 'y_max']].astype(int))

        orig_img = np.load(os.path.join(DATA_ROOT, 'Image_clahe', img_id + '.npy'))
        orig_lbl = np.load(os.path.join(DATA_ROOT, 'Label', img_id + '.npy')) > 0.5
        cropA_img = np.load(os.path.join(DATA_ROOT, 'Image_clahe_aug', img_id + '_cropA.npy'))
        cropA_lbl = np.load(os.path.join(DATA_ROOT, 'Label_aug', img_id + '_cropA.npy')) > 0.5
        cropB_img = np.load(os.path.join(DATA_ROOT, 'Image_clahe_aug', img_id + '_cropB.npy'))
        cropB_lbl = np.load(os.path.join(DATA_ROOT, 'Label_aug', img_id + '_cropB.npy')) > 0.5

        overlay_mask(axes[i, 0], orig_img, orig_lbl, 'NEW: {} (CLAHE-only dice={:.3f}) Original+GT+bbox'.format(img_id, dice), bbox=bbox)
        overlay_mask(axes[i, 1], cropA_img, cropA_lbl, 'Variant A (bbox-mask)')
        overlay_mask(axes[i, 2], cropB_img, cropB_lbl, 'Variant B (bbox-zoom)')

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'new_bad_images_delta.png')
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print('Saved:', out_path)
    print('DONE')


if __name__ == '__main__':
    main()
