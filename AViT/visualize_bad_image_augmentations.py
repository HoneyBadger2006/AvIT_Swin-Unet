"""
Full visual verification set for the bad-image crop augmentations (per user request:
every one of the 39 bad images, no sampling).

For each bad image, saves one full-resolution comparison PNG with 3 panels:
  1. Original + GT overlay + the computed bounding box drawn in yellow
  2. Variant A (bbox-mask: outside-box filled with mean color) + GT overlay
  3. Variant B (bbox-zoom: cropped to bbox then resized back to 512x512) + GT overlay

Also builds paginated contact sheets (8 images x 3 panels per page) for quick
browsing of the full set.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DATA_ROOT = '../data/isic2017'
BAD_IMAGES_CSV = '../per_image_analysis_v2/bad_image_augmentation/bad_images_fold0.csv'
BBOX_CSV = '../per_image_analysis_v2/bad_image_augmentation/bbox_coords.csv'
OUT_DIR = '../per_image_analysis_v2/bad_image_augmentation/visual_verification'
CONTACT_SHEET_DIR = '../per_image_analysis_v2/bad_image_augmentation/contact_sheets'
IMAGES_PER_PAGE = 8

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CONTACT_SHEET_DIR, exist_ok=True)


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
    bad_df = pd.read_csv(BAD_IMAGES_CSV, dtype={'image_id': str})
    bbox_df = pd.read_csv(BBOX_CSV, dtype={'image_id': str}).set_index('image_id')

    saved_paths = []
    for _, row in bad_df.iterrows():
        img_id = row['image_id']
        dice = row['dice']
        bbox = tuple(bbox_df.loc[img_id, ['x_min', 'y_min', 'x_max', 'y_max']].astype(int))

        orig_img = np.load(os.path.join(DATA_ROOT, 'Image_clahe', img_id + '.npy'))
        orig_lbl = np.load(os.path.join(DATA_ROOT, 'Label', img_id + '.npy')) > 0.5

        cropA_img = np.load(os.path.join(DATA_ROOT, 'Image_clahe_aug', img_id + '_cropA.npy'))
        cropA_lbl = np.load(os.path.join(DATA_ROOT, 'Label_aug', img_id + '_cropA.npy')) > 0.5

        cropB_img = np.load(os.path.join(DATA_ROOT, 'Image_clahe_aug', img_id + '_cropB.npy'))
        cropB_lbl = np.load(os.path.join(DATA_ROOT, 'Label_aug', img_id + '_cropB.npy')) > 0.5

        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
        fig.suptitle('{}  (fold-0 in-sample dice={:.3f})  bbox=({},{})-({},{})'.format(
            img_id, dice, *bbox), fontsize=12)

        overlay_mask(axes[0], orig_img, orig_lbl, 'Original + GT + bbox', bbox=bbox)
        overlay_mask(axes[1], cropA_img, cropA_lbl, 'Variant A: outside-bbox -> mean color, GT unchanged')
        overlay_mask(axes[2], cropB_img, cropB_lbl, 'Variant B: bbox crop -> resized to 512x512 (zoom), GT resized')

        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, '{}_dice{:.3f}.png'.format(img_id, dice))
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        saved_paths.append(out_path)
        print('Saved:', out_path)

    print('Saved {} individual verification images to {}'.format(len(saved_paths), OUT_DIR))

    # Paginated contact sheets, IMAGES_PER_PAGE bad-images per page, 3 panels each
    n_pages = (len(bad_df) + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE
    for page in range(n_pages):
        chunk = bad_df.iloc[page * IMAGES_PER_PAGE:(page + 1) * IMAGES_PER_PAGE]
        fig, axes = plt.subplots(len(chunk), 3, figsize=(13.5, 4.3 * len(chunk)))
        if len(chunk) == 1:
            axes = axes.reshape(1, 3)
        for i, (_, row) in enumerate(chunk.iterrows()):
            img_id = row['image_id']
            dice = row['dice']
            bbox = tuple(bbox_df.loc[img_id, ['x_min', 'y_min', 'x_max', 'y_max']].astype(int))

            orig_img = np.load(os.path.join(DATA_ROOT, 'Image_clahe', img_id + '.npy'))
            orig_lbl = np.load(os.path.join(DATA_ROOT, 'Label', img_id + '.npy')) > 0.5
            cropA_img = np.load(os.path.join(DATA_ROOT, 'Image_clahe_aug', img_id + '_cropA.npy'))
            cropA_lbl = np.load(os.path.join(DATA_ROOT, 'Label_aug', img_id + '_cropA.npy')) > 0.5
            cropB_img = np.load(os.path.join(DATA_ROOT, 'Image_clahe_aug', img_id + '_cropB.npy'))
            cropB_lbl = np.load(os.path.join(DATA_ROOT, 'Label_aug', img_id + '_cropB.npy')) > 0.5

            overlay_mask(axes[i, 0], orig_img, orig_lbl, '{} (dice={:.3f}) Original+GT+bbox'.format(img_id, dice), bbox=bbox)
            overlay_mask(axes[i, 1], cropA_img, cropA_lbl, 'Variant A (bbox-mask)')
            overlay_mask(axes[i, 2], cropB_img, cropB_lbl, 'Variant B (bbox-zoom)')

        plt.tight_layout()
        sheet_path = os.path.join(CONTACT_SHEET_DIR, 'contact_sheet_page{}.png'.format(page + 1))
        fig.savefig(sheet_path, dpi=95)
        plt.close(fig)
        print('Saved contact sheet:', sheet_path)

    print('DONE')


if __name__ == '__main__':
    main()
