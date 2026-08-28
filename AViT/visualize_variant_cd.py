"""
Visual verification for Variants C (expanded-bbox zoom) and D (70%-fill zoom),
for all 23 bad training images from the allin run, no sampling.

For each image, saves one comparison PNG with 3 panels:
  1. Original + GT overlay + tight bbox (yellow) + expanded/D-crop bbox (cyan)
  2. Variant C (bbox expanded 20%, cropped+resized) + GT overlay
  3. Variant D (crop sized so lesion fills 70% of frame, cropped+resized) + GT overlay

Also builds paginated contact sheets (8 images x 3 panels per page).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DATA_ROOT = '../data/isic2017'
BAD23_CSV = '../per_image_analysis_v2/bad_image_augmentation/bad_images_allin23.csv'
BBOX_CSV = '../per_image_analysis_v2/bad_image_augmentation/bbox_coords_variant_cd.csv'
TIGHT_BBOX_CSV = '../per_image_analysis_v2/bad_image_augmentation/bbox_coords.csv'
OUT_DIR = '../per_image_analysis_v2/bad_image_augmentation/visual_verification_variant_cd'
CONTACT_SHEET_DIR = '../per_image_analysis_v2/bad_image_augmentation/contact_sheets_variant_cd'
IMAGES_PER_PAGE = 8

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CONTACT_SHEET_DIR, exist_ok=True)


def get_bbox(label):
    ys, xs = np.where(label > 0.5)
    h, w = label.shape
    if len(xs) == 0:
        return 0, 0, w - 1, h - 1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def overlay_mask(ax, img, lbl, title, bboxes=None):
    ax.imshow(img)
    mask_rgba = np.zeros((*lbl.shape, 4))
    mask_rgba[lbl > 0.5] = [0.15, 1.0, 0.15, 0.35]
    ax.imshow(mask_rgba)
    if bboxes:
        for (x_min, y_min, x_max, y_max), color in bboxes:
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                      linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
    ax.set_title(title, fontsize=10)
    ax.axis('off')


def load_all(img_id):
    orig_img = np.load(os.path.join(DATA_ROOT, 'Image_clahe', img_id + '.npy'))
    orig_lbl = np.load(os.path.join(DATA_ROOT, 'Label', img_id + '.npy')) > 0.5
    cropC_img = np.load(os.path.join(DATA_ROOT, 'Image_clahe_aug', img_id + '_cropC.npy'))
    cropC_lbl = np.load(os.path.join(DATA_ROOT, 'Label_aug', img_id + '_cropC.npy')) > 0.5
    cropD_img = np.load(os.path.join(DATA_ROOT, 'Image_clahe_aug', img_id + '_cropD.npy'))
    cropD_lbl = np.load(os.path.join(DATA_ROOT, 'Label_aug', img_id + '_cropD.npy')) > 0.5
    return orig_img, orig_lbl, cropC_img, cropC_lbl, cropD_img, cropD_lbl


def main():
    bad_df = pd.read_csv(BAD23_CSV, dtype={'image_id': str})
    assert len(bad_df) == 23, 'expected 23 bad images, got {}'.format(len(bad_df))
    bbox_df = pd.read_csv(BBOX_CSV, dtype={'image_id': str}).set_index(['image_id', 'variant'])
    tight_bbox_df = pd.read_csv(TIGHT_BBOX_CSV, dtype={'image_id': str}).set_index('image_id')
    has_tight = set(tight_bbox_df.index)

    saved_paths = []
    for _, row in bad_df.iterrows():
        img_id = row['image_id']
        dice = row['dice']
        bboxC = tuple(bbox_df.loc[(img_id, 'cropC'), ['x_min', 'y_min', 'x_max', 'y_max']].astype(int))
        bboxD = tuple(bbox_df.loc[(img_id, 'cropD'), ['x_min', 'y_min', 'x_max', 'y_max']].astype(int))

        orig_img, orig_lbl, cropC_img, cropC_lbl, cropD_img, cropD_lbl = load_all(img_id)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
        fig.suptitle('{}  (allin in-sample dice={:.3f})  expanded-bbox(cyan)=({},{})-({},{})  70%-fill(magenta)=({},{})-({},{})'.format(
            img_id, dice, *bboxC, *bboxD), fontsize=10)

        overlay_boxes = [(bboxC, 'cyan'), (bboxD, 'magenta')]
        if img_id in has_tight:
            tight = tuple(tight_bbox_df.loc[img_id, ['x_min', 'y_min', 'x_max', 'y_max']].astype(int))
            overlay_boxes.insert(0, (tight, 'yellow'))
        overlay_mask(axes[0], orig_img, orig_lbl, 'Original + GT + tight/expanded/70%-fill bboxes', bboxes=overlay_boxes)
        overlay_mask(axes[1], cropC_img, cropC_lbl, 'Variant C: bbox expanded 20% -> zoom, GT resized')
        overlay_mask(axes[2], cropD_img, cropD_lbl, 'Variant D: lesion fills 70% of frame -> zoom, GT resized')

        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, '{}_dice{:.3f}.png'.format(img_id, dice))
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        saved_paths.append(out_path)
        print('Saved:', out_path)

    print('Saved {} individual verification images to {}'.format(len(saved_paths), OUT_DIR))

    n_pages = (len(bad_df) + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE
    for page in range(n_pages):
        chunk = bad_df.iloc[page * IMAGES_PER_PAGE:(page + 1) * IMAGES_PER_PAGE]
        fig, axes = plt.subplots(len(chunk), 3, figsize=(13.5, 4.3 * len(chunk)))
        if len(chunk) == 1:
            axes = axes.reshape(1, 3)
        for i, (_, row) in enumerate(chunk.iterrows()):
            img_id = row['image_id']
            dice = row['dice']
            orig_img, orig_lbl, cropC_img, cropC_lbl, cropD_img, cropD_lbl = load_all(img_id)
            overlay_mask(axes[i, 0], orig_img, orig_lbl, '{} (dice={:.3f}) Original+GT'.format(img_id, dice))
            overlay_mask(axes[i, 1], cropC_img, cropC_lbl, 'Variant C (expand20%)')
            overlay_mask(axes[i, 2], cropD_img, cropD_lbl, 'Variant D (fill70%)')

        plt.tight_layout()
        sheet_path = os.path.join(CONTACT_SHEET_DIR, 'contact_sheet_page{}.png'.format(page + 1))
        fig.savefig(sheet_path, dpi=95)
        plt.close(fig)
        print('Saved contact sheet:', sheet_path)

    print('DONE')


if __name__ == '__main__':
    main()
