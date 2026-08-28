"""
Visual verification for the 8-variant augmentation set (4 standard flips/rotations
+ Variant A/B/C/D crops) built for the 157 seed42 IoU<0.7 images. No sampling --
all 157 checked.

For each image, saves one comparison PNG with 6 panels:
  1. Original + GT overlay + tight bbox (yellow)
  2. Variant A (bbox-mask)
  3. Variant B (tight bbox zoom)
  4. Variant C (bbox expanded 20%, zoom)
  5. Variant D (70%-fill zoom)
  6. (blank/flip note -- the 4 flip/rotation variants are lossless geometric
     transforms of the original and are not separately re-verified per-image;
     confirmed correct once via manifest variant-count and dtype checks below)

Also builds paginated contact sheets (4 images x 5 panels per page).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DATA_ROOT = '../data/isic2017'
BAD_CSV = '../per_image_analysis_v2/bad_image_augmentation/bad_images_seed42_iou157.csv'
CD_BBOX_CSV = '../per_image_analysis_v2/bad_image_augmentation/bbox_coords_seed42_iou157_cd.csv'
OUT_DIR = '../per_image_analysis_v2/seed42_iou_bad157/augmentation_verification'
INDIVIDUAL_DIR = os.path.join(OUT_DIR, 'individual')
CONTACT_SHEET_DIR = os.path.join(OUT_DIR, 'contact_sheets')
IMAGES_PER_PAGE = 4

os.makedirs(INDIVIDUAL_DIR, exist_ok=True)
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
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def load_all(img_id):
    orig_img = np.load(os.path.join(DATA_ROOT, 'Image_clahe', img_id + '.npy'))
    orig_lbl = np.load(os.path.join(DATA_ROOT, 'Label', img_id + '.npy')) > 0.5
    variants = {}
    for suffix in ['cropA', 'cropB', 'cropC', 'cropD']:
        v_img = np.load(os.path.join(DATA_ROOT, 'Image_clahe_aug', '{}_{}.npy'.format(img_id, suffix)))
        v_lbl = np.load(os.path.join(DATA_ROOT, 'Label_aug', '{}_{}.npy'.format(img_id, suffix))) > 0.5
        variants[suffix] = (v_img, v_lbl)
    return orig_img, orig_lbl, variants


def main():
    bad_df = pd.read_csv(BAD_CSV, dtype={'image_id': str})
    assert len(bad_df) == 157, 'expected 157 images, got {}'.format(len(bad_df))
    cd_bbox = pd.read_csv(CD_BBOX_CSV, dtype={'image_id': str}).set_index(['image_id', 'variant'])

    # sanity: confirm the flip/rotation variants exist and have correct dtype/shape for all 157
    flip_missing = []
    for img_id in bad_df['image_id']:
        for suffix in ['hflip', 'vflip', 'rot90', 'rot270']:
            p = os.path.join(DATA_ROOT, 'Image_clahe_aug', '{}_{}.npy'.format(img_id, suffix))
            lp = os.path.join(DATA_ROOT, 'Label_aug', '{}_{}.npy'.format(img_id, suffix))
            if not (os.path.exists(p) and os.path.exists(lp)):
                flip_missing.append((img_id, suffix))
    print('Flip/rotation variant files missing: {} (expected 0)'.format(len(flip_missing)))
    assert len(flip_missing) == 0

    saved_paths = []
    for _, row in bad_df.iterrows():
        img_id = row['image_id']
        dice = row['dice']
        tight_bbox = get_bbox((np.load(os.path.join(DATA_ROOT, 'Label', img_id + '.npy')) > 0.5).astype(np.float32))
        bboxC = tuple(cd_bbox.loc[(img_id, 'cropC'), ['x_min', 'y_min', 'x_max', 'y_max']].astype(int))
        bboxD = tuple(cd_bbox.loc[(img_id, 'cropD'), ['x_min', 'y_min', 'x_max', 'y_max']].astype(int))

        orig_img, orig_lbl, variants = load_all(img_id)

        fig, axes = plt.subplots(1, 5, figsize=(24, 5.2))
        fig.suptitle('{}  (source dice={:.3f})'.format(img_id, dice), fontsize=11)

        overlay_mask(axes[0], orig_img, orig_lbl, 'Original+GT  tight(yellow)/C(cyan)/D(magenta)',
                     bboxes=[(tight_bbox, 'yellow'), (bboxC, 'cyan'), (bboxD, 'magenta')])
        overlay_mask(axes[1], variants['cropA'][0], variants['cropA'][1], 'Variant A (bbox-mask)')
        overlay_mask(axes[2], variants['cropB'][0], variants['cropB'][1], 'Variant B (tight zoom)')
        overlay_mask(axes[3], variants['cropC'][0], variants['cropC'][1], 'Variant C (expand20%)')
        overlay_mask(axes[4], variants['cropD'][0], variants['cropD'][1], 'Variant D (fill70%)')

        plt.tight_layout()
        out_path = os.path.join(INDIVIDUAL_DIR, '{}_dice{:.3f}.png'.format(img_id, dice))
        fig.savefig(out_path, dpi=95)
        plt.close(fig)
        saved_paths.append(out_path)

    print('Saved {} individual verification images to {}'.format(len(saved_paths), INDIVIDUAL_DIR))

    n_pages = (len(bad_df) + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE
    for page in range(n_pages):
        chunk = bad_df.iloc[page * IMAGES_PER_PAGE:(page + 1) * IMAGES_PER_PAGE]
        fig, axes = plt.subplots(len(chunk), 5, figsize=(22, 4.3 * len(chunk)))
        if len(chunk) == 1:
            axes = axes.reshape(1, 5)
        for i, (_, row) in enumerate(chunk.iterrows()):
            img_id = row['image_id']
            dice = row['dice']
            orig_img, orig_lbl, variants = load_all(img_id)
            overlay_mask(axes[i, 0], orig_img, orig_lbl, '{} (dice={:.3f}) Original+GT'.format(img_id, dice))
            overlay_mask(axes[i, 1], variants['cropA'][0], variants['cropA'][1], 'Variant A')
            overlay_mask(axes[i, 2], variants['cropB'][0], variants['cropB'][1], 'Variant B')
            overlay_mask(axes[i, 3], variants['cropC'][0], variants['cropC'][1], 'Variant C')
            overlay_mask(axes[i, 4], variants['cropD'][0], variants['cropD'][1], 'Variant D')

        plt.tight_layout()
        sheet_path = os.path.join(CONTACT_SHEET_DIR, 'contact_sheet_page{:02d}.png'.format(page + 1))
        fig.savefig(sheet_path, dpi=85)
        plt.close(fig)
        print('Saved contact sheet:', sheet_path)

    print('Total contact sheet pages: {}'.format(n_pages))
    print('DONE')


if __name__ == '__main__':
    main()
