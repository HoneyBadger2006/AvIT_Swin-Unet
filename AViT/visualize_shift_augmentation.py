"""
Visual verification for the 5-technique augmentation set (hflip, vflip, rot90,
rot270, rightward-shift-with-blurred-reflect-pad). No sampling. Any image where
the shift was skipped (would clip the lesion off the right edge) gets a clear
"SKIPPED" placeholder in its shift panel rather than silently omitted.

For each image: 6 panels -- Original+GT+tight bbox | hflip | vflip | rot90 |
rot270 | shift (or skipped notice). Paginated contact sheets, 4 images/page.
Generic over which bad-image list (37 SwinUnet, 189 AViT, etc.) via CLI args.
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DATA_ROOT = '../data/isic2017'
DEFAULT_BAD_CSV = '../per_image_analysis_v2/bad_image_augmentation/bad_images_seed42_dice37.csv'
DEFAULT_SKIP_CSV = '../per_image_analysis_v2/bad_image_augmentation/shift_skip_report.csv'
DEFAULT_OUT_DIR = '../per_image_analysis_v2/seed42_dice_bad37_shift/augmentation_verification'
IMAGES_PER_PAGE = 4
SHIFT_FRAC = 0.1


def get_bbox(label):
    ys, xs = np.where(label > 0.5)
    h, w = label.shape
    if len(xs) == 0:
        return 0, 0, w - 1, h - 1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def overlay_mask(ax, img, lbl, title, bbox=None):
    ax.imshow(img)
    mask_rgba = np.zeros((*lbl.shape, 4))
    mask_rgba[lbl > 0.5] = [0.15, 1.0, 0.15, 0.4]
    ax.imshow(mask_rgba)
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                  linewidth=1.5, edgecolor='yellow', facecolor='none')
        ax.add_patch(rect)
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def skipped_panel(ax, title):
    ax.text(0.5, 0.5, 'SHIFT SKIPPED\n(would clip lesion\noff right edge)',
            ha='center', va='center', fontsize=11, color='darkred',
            transform=ax.transAxes, bbox=dict(facecolor='mistyrose', edgecolor='darkred'))
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def load_variant(img_id, suffix):
    img = np.load(os.path.join(DATA_ROOT, 'Image_clahe_aug', '{}_{}.npy'.format(img_id, suffix)))
    lbl = np.load(os.path.join(DATA_ROOT, 'Label_aug', '{}_{}.npy'.format(img_id, suffix))) > 0.5
    return img, lbl


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bad_csv', default=DEFAULT_BAD_CSV)
    p.add_argument('--skip_csv', default=DEFAULT_SKIP_CSV)
    p.add_argument('--out_dir', default=DEFAULT_OUT_DIR)
    p.add_argument('--expected_n', type=int, default=37)
    args = p.parse_args()
    INDIVIDUAL_DIR = os.path.join(args.out_dir, 'individual')
    CONTACT_SHEET_DIR = os.path.join(args.out_dir, 'contact_sheets')
    os.makedirs(INDIVIDUAL_DIR, exist_ok=True)
    os.makedirs(CONTACT_SHEET_DIR, exist_ok=True)

    bad_df = pd.read_csv(args.bad_csv, dtype={'image_id': str})
    assert len(bad_df) == args.expected_n
    skipped_ids = set(pd.read_csv(args.skip_csv, dtype={'image_id': str})['image_id'])
    print('Verifying {} images ({} shift-skipped)'.format(len(bad_df), len(skipped_ids)))

    variant_titles = [('hflip', 'H-flip'), ('vflip', 'V-flip'), ('rot90', 'Rot 90'), ('rot270', 'Rot 270')]

    saved_paths = []
    for _, row in bad_df.iterrows():
        img_id = row['image_id']
        dice = row['dice']
        orig_img = np.load(os.path.join(DATA_ROOT, 'Image_clahe', img_id + '.npy'))
        orig_lbl = np.load(os.path.join(DATA_ROOT, 'Label', img_id + '.npy')) > 0.5
        bbox = get_bbox(orig_lbl.astype(np.float32))

        fig, axes = plt.subplots(1, 6, figsize=(27, 5))
        fig.suptitle('{}  (source dice={:.3f}){}'.format(
            img_id, dice, '  [SHIFT SKIPPED]' if img_id in skipped_ids else ''), fontsize=11)

        overlay_mask(axes[0], orig_img, orig_lbl, 'Original+GT+bbox', bbox=bbox)
        for i, (suffix, title) in enumerate(variant_titles, start=1):
            v_img, v_lbl = load_variant(img_id, suffix)
            overlay_mask(axes[i], v_img, v_lbl, title)

        if img_id in skipped_ids:
            skipped_panel(axes[5], 'Shift 10% right')
        else:
            v_img, v_lbl = load_variant(img_id, 'shiftR10blur')
            shift_px = int(round(SHIFT_FRAC * orig_lbl.shape[1]))
            overlay_mask(axes[5], v_img, v_lbl, 'Shift 10% right (blurred reflect-101)',
                        bbox=(shift_px, 0, orig_lbl.shape[1] - 1, orig_lbl.shape[0] - 1))

        plt.tight_layout()
        out_path = os.path.join(INDIVIDUAL_DIR, '{}_dice{:.3f}.png'.format(img_id, dice))
        fig.savefig(out_path, dpi=95)
        plt.close(fig)
        saved_paths.append(out_path)

    print('Saved {} individual verification images to {}'.format(len(saved_paths), INDIVIDUAL_DIR))

    n_pages = (len(bad_df) + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE
    for page in range(n_pages):
        chunk = bad_df.iloc[page * IMAGES_PER_PAGE:(page + 1) * IMAGES_PER_PAGE]
        fig, axes = plt.subplots(len(chunk), 6, figsize=(25, 4.3 * len(chunk)))
        if len(chunk) == 1:
            axes = axes.reshape(1, 6)
        for i, (_, row) in enumerate(chunk.iterrows()):
            img_id = row['image_id']
            dice = row['dice']
            orig_img = np.load(os.path.join(DATA_ROOT, 'Image_clahe', img_id + '.npy'))
            orig_lbl = np.load(os.path.join(DATA_ROOT, 'Label', img_id + '.npy')) > 0.5

            overlay_mask(axes[i, 0], orig_img, orig_lbl, '{} (dice={:.3f}) Original+GT'.format(img_id, dice))
            for j, (suffix, title) in enumerate(variant_titles, start=1):
                v_img, v_lbl = load_variant(img_id, suffix)
                overlay_mask(axes[i, j], v_img, v_lbl, title)
            if img_id in skipped_ids:
                skipped_panel(axes[i, 5], 'Shift 10% right')
            else:
                v_img, v_lbl = load_variant(img_id, 'shiftR10blur')
                overlay_mask(axes[i, 5], v_img, v_lbl, 'Shift 10% right')

        plt.tight_layout()
        sheet_path = os.path.join(CONTACT_SHEET_DIR, 'contact_sheet_page{:02d}.png'.format(page + 1))
        fig.savefig(sheet_path, dpi=85)
        plt.close(fig)
        print('Saved contact sheet:', sheet_path)

    print('Total contact sheet pages: {}'.format(n_pages))
    print('DONE')


if __name__ == '__main__':
    main()
