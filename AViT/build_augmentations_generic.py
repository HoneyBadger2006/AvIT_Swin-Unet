"""
Generic version of build_bad_image_augmentations.py: given any bad-images CSV,
generates the 6 augmented variants (hflip/vflip/rot90/rot270/cropA/cropB) for each
image and writes the manifest. Crop math depends only on the GT mask, so it is
fold/network-independent -- files are written with generate-if-missing semantics
so the same augmented pair can be safely reused across folds/networks that happen
to flag the same source image as bad.
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


def get_bbox(label):
    """Tight bbox exactly as computed from the GT mask -- no margin added."""
    ys, xs = np.where(label > 0.5)
    h, w = label.shape
    if len(xs) == 0:
        return 0, 0, w - 1, h - 1
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return int(x_min), int(y_min), int(x_max), int(y_max)


def make_hflip(img, lbl):
    return img[:, ::-1, :].copy(), lbl[:, ::-1].copy()


def make_vflip(img, lbl):
    return img[::-1, :, :].copy(), lbl[::-1, :].copy()


def make_rot90(img, lbl):
    return np.rot90(img, k=1, axes=(0, 1)).copy(), np.rot90(lbl, k=1, axes=(0, 1)).copy()


def make_rot270(img, lbl):
    return np.rot90(img, k=3, axes=(0, 1)).copy(), np.rot90(lbl, k=3, axes=(0, 1)).copy()


def make_cropA_bbox_mask(img, lbl):
    x_min, y_min, x_max, y_max = get_bbox(lbl)
    mean_color = img.reshape(-1, img.shape[-1]).mean(axis=0)
    out_img = np.empty_like(img)
    out_img[:, :, :] = mean_color.astype(img.dtype)
    out_img[y_min:y_max + 1, x_min:x_max + 1, :] = img[y_min:y_max + 1, x_min:x_max + 1, :]
    return out_img, lbl.copy(), (x_min, y_min, x_max, y_max)


def make_cropB_bbox_zoom(img, lbl):
    h, w = lbl.shape
    x_min, y_min, x_max, y_max = get_bbox(lbl)
    img_crop = img[y_min:y_max + 1, x_min:x_max + 1, :]
    lbl_crop = lbl[y_min:y_max + 1, x_min:x_max + 1]
    img_zoom = cv2.resize(img_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    lbl_zoom = cv2.resize(lbl_crop.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    return img_zoom, lbl_zoom, (x_min, y_min, x_max, y_max)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bad_csv', required=True)
    p.add_argument('--out_manifest', required=True)
    args = p.parse_args()

    src_img_dir = os.path.join(DATA_ROOT, SRC_IMAGE_SUBDIR)
    src_lbl_dir = os.path.join(DATA_ROOT, SRC_LABEL_SUBDIR)
    out_img_dir = os.path.join(DATA_ROOT, OUT_IMAGE_SUBDIR)
    out_lbl_dir = os.path.join(DATA_ROOT, OUT_LABEL_SUBDIR)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_manifest), exist_ok=True)

    bad_df = pd.read_csv(args.bad_csv, dtype={'image_id': str})
    print('Bad images to augment: {}'.format(len(bad_df)))

    manifest_rows = []
    for _, row in bad_df.iterrows():
        img_id = row['image_id']
        orig_dice = row['dice']
        img = np.load(os.path.join(src_img_dir, img_id + '.npy'))
        lbl = np.load(os.path.join(src_lbl_dir, img_id + '.npy')) > 0.5
        lbl = lbl.astype(np.float32)

        variants = {
            'hflip': make_hflip(img, lbl),
            'vflip': make_vflip(img, lbl),
            'rot90': make_rot90(img, lbl),
            'rot270': make_rot270(img, lbl),
        }
        cropA_img, cropA_lbl, _ = make_cropA_bbox_mask(img, lbl)
        cropB_img, cropB_lbl, _ = make_cropB_bbox_zoom(img, lbl)
        variants['cropA'] = (cropA_img, cropA_lbl)
        variants['cropB'] = (cropB_img, cropB_lbl)

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
    manifest_df.to_csv(args.out_manifest, index=False)
    print('Saved manifest: {} ({} rows, {} bad images x 6 variants)'.format(
        args.out_manifest, len(manifest_df), len(bad_df)))
    print('AUG_DONE')


if __name__ == '__main__':
    main()
