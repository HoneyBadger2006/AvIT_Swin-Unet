"""
Builds the augmented training data for the "bad image" hard-example pilot.

Reads bad_images_fold0.csv (fold-0 training images where the existing SwinUnet
CLAHE+FTL fold-0 checkpoint scores dice < 0.7, in-sample), and for each one writes
new .npy image/label pairs under new subdirectories so they can be added to the
training CSV as ordinary additional rows and flow through the existing
SkinDataset_csv pipeline unmodified.

Augmentation types (per bad image):
  hflip / vflip / rot90 / rot270  - standard geometric augmentations, image+label
                                     transformed identically.
  cropA (bbox-mask)  - GT bounding box [x_min,y_min,x_max,y_max] computed on the
                        512x512 label; everything OUTSIDE the box is replaced with
                        the image's mean color (per-channel). Label is UNCHANGED
                        (same resolution, same mask, since the lesion pixels inside
                        the box are untouched).
  cropB (bbox-zoom)  - crop the image AND label to that same bounding box, then
                        resize both back up to 512x512 (zoom-in on the lesion).
                        Label is resized with nearest-neighbor to keep it binary.

New IDs are '{orig_id}_{suffix}' (suffix in hflip/vflip/rot90/rot270/cropA/cropB).
Saved under Image_clahe_aug/ and Label_aug/ (mirroring the Image_clahe/Label
layout the model already expects), plus a manifest CSV listing every new row.
"""
import os
import numpy as np
import pandas as pd
import cv2

DATA_ROOT = '../data/isic2017'
SRC_IMAGE_SUBDIR = 'Image_clahe'
SRC_LABEL_SUBDIR = 'Label'
OUT_IMAGE_SUBDIR = 'Image_clahe_aug'
OUT_LABEL_SUBDIR = 'Label_aug'
BAD_IMAGES_CSV = '../per_image_analysis_v2/bad_image_augmentation/bad_images_fold0_clahe_only.csv'
OUT_DIR = '../per_image_analysis_v2/bad_image_augmentation'


def get_bbox(label):
    """label: (H,W) binary array. Returns the tight (x_min,y_min,x_max,y_max) inclusive box,
    exactly as computed from the GT mask -- no margin added."""
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
    """Fill everything outside the GT bbox with the image's average color. Label unchanged."""
    x_min, y_min, x_max, y_max = get_bbox(lbl)
    mean_color = img.reshape(-1, img.shape[-1]).mean(axis=0)
    out_img = np.empty_like(img)
    out_img[:, :, :] = mean_color.astype(img.dtype)
    out_img[y_min:y_max + 1, x_min:x_max + 1, :] = img[y_min:y_max + 1, x_min:x_max + 1, :]
    return out_img, lbl.copy(), (x_min, y_min, x_max, y_max)


def make_cropB_bbox_zoom(img, lbl):
    """Crop to the GT bbox, then resize crop back up to the original image size (zoom-in)."""
    h, w = lbl.shape
    x_min, y_min, x_max, y_max = get_bbox(lbl)
    img_crop = img[y_min:y_max + 1, x_min:x_max + 1, :]
    lbl_crop = lbl[y_min:y_max + 1, x_min:x_max + 1]
    img_zoom = cv2.resize(img_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    lbl_zoom = cv2.resize(lbl_crop.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    return img_zoom, lbl_zoom, (x_min, y_min, x_max, y_max)


def main():
    src_img_dir = os.path.join(DATA_ROOT, SRC_IMAGE_SUBDIR)
    src_lbl_dir = os.path.join(DATA_ROOT, SRC_LABEL_SUBDIR)
    out_img_dir = os.path.join(DATA_ROOT, OUT_IMAGE_SUBDIR)
    out_lbl_dir = os.path.join(DATA_ROOT, OUT_LABEL_SUBDIR)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    bad_df = pd.read_csv(BAD_IMAGES_CSV, dtype={'image_id': str})
    print('Bad images to augment: {}'.format(len(bad_df)))

    manifest_rows = []
    bbox_rows = []

    for _, row in bad_df.iterrows():
        img_id = row['image_id']
        orig_dice = row['dice']
        img = np.load(os.path.join(src_img_dir, img_id + '.npy'))
        lbl = np.load(os.path.join(src_lbl_dir, img_id + '.npy')) > 0.5
        lbl = lbl.astype(np.float32)

        variants = {}
        variants['hflip'] = make_hflip(img, lbl)
        variants['vflip'] = make_vflip(img, lbl)
        variants['rot90'] = make_rot90(img, lbl)
        variants['rot270'] = make_rot270(img, lbl)

        cropA_img, cropA_lbl, bboxA = make_cropA_bbox_mask(img, lbl)
        cropB_img, cropB_lbl, bboxB = make_cropB_bbox_zoom(img, lbl)
        variants['cropA'] = (cropA_img, cropA_lbl)
        variants['cropB'] = (cropB_img, cropB_lbl)

        for suffix, (v_img, v_lbl) in variants.items():
            new_id = '{}_{}'.format(img_id, suffix)
            np.save(os.path.join(out_img_dir, new_id + '.npy'), v_img.astype(np.uint8))
            np.save(os.path.join(out_lbl_dir, new_id + '.npy'), v_lbl.astype(np.float32))
            manifest_rows.append({
                'ID': new_id, 'source_id': img_id, 'augmentation': suffix,
                'source_dice': orig_dice, 'dataset': 'isic2017', 'diagnosis': 'unknown', 'diagnosis_id': 0,
            })

        bbox_rows.append({'image_id': img_id, 'x_min': bboxA[0], 'y_min': bboxA[1],
                           'x_max': bboxA[2], 'y_max': bboxA[3], 'source_dice': orig_dice})

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv = os.path.join(OUT_DIR, 'augmentation_manifest.csv')
    manifest_df.to_csv(manifest_csv, index=False)
    print('Saved manifest: {} ({} new image/label pairs, {} bad images x 6 variants)'.format(
        manifest_csv, len(manifest_df), len(bad_df)))

    bbox_df = pd.DataFrame(bbox_rows)
    bbox_csv = os.path.join(OUT_DIR, 'bbox_coords.csv')
    bbox_df.to_csv(bbox_csv, index=False)
    print('Saved bbox coords: {}'.format(bbox_csv))
    print('New image/label pairs written to {} / {}'.format(out_img_dir, out_lbl_dir))
    print('DONE')


if __name__ == '__main__':
    main()
