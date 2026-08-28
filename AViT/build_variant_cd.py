"""
Builds two new crop variants for a given bad-images CSV, on top of the existing
tight-bbox machinery (get_bbox = exact GT bbox, no margin):

  Variant C ("expanded bbox"): pad the tight GT bbox by 20% of its own width/height
    in each direction (clipped to image bounds), then crop+resize that expanded
    region to the full frame (same crop->zoom mechanics as Variant B), giving the
    model some visual context around the lesion instead of a razor-tight crop.

  Variant D ("70%-fill zoom"): crop a region centered on the GT bbox, sized so that
    after resizing to the full frame the lesion's bbox occupies exactly 70% of the
    frame (crop_size = bbox_size / 0.7), then resize to the frame -- a controlled
    zoom level between the untouched original (lesion at its natural, usually much
    smaller, size) and Variant B's 100%-fill zoom (lesion touches the crop edges).

Both variants only need the GT mask, so -- like build_augmentations_generic.py --
they are fold/network-independent and use generate-if-missing semantics.
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


def crop_and_zoom(img, lbl, x_min, y_min, x_max, y_max):
    h, w = lbl.shape
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w - 1, x_max)
    y_max = min(h - 1, y_max)
    img_crop = img[y_min:y_max + 1, x_min:x_max + 1, :]
    lbl_crop = lbl[y_min:y_max + 1, x_min:x_max + 1]
    img_zoom = cv2.resize(img_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    lbl_zoom = cv2.resize(lbl_crop.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    return img_zoom, lbl_zoom, (x_min, y_min, x_max, y_max)


def make_variantC_expand20(img, lbl):
    x_min, y_min, x_max, y_max = get_bbox(lbl)
    bw, bh = x_max - x_min + 1, y_max - y_min + 1
    pad_x, pad_y = int(round(0.2 * bw)), int(round(0.2 * bh))
    ex_x_min, ex_y_min = x_min - pad_x, y_min - pad_y
    ex_x_max, ex_y_max = x_max + pad_x, y_max + pad_y
    return crop_and_zoom(img, lbl, ex_x_min, ex_y_min, ex_x_max, ex_y_max)


def make_variantD_fill70(img, lbl):
    x_min, y_min, x_max, y_max = get_bbox(lbl)
    bw, bh = x_max - x_min + 1, y_max - y_min + 1
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    crop_w, crop_h = bw / 0.7, bh / 0.7
    d_x_min = int(round(cx - crop_w / 2.0))
    d_x_max = int(round(cx + crop_w / 2.0))
    d_y_min = int(round(cy - crop_h / 2.0))
    d_y_max = int(round(cy + crop_h / 2.0))
    return crop_and_zoom(img, lbl, d_x_min, d_y_min, d_x_max, d_y_max)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bad_csv', required=True)
    p.add_argument('--out_manifest', required=True)
    p.add_argument('--out_bbox_csv', required=True)
    args = p.parse_args()

    src_img_dir = os.path.join(DATA_ROOT, SRC_IMAGE_SUBDIR)
    src_lbl_dir = os.path.join(DATA_ROOT, SRC_LABEL_SUBDIR)
    out_img_dir = os.path.join(DATA_ROOT, OUT_IMAGE_SUBDIR)
    out_lbl_dir = os.path.join(DATA_ROOT, OUT_LABEL_SUBDIR)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_manifest), exist_ok=True)

    bad_df = pd.read_csv(args.bad_csv, dtype={'image_id': str})
    print('Bad images to build variants C/D for: {}'.format(len(bad_df)))

    manifest_rows = []
    bbox_rows = []
    for _, row in bad_df.iterrows():
        img_id = row['image_id']
        orig_dice = row['dice'] if 'dice' in row else row.get('orig_allin_dice')
        img = np.load(os.path.join(src_img_dir, img_id + '.npy'))
        lbl = np.load(os.path.join(src_lbl_dir, img_id + '.npy')) > 0.5
        lbl = lbl.astype(np.float32)

        cropC_img, cropC_lbl, bboxC = make_variantC_expand20(img, lbl)
        cropD_img, cropD_lbl, bboxD = make_variantD_fill70(img, lbl)
        variants = {'cropC': (cropC_img, cropC_lbl, bboxC), 'cropD': (cropD_img, cropD_lbl, bboxD)}

        for suffix, (v_img, v_lbl, bbox) in variants.items():
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
            bbox_rows.append({'image_id': img_id, 'variant': suffix,
                               'x_min': bbox[0], 'y_min': bbox[1], 'x_max': bbox[2], 'y_max': bbox[3]})

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(args.out_manifest, index=False)
    bbox_df = pd.DataFrame(bbox_rows)
    bbox_df.to_csv(args.out_bbox_csv, index=False)
    print('Saved manifest: {} ({} rows, {} images x 2 variants)'.format(
        args.out_manifest, len(manifest_df), len(bad_df)))
    print('Saved bbox coords: {}'.format(args.out_bbox_csv))
    print('VARIANT_CD_DONE')


if __name__ == '__main__':
    main()
