'''
Per-image measures computed directly from Image/Label .npy files, independent of any model:
hair coverage (whole image, boundary band, interior core), global contrast/sharpness,
boundary gradient strength, lesion size fraction, GT connected-component count.

Covers all 2600 official-split images (2000 train + 600 test) exactly once.
'''
import os
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import label as cc_label, binary_dilation, binary_erosion

ROOT = 'C:/Users/quanp/Downloads/ISIC 2017'
DATA_PATH = ROOT + '/data/isic2017/'
OUT_DIR = ROOT + '/per_image_analysis_v2'
IMG_SIZE = 224
BAND_WIDTH = 5   # pixels of dilation/erosion for boundary band
CORE_EROSION = 12  # pixels of erosion to define lesion "interior core"

os.makedirs(OUT_DIR, exist_ok=True)


def hair_mask(gray):
    # Standard DullRazor-style hair detection: blackhat highlights thin dark
    # structures (hair) against lighter skin background, then threshold.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return mask > 0


def measure_one(image_id):
    img_raw = np.load(os.path.join(DATA_PATH, 'Image', image_id + '.npy')).astype('uint8')
    lbl_raw = (np.load(os.path.join(DATA_PATH, 'Label', image_id + '.npy')) > 0.5)

    img = cv2.resize(img_raw, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(lbl_raw.astype('uint8'), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST) > 0

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img

    hmask = hair_mask(gray)
    hair_coverage_frac = hmask.mean()

    # Boundary band: dilate minus erode the GT mask
    dil = binary_dilation(gt, iterations=BAND_WIDTH)
    ero = binary_erosion(gt, iterations=BAND_WIDTH) if gt.sum() > 0 else np.zeros_like(gt)
    boundary_band = dil & ~ero
    core = binary_erosion(gt, iterations=CORE_EROSION) if gt.sum() > 0 else np.zeros_like(gt)

    if boundary_band.sum() > 0:
        hair_coverage_boundary_frac = hmask[boundary_band].mean()
    else:
        hair_coverage_boundary_frac = np.nan
    if core.sum() > 0:
        hair_coverage_interior_frac = hmask[core].mean()
    else:
        hair_coverage_interior_frac = np.nan

    # Global contrast/sharpness: Laplacian variance (low = blurry/low-contrast)
    contrast_laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Boundary gradient strength: mean Sobel gradient magnitude within the boundary band
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    boundary_gradient_mean = grad_mag[boundary_band].mean() if boundary_band.sum() > 0 else np.nan

    lesion_size_frac = gt.mean()
    _, gt_n = cc_label(gt)

    return {
        'image_id': image_id,
        'hair_coverage_frac': hair_coverage_frac,
        'hair_coverage_boundary_frac': hair_coverage_boundary_frac,
        'hair_coverage_interior_frac': hair_coverage_interior_frac,
        'contrast_laplacian_var': contrast_laplacian_var,
        'boundary_gradient_mean': boundary_gradient_mean,
        'lesion_size_frac': lesion_size_frac,
        'gt_component_count': gt_n,
    }


if __name__ == '__main__':
    train_df = pd.read_csv(DATA_PATH + 'meta_isic2017_train2000.csv', dtype={'ID': str})
    test_df = pd.read_csv(DATA_PATH + 'meta_isic2017_test600.csv', dtype={'ID': str})
    all_ids = list(train_df['ID']) + list(test_df['ID'])
    assert len(all_ids) == 2600
    assert len(set(all_ids)) == 2600

    records = []
    for i, image_id in enumerate(all_ids):
        records.append(measure_one(image_id))
        if (i + 1) % 200 == 0:
            print('  {}/{} measured'.format(i + 1, len(all_ids)), flush=True)

    df = pd.DataFrame(records)
    out_path = os.path.join(OUT_DIR, 'image_measures.csv')
    df.to_csv(out_path, index=False)
    print('Saved {} ({} rows)'.format(out_path, len(df)))
