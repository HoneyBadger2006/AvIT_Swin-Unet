"""
Computes the actual per-channel mean/std for a color-space subdir, over the full
2000-image training set, on the SAME [0,1]-rescaled representation the model
actually sees (raw pixel value / 255, matching norm01() in create_dataset.py,
applied BEFORE transforms.Normalize) -- not raw 0-255 units. This matters
especially for Hue, whose OpenCV range is [0,179] not [0,255], so its /255-scaled
mean/std are naturally smaller than S or V's.
"""
import argparse
import numpy as np
import pandas as pd

DATA_ROOT = '../data/isic2017'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image_subdir', required=True, help="e.g. 'Image_hsv'")
    args = p.parse_args()

    meta = pd.read_csv('{}/meta_isic2017_train2000.csv'.format(DATA_ROOT), dtype={'ID': str})
    print('Computing stats over {} training images from {}/'.format(len(meta), args.image_subdir))

    sum_c = np.zeros(3, dtype=np.float64)
    sumsq_c = np.zeros(3, dtype=np.float64)
    n_pixels = 0

    for image_id in meta['ID']:
        img = np.load('{}/{}/{}.npy'.format(DATA_ROOT, args.image_subdir, image_id)).astype(np.float64)
        img01 = np.clip(img, 0, 255) / 255.0
        sum_c += img01.reshape(-1, 3).sum(axis=0)
        sumsq_c += (img01.reshape(-1, 3) ** 2).sum(axis=0)
        n_pixels += img01.shape[0] * img01.shape[1]

    mean = sum_c / n_pixels
    var = sumsq_c / n_pixels - mean ** 2
    std = np.sqrt(np.clip(var, 0, None))

    print('\nPer-channel stats (on [0,1]-rescaled data, matching model input):')
    for i, ch in enumerate(['ch0', 'ch1', 'ch2']):
        print('  {}: mean={:.4f}  std={:.4f}'.format(ch, mean[i], std[i]))

    print('\nSanity check: all means should be in [0,1], all stds should be > 0 and < ~0.4')
    assert np.all(mean >= 0) and np.all(mean <= 1), 'mean out of [0,1] range: {}'.format(mean)
    assert np.all(std > 0) and np.all(std < 0.5), 'std out of sane range: {}'.format(std)
    print('Sanity check PASSED')

    print('\nFor CLI use:')
    print('  --norm_mean {:.4f} {:.4f} {:.4f}'.format(*mean))
    print('  --norm_std {:.4f} {:.4f} {:.4f}'.format(*std))
    print('NORM_STATS_DONE')


if __name__ == '__main__':
    main()
