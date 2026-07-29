'''
Precompute DullRazor-processed versions of the images needed, saving to
data/isic2017/Image_dullrazor/ (same ID, same .npy format as Image/, so the
existing SkinDataset_csv loading code works unchanged aside from swapping
the image subdirectory). Labels are untouched -- DullRazor only modifies
the RGB image.

By default processes the union of image IDs found across a list of meta CSVs
(so re-running with a bigger ID list only computes the delta).
'''
import os
import sys
import numpy as np
import pandas as pd
from dullrazor import dullrazor

ROOT = 'C:/Users/quanp/Downloads/ISIC 2017'
DATA_PATH = ROOT + '/data/isic2017/'
SRC_DIR = DATA_PATH + 'Image/'
DST_DIR = DATA_PATH + 'Image_dullrazor/'

os.makedirs(DST_DIR, exist_ok=True)


def precompute(meta_csvs):
    ids = set()
    for csv_name in meta_csvs:
        df = pd.read_csv(DATA_PATH + csv_name, dtype={'ID': str})
        ids |= set(df['ID'])
    ids = sorted(ids)
    print('{} unique image IDs across {}'.format(len(ids), meta_csvs))

    done, skipped = 0, 0
    for i, image_id in enumerate(ids):
        dst_path = DST_DIR + image_id + '.npy'
        if os.path.exists(dst_path):
            skipped += 1
            continue
        img = np.load(SRC_DIR + image_id + '.npy').astype('uint8')
        out = dullrazor(img)
        np.save(dst_path, out)
        done += 1
        if (i + 1) % 200 == 0:
            print('  {}/{} (processed={}, skipped_existing={})'.format(i + 1, len(ids), done, skipped), flush=True)

    print('DONE. processed={}, skipped_existing={}, total={}'.format(done, skipped, len(ids)))


if __name__ == '__main__':
    csvs = sys.argv[1:] if len(sys.argv) > 1 else [
        'train_meta_kfold_meta_isic2017_train2000_0.csv',
        'meta_isic2017_test600.csv',
    ]
    precompute(csvs)
