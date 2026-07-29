"""
Build meta_isic2017_train2000.csv and meta_isic2017_test600.csv from the existing
pooled meta_isic2017.csv, using the raw ISIC 2017 ground-truth folders as the
authoritative source of which official split each image ID belongs to.

The raw folders (isic2017_raw/ISIC-2017_Training_Part1_GroundTruth and
isic2017_raw/ISIC-2017_Test_v2_Part1_GroundTruth) are disjoint sets of IDs whose
union exactly matches the 2600 processed images in data/isic2017/Image, so no
reprocessing of images is needed -- this just splits the existing meta CSV.
"""
import os
import re
import pandas as pd

ROOT = 'C:/Users/quanp/Downloads/ISIC 2017'
RAW_DIR = os.path.join(ROOT, 'isic2017_raw')
DATA_DIR = os.path.join(ROOT, 'AViT', 'data', 'isic2017') if os.path.exists(
    os.path.join(ROOT, 'AViT', 'data', 'isic2017')) else os.path.join(ROOT, 'data', 'isic2017')

TRAIN_GT_DIR = os.path.join(RAW_DIR, 'ISIC-2017_Training_Part1_GroundTruth')
TEST_GT_DIR = os.path.join(RAW_DIR, 'ISIC-2017_Test_v2_Part1_GroundTruth')

ID_RE = re.compile(r'ISIC_(\d+)_segmentation\.png')


def ids_from_gt_dir(gt_dir):
    ids = set()
    for fname in os.listdir(gt_dir):
        m = ID_RE.match(fname)
        if m:
            ids.add(m.group(1))
    return ids


def main():
    train_ids = ids_from_gt_dir(TRAIN_GT_DIR)
    test_ids = ids_from_gt_dir(TEST_GT_DIR)

    assert len(train_ids) == 2000, 'expected 2000 official training IDs, got {}'.format(len(train_ids))
    assert len(test_ids) == 600, 'expected 600 official test IDs, got {}'.format(len(test_ids))
    assert train_ids.isdisjoint(test_ids), 'official train/test ID sets overlap, aborting'

    meta_path = os.path.join(DATA_DIR, 'meta_isic2017.csv')
    df = pd.read_csv(meta_path, dtype={'ID': str})

    processed_ids = set(df['ID'])
    union = train_ids | test_ids
    assert processed_ids == union, (
        'processed meta IDs do not exactly match the raw train+test union; '
        'missing from union: {}, extra in union: {}'.format(
            processed_ids - union, union - processed_ids))

    train_df = df[df['ID'].isin(train_ids)].reset_index(drop=True)
    test_df = df[df['ID'].isin(test_ids)].reset_index(drop=True)

    assert len(train_df) == 2000
    assert len(test_df) == 600
    assert set(train_df['ID']).isdisjoint(set(test_df['ID']))

    train_out = os.path.join(DATA_DIR, 'meta_isic2017_train2000.csv')
    test_out = os.path.join(DATA_DIR, 'meta_isic2017_test600.csv')
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    print('Wrote {} ({} rows)'.format(train_out, len(train_df)))
    print('Wrote {} ({} rows)'.format(test_out, len(test_df)))


if __name__ == '__main__':
    main()
