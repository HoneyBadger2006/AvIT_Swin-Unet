"""
Builds the pilot fold-0 dataset for the SwinUnet + CLAHE (no FTL) hard-example
augmentation pilot:
  1. Copies the 198 augmented .npy image/label pairs (33 bad images x 6 variants)
     from the *_aug staging subdirs into the canonical Image_clahe/ and Label/
     directories, so the existing SkinDataset_csv pipeline (which hardcodes the
     Label subdir and takes image_subdir as a single config value shared by every
     row) can load them with zero code changes.
  2. Writes train_meta_kfold_meta_isic2017_train2000_pilot0badaug_0.csv = the
     original 1600-row fold-0 training split + the 198 new augmented rows.
  3. Copies test_meta_kfold_meta_isic2017_train2000_0.csv (fold 0's held-out val
     split, 400 images) UNCHANGED to the matching _pilot0badaug_0 filename, so
     Dataset_wrap_csv's fold-rotation lookup succeeds without altering val/test.

This makes meta_csv_name='meta_isic2017_train2000_pilot0badaug.csv' resolve to a
train set that includes the augmentations and a val set identical to the existing
CLAHE-only baseline's fold 0, for a clean apples-to-apples pilot comparison. No
existing data files are modified.
"""
import os
import shutil
import pandas as pd

DATA_ROOT = '../data/isic2017'
MANIFEST_CSV = '../per_image_analysis_v2/bad_image_augmentation/augmentation_manifest.csv'

SRC_IMG_AUG = os.path.join(DATA_ROOT, 'Image_clahe_aug')
SRC_LBL_AUG = os.path.join(DATA_ROOT, 'Label_aug')
DST_IMG = os.path.join(DATA_ROOT, 'Image_clahe')
DST_LBL = os.path.join(DATA_ROOT, 'Label')

ORIG_TRAIN_CSV = os.path.join(DATA_ROOT, 'train_meta_kfold_meta_isic2017_train2000_0.csv')
ORIG_VAL_CSV = os.path.join(DATA_ROOT, 'test_meta_kfold_meta_isic2017_train2000_0.csv')
NEW_TRAIN_CSV = os.path.join(DATA_ROOT, 'train_meta_kfold_meta_isic2017_train2000_pilot0badaug_0.csv')
NEW_VAL_CSV = os.path.join(DATA_ROOT, 'test_meta_kfold_meta_isic2017_train2000_pilot0badaug_0.csv')


def main():
    manifest = pd.read_csv(MANIFEST_CSV, dtype={'ID': str, 'source_id': str})
    print('Augmented rows in manifest: {}'.format(len(manifest)))

    copied = 0
    for new_id in manifest['ID']:
        for src_dir, dst_dir in [(SRC_IMG_AUG, DST_IMG), (SRC_LBL_AUG, DST_LBL)]:
            src = os.path.join(src_dir, new_id + '.npy')
            dst = os.path.join(dst_dir, new_id + '.npy')
            if not os.path.exists(dst):
                shutil.copyfile(src, dst)
                copied += 1
    print('Copied {} new .npy files into canonical Image_clahe/ and Label/'.format(copied))

    orig_train = pd.read_csv(ORIG_TRAIN_CSV, dtype={'ID': str})
    aug_rows = manifest[['ID', 'dataset', 'diagnosis', 'diagnosis_id']]
    new_train = pd.concat([orig_train, aug_rows], ignore_index=True)
    new_train.to_csv(NEW_TRAIN_CSV, index=False)
    print('Wrote {}: {} rows ({} original + {} augmented)'.format(
        NEW_TRAIN_CSV, len(new_train), len(orig_train), len(aug_rows)))

    shutil.copyfile(ORIG_VAL_CSV, NEW_VAL_CSV)
    val_df = pd.read_csv(NEW_VAL_CSV, dtype={'ID': str})
    print('Copied {} unchanged: {} rows'.format(NEW_VAL_CSV, len(val_df)))
    print('DONE')


if __name__ == '__main__':
    main()
