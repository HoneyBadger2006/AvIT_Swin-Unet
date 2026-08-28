"""
Generic version of build_pilot_csv.py: given a fold and an augmentation manifest,
copies the augmented .npy pairs into the canonical Image_clahe/ and Label/
directories (generate-if-missing) and writes the expanded train CSV + an unchanged
copy of the val CSV under a shared meta_csv_name tag, so Dataset_wrap_csv's
fold-rotation lookup resolves cleanly for any fold under that tag.
"""
import argparse
import os
import shutil
import pandas as pd

DATA_ROOT = '../data/isic2017'
SRC_IMG_AUG = os.path.join(DATA_ROOT, 'Image_clahe_aug')
SRC_LBL_AUG = os.path.join(DATA_ROOT, 'Label_aug')
DST_IMG = os.path.join(DATA_ROOT, 'Image_clahe')
DST_LBL = os.path.join(DATA_ROOT, 'Label')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fold', type=str, required=True)
    p.add_argument('--manifest_csv', required=True)
    p.add_argument('--meta_tag', required=True,
                    help="e.g. 'swinunet_badaug' -> writes train_meta_kfold_meta_isic2017_train2000_swinunet_badaug_{fold}.csv")
    args = p.parse_args()

    manifest = pd.read_csv(args.manifest_csv, dtype={'ID': str, 'source_id': str})
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

    orig_train_csv = os.path.join(DATA_ROOT, 'train_meta_kfold_meta_isic2017_train2000_{}.csv'.format(args.fold))
    orig_val_csv = os.path.join(DATA_ROOT, 'test_meta_kfold_meta_isic2017_train2000_{}.csv'.format(args.fold))
    new_train_csv = os.path.join(DATA_ROOT, 'train_meta_kfold_meta_isic2017_train2000_{}_{}.csv'.format(args.meta_tag, args.fold))
    new_val_csv = os.path.join(DATA_ROOT, 'test_meta_kfold_meta_isic2017_train2000_{}_{}.csv'.format(args.meta_tag, args.fold))

    orig_train = pd.read_csv(orig_train_csv, dtype={'ID': str})
    aug_rows = manifest[['ID', 'dataset', 'diagnosis', 'diagnosis_id']]
    new_train = pd.concat([orig_train, aug_rows], ignore_index=True)
    new_train.to_csv(new_train_csv, index=False)
    print('Wrote {}: {} rows ({} original + {} augmented)'.format(
        new_train_csv, len(new_train), len(orig_train), len(aug_rows)))

    shutil.copyfile(orig_val_csv, new_val_csv)
    print('Copied {} unchanged'.format(new_val_csv))
    print('CSV_DONE')


if __name__ == '__main__':
    main()
