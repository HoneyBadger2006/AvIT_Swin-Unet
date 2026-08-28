"""
Evaluates a retrained checkpoint against the ORIGINAL (non-augmented) images that
were flagged bad (dice < 0.7, seed42) -- the training-side recovery metric.
Uses SkinDataset_csv directly on the filtered DataFrame, same pattern as
eval_train23_recovery.py, for byte-for-byte-identical preprocessing. Generic over
which bad-image list / model, so it covers SwinUnet's 37, AViT's 189, etc.
"""
import argparse
import os
import pandas as pd
import torch
import yaml
import medpy.metric.binary as metrics

from Datasets.create_dataset import SkinDataset_csv
from Utils.pieces import DotDict
from tta_inference import build_model, forward_logits

DATA_ROOT = '../data/isic2017/'
DEFAULT_BAD_CSV = '../per_image_analysis_v2/bad_image_augmentation/bad_images_seed42_dice37.csv'
META_CSV = '../data/isic2017/meta_isic2017_train2000.csv'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--model_name', default='SwinUnet')
    p.add_argument('--image_subdir', default='Image_clahe')
    p.add_argument('--bad_csv', default=DEFAULT_BAD_CSV)
    p.add_argument('--expected_n', type=int, default=37)
    p.add_argument('--out_csv', required=True)
    p.add_argument('--config_yml', default='Configs/multi_train_local.yml')
    args = p.parse_args()

    config = DotDict(yaml.load(open(args.config_yml), Loader=yaml.FullLoader))
    bad37 = pd.read_csv(args.bad_csv, dtype={'image_id': str})
    print('Original bad training images to re-check: {}'.format(len(bad37)))
    assert len(bad37) == args.expected_n

    meta = pd.read_csv(META_CSV, dtype={'ID': str})
    sub_df = meta[meta['ID'].isin(bad37['image_id'])].reset_index(drop=True)
    assert len(sub_df) == args.expected_n, 'expected {} matched rows in meta csv, got {}'.format(args.expected_n, len(sub_df))

    ds = SkinDataset_csv('isic2017', config.data.img_size, sub_df, use_aug=False,
                          data_path=DATA_ROOT, image_subdir=args.image_subdir)
    loader = torch.utils.data.DataLoader(ds, batch_size=config.test.batch_size, shuffle=False,
                                          num_workers=config.test.num_workers, pin_memory=True, drop_last=False)

    model = build_model(args.model_name, config)
    model.load_state_dict(torch.load(args.ckpt))
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in loader:
            img = batch['image'].cuda().float()
            label = batch['label'].cuda().float()
            ids = batch['ID']
            logits = forward_logits(model, args.model_name, img)
            prob = torch.sigmoid(logits)
            pred = (prob.cpu().numpy() > 0.5)
            label_np = label.cpu().numpy() > 0.5
            for i in range(img.shape[0]):
                dice = metrics.dc(pred[i, 0], label_np[i, 0])
                rows.append({'image_id': ids[i], 'retrain_dice': dice})

    df = pd.DataFrame(rows)
    df = df.merge(bad37.rename(columns={'dice': 'orig_dice'}), on='image_id')
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    still_bad = df[df['retrain_dice'] < 0.7]
    print('\nMean retrain dice on the {}: {:.4f} (was {:.4f})'.format(
        args.expected_n, df['retrain_dice'].mean(), df['orig_dice'].mean()))
    print('Still below 0.7 after retrain: {} / {}'.format(len(still_bad), len(df)))
    print('Saved: {}'.format(args.out_csv))
    print('TRAIN_RECOVERY_EVAL_DONE')


if __name__ == '__main__':
    main()
