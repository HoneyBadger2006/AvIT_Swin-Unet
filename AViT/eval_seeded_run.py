"""
Evaluates a seeded no-CV allin checkpoint against all 2000 training images,
reporting both the Dice<0.7 and IoU<0.7 bad-image counts. IoU is derived from Dice
via the exact algebraic identity IoU = Dice/(2-Dice) (already verified, in
eval_iou_vs_dice_check.py, to match medpy.metric.binary.jc() bit-for-bit for every
image), so a single Dice pass gives both counts with no extra computation.
"""
import argparse
import os
import pandas as pd
import torch
import yaml
import medpy.metric.binary as metrics

from Datasets.create_dataset import Dataset_wrap_csv
from Utils.pieces import DotDict
from tta_inference import build_model, forward_logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--seed_label', required=True, help="e.g. 'seed42'")
    p.add_argument('--out_csv', required=True)
    p.add_argument('--config_yml', default='Configs/multi_train_local.yml')
    args = p.parse_args()

    config = DotDict(yaml.load(open(args.config_yml), Loader=yaml.FullLoader))
    datas = Dataset_wrap_csv(k_fold='allin', use_old_split=True, img_size=config.data.img_size,
                              dataset_name='isic2017', split_ratio=config.data.split_ratio,
                              train_aug=False, data_folder=config.data.data_folder,
                              meta_csv_name='meta_isic2017_train2000.csv', fixed_test_csv_name=None,
                              image_subdir='Image_clahe')
    train_ds = datas['train']
    print('Pool: {} images'.format(len(train_ds)))

    loader = torch.utils.data.DataLoader(train_ds, batch_size=config.test.batch_size, shuffle=False,
                                          num_workers=config.test.num_workers, pin_memory=True, drop_last=False)

    model = build_model('SwinUnet', config)
    model.load_state_dict(torch.load(args.ckpt))
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in loader:
            img = batch['image'].cuda().float()
            label = batch['label'].cuda().float()
            ids = batch['ID']
            logits = forward_logits(model, 'SwinUnet', img)
            prob = torch.sigmoid(logits)
            pred = (prob.cpu().numpy() > 0.5)
            label_np = label.cpu().numpy() > 0.5
            for i in range(img.shape[0]):
                dice = metrics.dc(pred[i, 0], label_np[i, 0])
                iou = dice / (2 - dice) if dice < 2 else 1.0
                rows.append({'image_id': ids[i], 'dice': dice, 'iou': iou})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    n_dice_bad = (df['dice'] < 0.7).sum()
    n_iou_bad = (df['iou'] < 0.7).sum()
    print('n =', len(df))
    print('Mean dice: {:.4f}  Mean iou: {:.4f}'.format(df['dice'].mean(), df['iou'].mean()))
    print('[{}] Dice<0.7 count: {} / {}'.format(args.seed_label, n_dice_bad, len(df)))
    print('[{}] IoU<0.7 count:  {} / {}'.format(args.seed_label, n_iou_bad, len(df)))
    print('Saved:', args.out_csv)
    print('SEEDED_EVAL_DONE')


if __name__ == '__main__':
    main()
