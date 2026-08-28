"""
Generic version of find_bad_images.py: identifies bad images (dice < 0.7) for ANY
network/fold/checkpoint, on either that fold's own training split (--split train,
in-sample -- used by the overnight orchestrator to build each fold's augmentation
target set) or that fold's held-out validation slice (--split val, the ~20% of
meta_isic2017_train2000.csv NOT used to train that fold -- genuinely out-of-sample,
distinct from the fixed test600 set).
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

DICE_THRESHOLD = 0.7


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--model_name', required=True)
    p.add_argument('--fold', type=int, required=True)
    p.add_argument('--image_subdir', default='Image_clahe')
    p.add_argument('--split', choices=['train', 'val'], default='train',
                    help="'train' = in-sample (fold's own training split); "
                         "'val' = fold's held-out ~20%% slice, genuinely out-of-sample")
    p.add_argument('--out_csv', required=True)
    p.add_argument('--config_yml', default='Configs/multi_train_local.yml')
    args = p.parse_args()

    config = DotDict(yaml.load(open(args.config_yml), Loader=yaml.FullLoader))
    datas = Dataset_wrap_csv(k_fold=str(args.fold), use_old_split=True, img_size=config.data.img_size,
                              dataset_name='isic2017', split_ratio=config.data.split_ratio,
                              train_aug=False, data_folder=config.data.data_folder,
                              meta_csv_name='meta_isic2017_train2000.csv', fixed_test_csv_name=None,
                              image_subdir=args.image_subdir)
    train_ds = datas[args.split]
    print('Fold-{} {} split: {} images'.format(args.fold, args.split, len(train_ds)))

    loader = torch.utils.data.DataLoader(train_ds, batch_size=config.test.batch_size, shuffle=False,
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
                rows.append({'image_id': ids[i], 'dice': dice})

    df = pd.DataFrame(rows)
    print('Mean {} dice: {:.4f}'.format(args.split, df['dice'].mean()))
    bad = df[df['dice'] < DICE_THRESHOLD].sort_values('dice').reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    bad.to_csv(args.out_csv, index=False)
    print('Bad images (dice < {}): {} / {}'.format(DICE_THRESHOLD, len(bad), len(df)))
    print('Saved: {}'.format(args.out_csv))
    print('BADIMG_DONE')


if __name__ == '__main__':
    main()
