"""
Evaluates an arbitrary pilot checkpoint against ALL 2000 training-pool images
(generic version of eval_allin_final.py, parametrized checkpoint path), for
computing "became difficult" on the training population: images that scored
>=0.7 under the allin baseline (the same reference that originally identified the
23 bad training images, per_image_analysis_v2/bad_image_augmentation/
allin_final_per_image_dice.csv) but drop below 0.7 under a given fold's retrained
pilot checkpoint.
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
    p.add_argument('--model_name', default='SwinUnet')
    p.add_argument('--image_subdir', default='Image_clahe')
    p.add_argument('--out_csv', required=True)
    p.add_argument('--config_yml', default='Configs/multi_train_local.yml')
    args = p.parse_args()

    config = DotDict(yaml.load(open(args.config_yml), Loader=yaml.FullLoader))
    datas = Dataset_wrap_csv(k_fold='allin', use_old_split=True, img_size=config.data.img_size,
                              dataset_name='isic2017', split_ratio=config.data.split_ratio,
                              train_aug=False, data_folder=config.data.data_folder,
                              meta_csv_name='meta_isic2017_train2000.csv', fixed_test_csv_name=None,
                              image_subdir=args.image_subdir)
    train_ds = datas['train']
    print('Pool: {} images'.format(len(train_ds)))

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
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print('n =', len(df))
    print('Mean dice: {:.4f}'.format(df['dice'].mean()))
    print('Bad (dice<0.7): {} / {}'.format((df['dice'] < 0.7).sum(), len(df)))
    print('Saved:', args.out_csv)
    print('TRAINFULL_EVAL_DONE')


if __name__ == '__main__':
    main()
