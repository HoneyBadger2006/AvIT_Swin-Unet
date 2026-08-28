"""
Identifies "bad images" for the hard-example augmentation pilot: images in fold 0's
OWN training split (train_meta_kfold_meta_isic2017_train2000_0.csv, 1600 images, the
same 1600 images the SwinUnet CLAHE+FTL fold-0 checkpoint was actually trained on)
where that checkpoint's per-image Dice is < 0.7.

This is an in-sample check (the checkpoint saw these images during training), which is
intentional: the goal is to find images the current best config still can't fit well
even in-sample -- a standard hard-example-mining signal -- not to estimate held-out
generalization (that's what the existing test600 per-image results are for).

Pilot config = SwinUnet + CLAHE preprocessing + compound FTL loss, fold 0
(../results/isic2017_swinunet_clahe_ftl_k0_SwinUnet_20260730_0001/best.pth), matching
the per-image-verified stage-3/4 winner for SwinUnet from per_image_final_pipeline.py.
"""
import os
import pandas as pd
import torch
import yaml
import medpy.metric.binary as metrics

from Datasets.create_dataset import Dataset_wrap_csv
from Utils.pieces import DotDict
from tta_inference import build_model, forward_logits

CKPT = '../results/isic2017_swinunet_clahe_k0_SwinUnet_20260728_0301/best.pth'
MODEL_NAME = 'SwinUnet'
OUT_DIR = '../per_image_analysis_v2/bad_image_augmentation'
DICE_THRESHOLD = 0.7


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    config = DotDict(yaml.load(open('Configs/multi_train_local.yml'), Loader=yaml.FullLoader))

    datas = Dataset_wrap_csv(k_fold='0', use_old_split=True, img_size=config.data.img_size,
                              dataset_name='isic2017', split_ratio=config.data.split_ratio,
                              train_aug=False, data_folder=config.data.data_folder,
                              meta_csv_name='meta_isic2017_train2000.csv', fixed_test_csv_name=None,
                              image_subdir='Image_clahe')
    train_ds = datas['train']
    print('Fold-0 training split: {} images'.format(len(train_ds)))

    loader = torch.utils.data.DataLoader(train_ds, batch_size=config.test.batch_size, shuffle=False,
                                          num_workers=config.test.num_workers, pin_memory=True, drop_last=False)

    model = build_model(MODEL_NAME, config)
    model.load_state_dict(torch.load(CKPT))
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in loader:
            img = batch['image'].cuda().float()
            label = batch['label'].cuda().float()
            ids = batch['ID']

            logits = forward_logits(model, MODEL_NAME, img)
            prob = torch.sigmoid(logits)
            pred = (prob.cpu().numpy() > 0.5)
            label_np = label.cpu().numpy() > 0.5

            for i in range(img.shape[0]):
                dice = metrics.dc(pred[i, 0], label_np[i, 0])
                rows.append({'image_id': ids[i], 'dice': dice})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, 'fold0_train_per_image_dice_clahe_only.csv')
    df.to_csv(csv_path, index=False)
    print('Saved: {} ({} rows)'.format(csv_path, len(df)))
    print('Mean dice: {:.4f}'.format(df['dice'].mean()))

    bad = df[df['dice'] < DICE_THRESHOLD].sort_values('dice').reset_index(drop=True)
    bad_csv = os.path.join(OUT_DIR, 'bad_images_fold0_clahe_only.csv')
    bad.to_csv(bad_csv, index=False)
    print('Bad images (dice < {}): {} / {} ({:.1f}%)'.format(DICE_THRESHOLD, len(bad), len(df), 100 * len(bad) / len(df)))
    print('Saved: {}'.format(bad_csv))
    print(bad.to_string(index=False))
    print('DONE')


if __name__ == '__main__':
    main()
