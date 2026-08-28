"""
Evaluates the all-2000-images ("allin", no k-fold) SwinUnet+CLAHE run's FINAL-epoch
checkpoint against the same 2000 images it was trained on (in-sample, as specified).
Reports the bad-image count (dice < 0.7) and the full dice distribution.
"""
import numpy as np
import pandas as pd
import torch
import yaml
import medpy.metric.binary as metrics

from Datasets.create_dataset import Dataset_wrap_csv
from Utils.pieces import DotDict
from tta_inference import build_model, forward_logits

CKPT = '../results/isic2017_swinunet_clahe_allin_SwinUnet_20260817_0324/final.pth'
OUT_CSV = '../per_image_analysis_v2/bad_image_augmentation/allin_final_per_image_dice.csv'


def main():
    config = DotDict(yaml.load(open('Configs/multi_train_local.yml'), Loader=yaml.FullLoader))
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
    model.load_state_dict(torch.load(CKPT))
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
                rows.append({'image_id': ids[i], 'dice': dice})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print('n =', len(df))
    print('Mean dice: {:.4f}'.format(df['dice'].mean()))

    bad = df[df['dice'] < 0.7]
    print('Bad images (dice < 0.7): {} / {} ({:.2f}%)'.format(len(bad), len(df), 100 * len(bad) / len(df)))

    bins = [(0.9, 1.0001), (0.7, 0.9), (0.5, 0.7), (0.3, 0.5), (0.0, 0.3)]
    print('\nFull distribution:')
    for lo, hi in bins:
        n = ((df['dice'] >= lo) & (df['dice'] < hi)).sum()
        print('  [{:.1f}, {:.1f}): {} ({:.2f}%)'.format(lo, hi, n, 100 * n / len(df)))

    print('\nSaved:', OUT_CSV)
    print('EVAL_DONE')


if __name__ == '__main__':
    main()
