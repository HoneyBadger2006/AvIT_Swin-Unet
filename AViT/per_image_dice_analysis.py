'''
Per-image Dice analysis on the ISIC-2017 fold-0 test set for Swin-UNet and AViT.
Loads both fold-0 checkpoints, runs each test image through both models individually,
computes per-image Dice (matching the training script's threshold/metric: medpy dc @ 0.5),
and saves worst-case comparisons.
'''
import os
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import medpy.metric.binary as metrics

from Datasets.create_dataset import SkinDataset_csv
from Models.Transformer.SwinUnet import SwinUnet
from Models.Transformer.Swin_adapters import SwinSimpleSeg_CNNprompt_adapt

DATA_PATH = 'C:/Users/quanp/Downloads/ISIC 2017/data/isic2017/'
PRETRAINED_FOLDER = 'C:/Users/quanp/Downloads/ISIC 2017/pretrained_ckpt'
SWINUNET_CKPT = 'C:/Users/quanp/Downloads/ISIC 2017/results/isic2017_swinunet_SwinUnet_20260713_0051/best.pth'
AVIT_CKPT = 'C:/Users/quanp/Downloads/ISIC 2017/results/isic2017_avit_SwinSeg_CNNprompt_adapt_20260713_0420/best.pth'
OUT_DIR = 'C:/Users/quanp/Downloads/ISIC 2017/per_image_analysis'
IMG_SIZE = 224

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_df = pd.read_csv(DATA_PATH + 'test_meta_kfold_0.csv', dtype={'ID': str})
test_ds = SkinDataset_csv('isic2017', IMG_SIZE, test_df, use_aug=False, data_path=DATA_PATH)
loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
print('Test set (fold 0): {} images'.format(len(test_ds)))

swinunet = SwinUnet(img_size=IMG_SIZE).to(device)
swinunet.load_state_dict(torch.load(SWINUNET_CKPT, map_location=device))
swinunet.eval()

avit = SwinSimpleSeg_CNNprompt_adapt(
    img_size=IMG_SIZE, pretrained=False, pretrained_swin_name='swin_large_patch4_window7_224_22k',
    pretrained_folder=PRETRAINED_FOLDER, embed_dim=192, drop_path_rate=0.2,
    depths=[2, 2, 18], num_heads=[6, 12, 24], window_size=7,
    debug=False, adapt_method=False, num_domains=1,
).to(device)
avit.load_state_dict(torch.load(AVIT_CKPT, map_location=device))
avit.eval()
print('Both models loaded.')

records = []
pred_cache = {}  # image_id -> (label_224_bool, pred_su_bool, pred_av_bool)

with torch.no_grad():
    for batch in loader:
        img_id = batch['ID'][0]
        img = batch['image'].to(device).float()
        label = batch['label'].to(device).float()

        out_su = torch.sigmoid(swinunet(img))
        pred_su = out_su.cpu().numpy() > 0.5
        lbl_np = label.cpu().numpy()
        dice_su = metrics.dc(pred_su, lbl_np)

        out_av = torch.sigmoid(avit(img, d='0')['seg'])
        pred_av = out_av.cpu().numpy() > 0.5
        dice_av = metrics.dc(pred_av, lbl_np)

        records.append({'image_id': img_id, 'swinunet_dice': dice_su, 'avit_dice': dice_av})
        pred_cache[img_id] = (
            lbl_np.squeeze().astype(bool),
            pred_su.squeeze().astype(bool),
            pred_av.squeeze().astype(bool),
        )

df = pd.DataFrame(records)
csv_path = os.path.join(OUT_DIR, 'per_image_dice.csv')
df.to_csv(csv_path, index=False)
print('Saved {} with {} rows'.format(csv_path, len(df)))

worst_su = df.sort_values('swinunet_dice').head(10).reset_index(drop=True)
worst_av = df.sort_values('avit_dice').head(10).reset_index(drop=True)
overlap = sorted(set(worst_su['image_id']) & set(worst_av['image_id']))

print('\n=== Worst 10 for Swin-UNet ===')
print(worst_su.to_string(index=False))
print('\n=== Worst 10 for AViT ===')
print(worst_av.to_string(index=False))
print('\n=== Overlap (hard for BOTH networks) ===')
print(overlap)

df['worst_dice'] = df[['swinunet_dice', 'avit_dice']].min(axis=1)
worst5 = df.sort_values('worst_dice').head(5).reset_index(drop=True)
print('\n=== 5 worst overall (by min of the two Dice scores) ===')
print(worst5.to_string(index=False))

for i, row in worst5.iterrows():
    img_id = row['image_id']
    img_raw = np.load(os.path.join(DATA_PATH, 'Image', img_id + '.npy'))
    img_disp = cv2.resize(img_raw.astype('uint8'), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    label_224, pred_su_224, pred_av_224 = pred_cache[img_id]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    axes[0].imshow(img_disp)
    axes[0].set_title('Original: {}'.format(img_id))
    axes[1].imshow(label_224, cmap='gray')
    axes[1].set_title('Ground truth')
    axes[2].imshow(pred_su_224, cmap='gray')
    axes[2].set_title('Swin-UNet (Dice={:.3f})'.format(row['swinunet_dice']))
    axes[3].imshow(pred_av_224, cmap='gray')
    axes[3].set_title('AViT (Dice={:.3f})'.format(row['avit_dice']))
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, 'worst_{:02d}_{}.png'.format(i + 1, img_id))
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    print('Saved {}'.format(out_png))

print('\nDone.')
