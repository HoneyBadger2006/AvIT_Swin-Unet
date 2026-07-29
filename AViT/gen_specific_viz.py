'''
Standalone 3-panel visualizations for two specific images requested for slides:
- 0012837: SwinUnet baseline, test set (37 GT components, flagged in taxonomy work)
- 0015353: SwinUnet baseline, test set (biggest GT->pred component gap, 77 -> 1)
Same format/style as build_problem_taxonomy.py's PNGs.
'''
import os
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import medpy.metric.binary as metrics
from scipy.ndimage import label as cc_label

from Datasets.create_dataset import SkinDataset_csv
from Models.Transformer.SwinUnet import SwinUnet

ROOT = 'C:/Users/quanp/Downloads/ISIC 2017'
DATA_PATH = ROOT + '/data/isic2017/'
OUT_DIR = ROOT + '/per_image_analysis_official'
IMG_SIZE = 224
CKPT = ROOT + '/results/isic2017_swinunet_official_baseline_k0_SwinUnet_20260721_1321/best.pth'
TARGET_IDS = ['0012837', '0015353']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_df = pd.read_csv(DATA_PATH + 'meta_isic2017_test600.csv', dtype={'ID': str})
df = test_df[test_df['ID'].isin(TARGET_IDS)].reset_index(drop=True)
assert len(df) == 2, 'expected 2 matching rows, got {}'.format(len(df))

model = SwinUnet(img_size=IMG_SIZE)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.to(device).eval()

ds = SkinDataset_csv('isic2017', IMG_SIZE, df, use_aug=False, data_path=DATA_PATH)
loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)

with torch.no_grad():
    for batch in loader:
        img = batch['image'].to(device).float()
        label = batch['label'].to(device).float()
        out = torch.sigmoid(model(img))
        pred = out.cpu().numpy() > 0.5
        lbl = label.cpu().numpy() > 0.5

        for i in range(img.shape[0]):
            img_id = batch['ID'][i]
            p = pred[i, 0]
            g = lbl[i, 0]
            dice = metrics.dc(p, g)
            _, gt_n = cc_label(g)
            _, pred_n = cc_label(p)

            img_raw = np.load(os.path.join(DATA_PATH, 'Image', img_id + '.npy'))
            img_disp = cv2.resize(img_raw.astype('uint8'), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
            axes[0].imshow(img_disp)
            axes[0].set_title('Original: {}'.format(img_id))
            axes[1].imshow(g, cmap='gray')
            axes[1].set_title('Ground truth ({} comp.)'.format(gt_n))
            axes[2].imshow(p, cmap='gray')
            axes[2].set_title('SwinUnet baseline (Dice={:.3f}, {} comp.)'.format(dice, pred_n))
            for ax in axes:
                ax.axis('off')
            plt.tight_layout()
            out_png = os.path.join(OUT_DIR, 'specific_{}_SwinUnet_baseline.png'.format(img_id))
            plt.savefig(out_png, dpi=120)
            plt.close(fig)
            print('Saved {} | dice={:.6f} gt_components={} pred_components={}'.format(
                out_png, dice, gt_n, pred_n))
