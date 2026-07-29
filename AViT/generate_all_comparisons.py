'''
Generates a side-by-side comparison PNG (original / ground truth / Swin-UNet / AViT)
for every image in the ISIC-2017 fold-0 test set, so the local review frontend can show
any image on demand without needing a backend. Reuses the same inference as
per_image_dice_analysis.py; does not recompute or overwrite per_image_dice.csv.
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
ALL_DIR = os.path.join(OUT_DIR, 'all_comparisons')
IMG_SIZE = 224

os.makedirs(ALL_DIR, exist_ok=True)
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
print('Both models loaded. Generating {} comparison images into {} ...'.format(len(test_ds), ALL_DIR))

count = 0
with torch.no_grad():
    for batch in loader:
        img_id = batch['ID'][0]
        img = batch['image'].to(device).float()
        label = batch['label'].to(device).float()

        out_su = torch.sigmoid(swinunet(img))
        pred_su = (out_su.cpu().numpy() > 0.5).squeeze()
        lbl_np = label.cpu().numpy()
        dice_su = metrics.dc(out_su.cpu().numpy() > 0.5, lbl_np)

        out_av = torch.sigmoid(avit(img, d='0')['seg'])
        pred_av = (out_av.cpu().numpy() > 0.5).squeeze()
        dice_av = metrics.dc(out_av.cpu().numpy() > 0.5, lbl_np)

        label_224 = lbl_np.squeeze().astype(bool)

        img_raw = np.load(os.path.join(DATA_PATH, 'Image', img_id + '.npy'))
        img_disp = cv2.resize(img_raw.astype('uint8'), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
        axes[0].imshow(img_disp)
        axes[0].set_title('Original: {}'.format(img_id))
        axes[1].imshow(label_224, cmap='gray')
        axes[1].set_title('Ground truth')
        axes[2].imshow(pred_su, cmap='gray')
        axes[2].set_title('Swin-UNet (Dice={:.3f})'.format(dice_su))
        axes[3].imshow(pred_av, cmap='gray')
        axes[3].set_title('AViT (Dice={:.3f})'.format(dice_av))
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(ALL_DIR, '{}.png'.format(img_id)), dpi=100)
        plt.close(fig)

        count += 1
        if count % 50 == 0:
            print('  {}/{} done'.format(count, len(test_ds)))

print('Done. {} comparison images saved to {}'.format(count, ALL_DIR))
