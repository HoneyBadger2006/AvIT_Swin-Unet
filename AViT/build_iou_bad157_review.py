"""
Builds Prof. Samavi's requested review folder for the 157 images seed42's checkpoint
scores below 0.7 IoU (the concrete, reproducible list -- not the 3-seed average of
231, which isn't a real list of images). For each image saves:
  - the original (CLAHE) image
  - the ground-truth mask
  - the model's predicted mask (freshly re-inferred from seed42's checkpoint)
as a labeled 4-panel comparison PNG (original | +GT overlay | +prediction overlay |
GT-vs-prediction disagreement), titled with image ID, Dice, and IoU. Also builds
paginated contact sheets for quick browsing, and a summary CSV + distribution stats.
No sampling -- all 157 images.
"""
import os
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
import medpy.metric.binary as metrics

from Datasets.create_dataset import Dataset_wrap_csv
from Utils.pieces import DotDict
from tta_inference import build_model, forward_logits

CKPT = '../results/isic2017_swinunet_clahe_allin_seed42_SwinUnet_20260821_0324/final.pth'
DICE_IOU_CSV = '../per_image_analysis_v2/bad_image_augmentation/seed42_per_image_dice_iou.csv'
OUT_DIR = '../per_image_analysis_v2/seed42_iou_bad157'
INDIVIDUAL_DIR = os.path.join(OUT_DIR, 'individual')
CONTACT_SHEET_DIR = os.path.join(OUT_DIR, 'contact_sheets')
IMAGES_PER_PAGE = 6

os.makedirs(INDIVIDUAL_DIR, exist_ok=True)
os.makedirs(CONTACT_SHEET_DIR, exist_ok=True)


def overlay(ax, img, mask, color, title):
    ax.imshow(img)
    rgba = np.zeros((*mask.shape, 4))
    rgba[mask > 0.5] = color
    ax.imshow(rgba)
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def combined_diff(ax, img, gt, pred, title):
    ax.imshow(img)
    rgba = np.zeros((*gt.shape, 4))
    rgba[(gt > 0.5) & (pred <= 0.5)] = [0.15, 1.0, 0.15, 0.5]   # GT only (missed) -> green
    rgba[(gt <= 0.5) & (pred > 0.5)] = [1.0, 0.15, 0.15, 0.5]   # Pred only (false positive) -> red
    rgba[(gt > 0.5) & (pred > 0.5)] = [1.0, 1.0, 0.15, 0.5]     # agreement -> yellow
    ax.imshow(rgba)
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def main():
    dice_iou = pd.read_csv(DICE_IOU_CSV, dtype={'image_id': str})
    bad = dice_iou[dice_iou['iou'] < 0.7].sort_values('iou').reset_index(drop=True)
    print('Images to review (IoU < 0.7, seed42): {}'.format(len(bad)))
    assert len(bad) == 157

    config = DotDict(yaml.load(open('Configs/multi_train_local.yml'), Loader=yaml.FullLoader))
    datas = Dataset_wrap_csv(k_fold='allin', use_old_split=True, img_size=config.data.img_size,
                              dataset_name='isic2017', split_ratio=config.data.split_ratio,
                              train_aug=False, data_folder=config.data.data_folder,
                              meta_csv_name='meta_isic2017_train2000.csv', fixed_test_csv_name=None,
                              image_subdir='Image_clahe')
    train_ds = datas['train']
    id_to_index = {row['ID']: i for i, row in train_ds.df.reset_index(drop=True).iterrows()}

    model = build_model('SwinUnet', config)
    model.load_state_dict(torch.load(CKPT))
    model.eval()

    DATA_ROOT = '../data/isic2017'
    rows = []
    saved_paths = []
    cache = {}  # image_id -> (orig_img, gt, pred), reused for the contact sheets below
    with torch.no_grad():
        for _, row in bad.iterrows():
            image_id = row['image_id']
            saved_dice, saved_iou = row['dice'], row['iou']

            idx = id_to_index[image_id]
            sample = train_ds[idx]
            img_t = sample['image'].unsqueeze(0).cuda().float()
            label_t = sample['label'].unsqueeze(0).cuda().float()
            logits = forward_logits(model, 'SwinUnet', img_t)
            prob = torch.sigmoid(logits)
            pred = (prob.cpu().numpy()[0, 0] > 0.5)
            gt = (label_t.cpu().numpy()[0, 0] > 0.5)

            fresh_dice = metrics.dc(pred, gt)
            fresh_iou = metrics.jc(pred, gt)

            # display image: raw CLAHE array, for visual clarity (not the normalized tensor)
            orig_img = np.load(os.path.join(DATA_ROOT, 'Image_clahe', image_id + '.npy'))
            cache[image_id] = (orig_img, gt, pred)

            fig, axes = plt.subplots(1, 4, figsize=(19, 5))
            fig.suptitle('{}   saved: Dice={:.3f} IoU={:.3f}   |   fresh recheck: Dice={:.3f} IoU={:.3f}'.format(
                image_id, saved_dice, saved_iou, fresh_dice, fresh_iou), fontsize=11)
            axes[0].imshow(orig_img); axes[0].set_title('Original (CLAHE)', fontsize=9); axes[0].axis('off')
            overlay(axes[1], orig_img, gt.astype(np.float32), [0.15, 1.0, 0.15, 0.45], 'Ground truth')
            overlay(axes[2], orig_img, pred.astype(np.float32), [1.0, 0.15, 0.15, 0.45], 'Model prediction (seed42)')
            combined_diff(axes[3], orig_img, gt.astype(np.float32), pred.astype(np.float32),
                          'Diff: green=missed  red=false-pos  yellow=agree')

            plt.tight_layout()
            out_path = os.path.join(INDIVIDUAL_DIR, '{}_iou{:.3f}_dice{:.3f}.png'.format(image_id, saved_iou, saved_dice))
            fig.savefig(out_path, dpi=100)
            plt.close(fig)
            saved_paths.append(out_path)

            rows.append({'image_id': image_id, 'saved_dice': saved_dice, 'saved_iou': saved_iou,
                         'fresh_dice': fresh_dice, 'fresh_iou': fresh_iou})

    print('Saved {} individual review images to {}'.format(len(saved_paths), INDIVIDUAL_DIR))

    result_df = pd.DataFrame(rows)
    result_df.to_csv(os.path.join(OUT_DIR, 'review157_summary.csv'), index=False)

    # cross-check: fresh re-inference still agrees with the saved list (no drift)
    still_bad = (result_df['fresh_iou'] < 0.7).sum()
    max_dice_drift = (result_df['fresh_dice'] - result_df['saved_dice']).abs().max()
    max_iou_drift = (result_df['fresh_iou'] - result_df['saved_iou']).abs().max()
    print('\n[Cross-check] Fresh re-inference: {} / 157 still IoU<0.7 (max dice drift={:.2e}, max iou drift={:.2e})'.format(
        still_bad, max_dice_drift, max_iou_drift))

    # contact sheets
    n_pages = (len(bad) + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE
    for page in range(n_pages):
        chunk = bad.iloc[page * IMAGES_PER_PAGE:(page + 1) * IMAGES_PER_PAGE]
        fig, axes = plt.subplots(len(chunk), 4, figsize=(19, 4.5 * len(chunk)))
        if len(chunk) == 1:
            axes = axes.reshape(1, 4)
        for i, (_, row) in enumerate(chunk.iterrows()):
            image_id = row['image_id']
            dice_v, iou_v = row['dice'], row['iou']
            orig_img, gt, pred = cache[image_id]

            axes[i, 0].imshow(orig_img)
            axes[i, 0].set_title('{}  Dice={:.3f} IoU={:.3f}'.format(image_id, dice_v, iou_v), fontsize=9)
            axes[i, 0].axis('off')
            overlay(axes[i, 1], orig_img, gt.astype(np.float32), [0.15, 1.0, 0.15, 0.45], 'GT')
            overlay(axes[i, 2], orig_img, pred.astype(np.float32), [1.0, 0.15, 0.15, 0.45], 'Prediction')
            combined_diff(axes[i, 3], orig_img, gt.astype(np.float32), pred.astype(np.float32), 'Diff')

        plt.tight_layout()
        sheet_path = os.path.join(CONTACT_SHEET_DIR, 'contact_sheet_page{:02d}.png'.format(page + 1))
        fig.savefig(sheet_path, dpi=90)
        plt.close(fig)
        print('Saved contact sheet:', sheet_path)

    print('\n=== Distribution summary (157 images, saved dice/iou) ===')
    print(bad[['dice', 'iou']].describe())
    print()
    for lo, hi in [(0.6, 0.7), (0.5, 0.6), (0.3, 0.5), (0.0, 0.3)]:
        n = ((bad['iou'] >= lo) & (bad['iou'] < hi)).sum()
        print('  IoU [{:.1f},{:.1f}): {}'.format(lo, hi, n))

    print('\nBUILD_REVIEW157_DONE')


if __name__ == '__main__':
    main()
