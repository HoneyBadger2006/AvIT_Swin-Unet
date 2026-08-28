"""
Visual verification folder for the 31-image SwinUnet/AViT overlap set
(swinunet_avit_overlap37_189.csv -- images hard for BOTH networks' own allin
seed42 identification runs). For each image: Original+GT | SwinUnet prediction
| AViT prediction, each prediction panel labeled with that network's own dice
on this image (from the same identification checkpoints that defined the 37/189
lists, so the labeled scores match exactly what put these images on both lists).
No sampling -- all 31 checked. Individual panels + paginated contact sheets,
matching the 37-image SwinUnet folder's format.
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

DATA_ROOT = '../data/isic2017'
OVERLAP_CSV = '../per_image_analysis_v2/bad_image_augmentation/swinunet_avit_overlap37_189.csv'
SWINUNET_CKPT = '../results/isic2017_swinunet_clahe_allin_seed42_SwinUnet_20260821_0324/final.pth'
AVIT_CKPT = '../results/isic2017_avit_clahe_allin_seed42_SwinSeg_CNNprompt_adapt_20260824_1140/final.pth'
OUT_DIR = '../per_image_analysis_v2/swinunet_avit_overlap31/augmentation_verification'
INDIVIDUAL_DIR = os.path.join(OUT_DIR, 'individual')
CONTACT_SHEET_DIR = os.path.join(OUT_DIR, 'contact_sheets')
IMAGES_PER_PAGE = 4

os.makedirs(INDIVIDUAL_DIR, exist_ok=True)
os.makedirs(CONTACT_SHEET_DIR, exist_ok=True)


def overlay_mask(ax, img, lbl, color, title):
    ax.imshow(img)
    rgba = np.zeros((*lbl.shape, 4))
    rgba[lbl > 0.5] = color
    ax.imshow(rgba)
    ax.set_title(title, fontsize=10)
    ax.axis('off')


def run_inference(ckpt, model_name, image_ids):
    config = DotDict(yaml.load(open('Configs/multi_train_local.yml'), Loader=yaml.FullLoader))
    datas = Dataset_wrap_csv(k_fold='allin', use_old_split=True, img_size=config.data.img_size,
                              dataset_name='isic2017', split_ratio=config.data.split_ratio,
                              train_aug=False, data_folder=config.data.data_folder,
                              meta_csv_name='meta_isic2017_train2000.csv', fixed_test_csv_name=None,
                              image_subdir='Image_clahe')
    train_ds = datas['train']
    id_to_index = {row['ID']: i for i, row in train_ds.df.reset_index(drop=True).iterrows()}

    model = build_model(model_name, config)
    model.load_state_dict(torch.load(ckpt))
    model.eval()

    results = {}
    with torch.no_grad():
        for image_id in image_ids:
            idx = id_to_index[image_id]
            sample = train_ds[idx]
            img_t = sample['image'].unsqueeze(0).cuda().float()
            label_t = sample['label'].unsqueeze(0).cuda().float()
            logits = forward_logits(model, model_name, img_t)
            prob = torch.sigmoid(logits)
            pred = (prob.cpu().numpy()[0, 0] > 0.5)
            gt = (label_t.cpu().numpy()[0, 0] > 0.5)
            dice = metrics.dc(pred, gt)
            results[image_id] = {'pred': pred, 'gt': gt, 'dice': dice}
    del model
    torch.cuda.empty_cache()
    return results


def main():
    overlap = pd.read_csv(OVERLAP_CSV, dtype={'image_id': str})
    assert len(overlap) == 31, len(overlap)
    image_ids = sorted(overlap['image_id'])
    print('Building overlap review for {} images'.format(len(image_ids)))

    print('Running SwinUnet inference...')
    sw_results = run_inference(SWINUNET_CKPT, 'SwinUnet', image_ids)
    print('Running AViT inference...')
    av_results = run_inference(AVIT_CKPT, 'SwinSeg_CNNprompt_adapt', image_ids)

    orig224 = {}
    for image_id in image_ids:
        import cv2
        img = np.load(os.path.join(DATA_ROOT, 'Image_clahe', image_id + '.npy'))
        orig224[image_id] = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)

    saved_paths = []
    for image_id in image_ids:
        img = orig224[image_id]
        gt = sw_results[image_id]['gt']  # same GT for both (same image)
        sw_dice = sw_results[image_id]['dice']
        av_dice = av_results[image_id]['dice']

        fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
        fig.suptitle('{}  (both networks flagged hard at identification)'.format(image_id), fontsize=12)
        overlay_mask(axes[0], img, gt, [0.15, 1.0, 0.15, 0.5], 'Original + GT')
        overlay_mask(axes[1], img, sw_results[image_id]['pred'], [0.2, 0.4, 1.0, 0.5],
                     'SwinUnet prediction\nDice = {:.3f}'.format(sw_dice))
        overlay_mask(axes[2], img, av_results[image_id]['pred'], [1.0, 0.2, 0.2, 0.5],
                     'AViT prediction\nDice = {:.3f}'.format(av_dice))

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        out_path = os.path.join(INDIVIDUAL_DIR, '{}_sw{:.3f}_av{:.3f}.png'.format(image_id, sw_dice, av_dice))
        fig.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        saved_paths.append(out_path)

    print('Saved {} individual review images to {}'.format(len(saved_paths), INDIVIDUAL_DIR))

    n_pages = (len(image_ids) + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE
    for page in range(n_pages):
        chunk = image_ids[page * IMAGES_PER_PAGE:(page + 1) * IMAGES_PER_PAGE]
        fig, axes = plt.subplots(len(chunk), 3, figsize=(13.5, 4.6 * len(chunk)))
        if len(chunk) == 1:
            axes = axes.reshape(1, 3)
        for i, image_id in enumerate(chunk):
            img = orig224[image_id]
            gt = sw_results[image_id]['gt']
            sw_dice = sw_results[image_id]['dice']
            av_dice = av_results[image_id]['dice']
            overlay_mask(axes[i, 0], img, gt, [0.15, 1.0, 0.15, 0.5], '{} - Original + GT'.format(image_id))
            overlay_mask(axes[i, 1], img, sw_results[image_id]['pred'], [0.2, 0.4, 1.0, 0.5],
                         'SwinUnet Dice={:.3f}'.format(sw_dice))
            overlay_mask(axes[i, 2], img, av_results[image_id]['pred'], [1.0, 0.2, 0.2, 0.5],
                         'AViT Dice={:.3f}'.format(av_dice))

        plt.tight_layout()
        sheet_path = os.path.join(CONTACT_SHEET_DIR, 'contact_sheet_page{:02d}.png'.format(page + 1))
        fig.savefig(sheet_path, dpi=90)
        plt.close(fig)
        print('Saved contact sheet:', sheet_path)

    print('Total contact sheet pages: {}'.format(n_pages))
    print('DONE')


if __name__ == '__main__':
    main()
