'''
Per-image problem taxonomy for the official-split k0 checkpoints (4 configs:
SwinUnet baseline/focal, AViT baseline/focal). Inference only, no training.

Collection 1 (train): each config's own fold-0 training images (1600, direct exposure).
Collection 2 (test): the fixed 600-image official test set (same for all configs).

For each collection, keeps the worst 15-20 per config (by Dice), tags severity tier
and ground-truth/prediction connected-component counts (multi-lesion / erasure flag),
and saves side-by-side PNGs for the global worst 10 images per collection.
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
from Models.Transformer.Swin_adapters import SwinSimpleSeg_CNNprompt_adapt

ROOT = 'C:/Users/quanp/Downloads/ISIC 2017'
DATA_PATH = ROOT + '/data/isic2017/'
PRETRAINED_FOLDER = ROOT + '/pretrained_ckpt'
OUT_DIR = ROOT + '/per_image_analysis_official'
IMG_SIZE = 224
WORST_PER_CONFIG = 20
WORST_PNG_PER_COLLECTION = 10

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIGS = [
    {'network': 'SwinUnet', 'config': 'baseline',
     'ckpt': ROOT + '/results/isic2017_swinunet_official_baseline_k0_SwinUnet_20260721_1321/best.pth'},
    {'network': 'SwinUnet', 'config': 'focal',
     'ckpt': ROOT + '/results/isic2017_swinunet_official_focal_k0_SwinUnet_20260721_1640/best.pth'},
    {'network': 'AViT', 'config': 'baseline',
     'ckpt': ROOT + '/results/isic2017_avit_official_baseline_k0_SwinSeg_CNNprompt_adapt_20260721_1949/best.pth'},
    {'network': 'AViT', 'config': 'focal',
     'ckpt': ROOT + '/results/isic2017_avit_official_focal_k0_SwinSeg_CNNprompt_adapt_20260722_0001/best.pth'},
]

TRAIN_CSV = DATA_PATH + 'train_meta_kfold_meta_isic2017_train2000_0.csv'
TEST_CSV = DATA_PATH + 'meta_isic2017_test600.csv'


def build_model(network, ckpt):
    if network == 'SwinUnet':
        model = SwinUnet(img_size=IMG_SIZE)
    else:
        model = SwinSimpleSeg_CNNprompt_adapt(
            img_size=IMG_SIZE, pretrained=False, pretrained_swin_name='swin_large_patch4_window7_224_22k',
            pretrained_folder=PRETRAINED_FOLDER, embed_dim=192, drop_path_rate=0.2,
            depths=[2, 2, 18], num_heads=[6, 12, 24], window_size=7,
            debug=False, adapt_method=False, num_domains=1,
        )
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    return model


def severity_tier(dice):
    if dice < 0.3:
        return 'very hard'
    elif dice < 0.6:
        return 'hard'
    else:
        return 'mild'


def run_inference(model, network, df):
    ds = SkinDataset_csv('isic2017', IMG_SIZE, df, use_aug=False, data_path=DATA_PATH)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)
    records = []
    pred_cache = {}
    with torch.no_grad():
        for batch in loader:
            img = batch['image'].to(device).float()
            label = batch['label'].to(device).float()
            if network == 'SwinUnet':
                out = torch.sigmoid(model(img))
            else:
                out = torch.sigmoid(model(img, d='0')['seg'])
            pred = out.cpu().numpy() > 0.5
            lbl = label.cpu().numpy() > 0.5
            for i in range(img.shape[0]):
                img_id = batch['ID'][i]
                p = pred[i, 0]
                g = lbl[i, 0]
                dice = metrics.dc(p, g)
                _, gt_n = cc_label(g)
                _, pred_n = cc_label(p)
                records.append({
                    'image_id': img_id,
                    'dice': dice,
                    'ground_truth_component_count': gt_n,
                    'prediction_component_count': pred_n,
                })
                pred_cache[img_id] = (g, p)
    return pd.DataFrame(records), pred_cache


def build_collection(name, df):
    print('=== Collection: {} ({} images) ==='.format(name, len(df)))
    all_worst = []
    pred_caches = {}  # (network, config) -> {image_id: (gt, pred)}
    for cfg in CONFIGS:
        key = (cfg['network'], cfg['config'])
        print('  Running {} {} ...'.format(*key))
        model = build_model(cfg['network'], cfg['ckpt'])
        rec_df, pred_cache = run_inference(model, cfg['network'], df)
        pred_caches[key] = pred_cache
        rec_df['network'] = cfg['network']
        rec_df['config'] = cfg['config']
        rec_df['severity_tier'] = rec_df['dice'].apply(severity_tier)
        rec_df['possible_multi_lesion_flag'] = rec_df['ground_truth_component_count'] > 1
        worst = rec_df.sort_values('dice').head(WORST_PER_CONFIG)
        all_worst.append(worst)
        del model
        torch.cuda.empty_cache()

    combined = pd.concat(all_worst, ignore_index=True)
    combined = combined[['image_id', 'network', 'config', 'dice', 'severity_tier',
                          'ground_truth_component_count', 'prediction_component_count',
                          'possible_multi_lesion_flag']]
    csv_path = os.path.join(OUT_DIR, '{}_problems.csv'.format(name))
    combined.to_csv(csv_path, index=False)
    print('Saved {} ({} rows)'.format(csv_path, len(combined)))

    # Global worst 10 across all 4 configs for this collection, for PNGs
    png_rows = combined.sort_values('dice').head(WORST_PNG_PER_COLLECTION).reset_index(drop=True)
    for i, row in png_rows.iterrows():
        img_id = row['image_id']
        key = (row['network'], row['config'])
        g, p = pred_caches[key][img_id]
        img_raw = np.load(os.path.join(DATA_PATH, 'Image', img_id + '.npy'))
        img_disp = cv2.resize(img_raw.astype('uint8'), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
        axes[0].imshow(img_disp)
        axes[0].set_title('Original: {}'.format(img_id))
        axes[1].imshow(g, cmap='gray')
        axes[1].set_title('Ground truth ({} comp.)'.format(row['ground_truth_component_count']))
        axes[2].imshow(p, cmap='gray')
        axes[2].set_title('{} {} (Dice={:.3f}, {} comp.)'.format(
            row['network'], row['config'], row['dice'], row['prediction_component_count']))
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        out_png = os.path.join(OUT_DIR, '{}_worst_{:02d}_{}_{}_{}.png'.format(
            name, i + 1, img_id, row['network'], row['config']))
        plt.savefig(out_png, dpi=120)
        plt.close(fig)
        print('  Saved PNG {}'.format(out_png))

    return combined


if __name__ == '__main__':
    train_df = pd.read_csv(TRAIN_CSV, dtype={'ID': str}).rename(columns={'ID': 'ID'})
    test_df = pd.read_csv(TEST_CSV, dtype={'ID': str})

    train_result = build_collection('train', train_df)
    test_result = build_collection('test', test_df)

    print('\n=== DONE ===')
    print('train_problems.csv: {} rows'.format(len(train_result)))
    print('test_problems.csv: {} rows'.format(len(test_result)))
