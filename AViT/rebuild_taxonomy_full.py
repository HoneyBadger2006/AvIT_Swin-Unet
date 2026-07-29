'''
Full-dataset (no sampling) per-image Dice + loss collection for all 4 configs
(SwinUnet baseline/+Focal, AViT baseline/+Focal) on ALL 2000 official training
images and ALL 600 official test images, using each config's k0 checkpoint.

batch_size=1 throughout: dice_loss and FocalLoss reduce over the whole batch
tensor, so batch>1 would silently blend multiple images into one "per-image"
number. batch=1 is required for per-image loss_value to be correct.

Saves 8 CSVs (train/test x 4 configs): image_id, dice_score, loss_value.
Caches baseline predictions (SwinUnet, AViT) for every image so the
cross-network overlap analysis and visualizations don't need a second
inference pass.
'''
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import medpy.metric.binary as metrics
from scipy.ndimage import label as cc_label

from Datasets.create_dataset import SkinDataset_csv
from Utils.pieces import DotDict
from Utils.losses import dice_loss
from Models.Transformer.SwinUnet import SwinUnet
from Models.Transformer.Swin_adapters import SwinSimpleSeg_CNNprompt_adapt
from multi_train_adapt import compute_loss

ROOT = 'C:/Users/quanp/Downloads/ISIC 2017'
DATA_PATH = ROOT + '/data/isic2017/'
PRETRAINED_FOLDER = ROOT + '/pretrained_ckpt'
OUT_DIR = ROOT + '/per_image_analysis_v2'
VIZ_DIR = os.path.join(OUT_DIR, 'low_dice_visualizations')
IMG_SIZE = 224

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIGS = [
    {'network': 'SwinUnet', 'config': 'baseline', 'is_baseline': True,
     'ckpt': ROOT + '/results/isic2017_swinunet_official_baseline_k0_SwinUnet_20260721_1321/best.pth'},
    {'network': 'SwinUnet', 'config': 'focal', 'is_baseline': False,
     'ckpt': ROOT + '/results/isic2017_swinunet_official_focal_k0_SwinUnet_20260721_1640/best.pth'},
    {'network': 'AViT', 'config': 'baseline', 'is_baseline': True,
     'ckpt': ROOT + '/results/isic2017_avit_official_baseline_k0_SwinSeg_CNNprompt_adapt_20260721_1949/best.pth'},
    {'network': 'AViT', 'config': 'focal', 'is_baseline': False,
     'ckpt': ROOT + '/results/isic2017_avit_official_focal_k0_SwinSeg_CNNprompt_adapt_20260722_0001/best.pth'},
]

CRITERION = [nn.BCELoss(), dice_loss]  # AViT's non-focal base loss, matches training


def build_model(network, ckpt):
    if network == 'SwinUnet':
        model = SwinUnet(img_size=IMG_SIZE)
        model_name = 'SwinUnet'
    else:
        model = SwinSimpleSeg_CNNprompt_adapt(
            img_size=IMG_SIZE, pretrained=False, pretrained_swin_name='swin_large_patch4_window7_224_22k',
            pretrained_folder=PRETRAINED_FOLDER, embed_dim=192, drop_path_rate=0.2,
            depths=[2, 2, 18], num_heads=[6, 12, 24], window_size=7,
            debug=False, adapt_method=False, num_domains=1,
        )
        model_name = 'SwinSeg_CNNprompt_adapt'
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    return model, model_name


def run_full_inference(model, network, model_name, use_focal, df, cache_masks):
    cfg_obj = DotDict({'model': model_name, 'train': {'use_focal_loss': use_focal, 'focal_lambda': 1.0}})
    ds = SkinDataset_csv('isic2017', IMG_SIZE, df, use_aug=False, data_path=DATA_PATH)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    records = []
    mask_cache = {} if cache_masks else None
    n = len(ds)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            img_id = batch['ID'][0]
            img = batch['image'].to(device).float()
            label = batch['label'].to(device).float()
            if network == 'SwinUnet':
                output = model(img)
            else:
                output = model(img, d='0')['seg']
            loss, prob = compute_loss(cfg_obj, CRITERION, output, label)
            pred = (prob.cpu().numpy() > 0.5)
            lbl = (label.cpu().numpy() > 0.5)
            dice = metrics.dc(pred, lbl)
            records.append({'image_id': img_id, 'dice_score': dice, 'loss_value': loss.item()})
            if cache_masks:
                mask_cache[img_id] = (lbl[0, 0], pred[0, 0])
            if (i + 1) % 400 == 0:
                print('    {}/{}'.format(i + 1, n), flush=True)
    return pd.DataFrame(records), mask_cache


def make_4panel(image_id, gt, pred, dice, network, config, out_path):
    img_raw = np.load(os.path.join(DATA_PATH, 'Image', image_id + '.npy')).astype('uint8')
    img_disp = cv2.resize(img_raw, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    diff = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype='uint8') * 255
    fn = gt & ~pred   # missed lesion (false negative) -> red
    fp = ~gt & pred   # false alarm (false positive) -> blue
    diff[fn] = [220, 30, 30]
    diff[fp] = [30, 60, 220]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    axes[0].imshow(img_disp)
    axes[0].set_title('Original: {}'.format(image_id))
    axes[1].imshow(gt, cmap='gray')
    axes[1].set_title('Ground truth')
    axes[2].imshow(pred, cmap='gray')
    axes[2].set_title('{} {} (Dice={:.3f})'.format(network, config, dice))
    axes[3].imshow(diff)
    axes[3].set_title('Error (red=FN, blue=FP)')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    train_df = pd.read_csv(DATA_PATH + 'meta_isic2017_train2000.csv', dtype={'ID': str})
    test_df = pd.read_csv(DATA_PATH + 'meta_isic2017_test600.csv', dtype={'ID': str})
    assert len(train_df) == 2000 and len(test_df) == 600

    all_csvs = {}
    baseline_caches = {}  # network -> {'train': mask_cache, 'test': mask_cache}

    for cfg in CONFIGS:
        key = '{}_{}'.format(cfg['network'].lower(), cfg['config'])
        print('=== {} ==='.format(key))
        model, model_name = build_model(cfg['network'], cfg['ckpt'])

        for split_name, df in [('train', train_df), ('test', test_df)]:
            print('  {} split ({} images)'.format(split_name, len(df)))
            cache_this = cfg['is_baseline']
            rec_df, mask_cache = run_full_inference(
                model, cfg['network'], model_name, not cfg['is_baseline'], df, cache_this)
            csv_path = os.path.join(OUT_DIR, '{}_{}.csv'.format(split_name, key))
            rec_df.to_csv(csv_path, index=False)
            print('  Saved {} ({} rows)'.format(csv_path, len(rec_df)))
            all_csvs[(split_name, key)] = rec_df
            if cache_this:
                baseline_caches.setdefault(cfg['network'], {})[split_name] = mask_cache

        del model
        torch.cuda.empty_cache()

    # ---- Section 3: % of TRAIN images with Dice < 0.5, per config ----
    print('\n=== TRAIN Dice<0.5 percentages per config ===')
    train_pct = {}
    for cfg in CONFIGS:
        key = '{}_{}'.format(cfg['network'].lower(), cfg['config'])
        df = all_csvs[('train', key)]
        pct = 100.0 * (df['dice_score'] < 0.5).mean()
        train_pct[key] = pct
        print('  {}: {:.2f}% ({} / {} images)'.format(
            key, pct, int((df['dice_score'] < 0.5).sum()), len(df)))

    # ---- Section 4: cross-network baseline overlap across all 2600 images ----
    su_base = pd.concat([
        all_csvs[('train', 'swinunet_baseline')].assign(split='train'),
        all_csvs[('test', 'swinunet_baseline')].assign(split='test'),
    ], ignore_index=True)
    av_base = pd.concat([
        all_csvs[('train', 'avit_baseline')].assign(split='train'),
        all_csvs[('test', 'avit_baseline')].assign(split='test'),
    ], ignore_index=True)
    assert len(su_base) == 2600 and len(av_base) == 2600

    su_low = set(su_base.loc[su_base['dice_score'] < 0.5, 'image_id'])
    av_low = set(av_base.loc[av_base['dice_score'] < 0.5, 'image_id'])
    overlap = su_low & av_low
    union = su_low | av_low

    print('\n=== Cross-network baseline overlap (Dice<0.5, all 2600 images) ===')
    print('  SwinUnet baseline low-dice: {} images'.format(len(su_low)))
    print('  AViT baseline low-dice: {} images'.format(len(av_low)))
    print('  OVERLAP (both networks): {} images'.format(len(overlap)))
    print('  UNION (either network): {} images'.format(len(union)))

    overlap_df = pd.DataFrame({'image_id': sorted(overlap)})
    overlap_df.to_csv(os.path.join(OUT_DIR, 'cross_network_overlap_ids.csv'), index=False)

    # ---- statistical comparison: overlap set vs rest, using image_measures.csv ----
    measures = pd.read_csv(os.path.join(OUT_DIR, 'image_measures.csv'), dtype={'image_id': str})
    measures['in_overlap'] = measures['image_id'].isin(overlap)
    measures.to_csv(os.path.join(OUT_DIR, 'image_measures_with_overlap_flag.csv'), index=False)

    from scipy.stats import mannwhitneyu
    stat_cols = ['hair_coverage_frac', 'hair_coverage_boundary_frac', 'hair_coverage_interior_frac',
                 'contrast_laplacian_var', 'boundary_gradient_mean', 'lesion_size_frac', 'gt_component_count']
    print('\n=== Statistical comparison: overlap set (n={}) vs rest (n={}) ==='.format(
        measures['in_overlap'].sum(), (~measures['in_overlap']).sum()))
    stat_results = []
    for col in stat_cols:
        a = measures.loc[measures['in_overlap'], col].dropna()
        b = measures.loc[~measures['in_overlap'], col].dropna()
        u, p = mannwhitneyu(a, b, alternative='two-sided')
        stat_results.append({
            'measure': col, 'overlap_median': a.median(), 'rest_median': b.median(),
            'overlap_mean': a.mean(), 'rest_mean': b.mean(), 'mannwhitney_p': p,
        })
        print('  {}: overlap median={:.4f} mean={:.4f} | rest median={:.4f} mean={:.4f} | p={:.2e}'.format(
            col, a.median(), a.mean(), b.median(), b.mean(), p))
    pd.DataFrame(stat_results).to_csv(os.path.join(OUT_DIR, 'overlap_vs_rest_stats.csv'), index=False)

    # ---- Section 5: 4-panel visualizations for every image with Dice<0.5 in either baseline network ----
    print('\n=== Generating 4-panel visualizations ===')
    total_viz = 0
    for network, low_set in [('SwinUnet', su_low), ('AViT', av_low)]:
        subdir = os.path.join(VIZ_DIR, '{}_baseline'.format(network.lower()))
        os.makedirs(subdir, exist_ok=True)
        df_net = su_base if network == 'SwinUnet' else av_base
        for img_id in sorted(low_set):
            split = df_net.loc[df_net['image_id'] == img_id, 'split'].iloc[0]
            dice = df_net.loc[df_net['image_id'] == img_id, 'dice_score'].iloc[0]
            gt, pred = baseline_caches[network][split][img_id]
            out_path = os.path.join(subdir, '{}_{}_dice{:.3f}.png'.format(img_id, split, dice))
            make_4panel(img_id, gt, pred, dice, network, 'baseline', out_path)
            total_viz += 1
        print('  {}: {} visualizations saved to {}'.format(network, len(low_set), subdir))

    print('\n=== SUMMARY ===')
    print('Union (Dice<0.5 in either network): {} images -> {} total visualization files saved'.format(
        len(union), total_viz))
    print('DONE.')


if __name__ == '__main__':
    main()
