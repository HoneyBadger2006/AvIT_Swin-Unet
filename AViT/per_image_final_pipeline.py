"""
Recomputes the full baseline -> CLAHE -> CLAHE+FTL -> TTA pipeline for both networks
using PER-IMAGE Dice (mean of each image's individual Dice score), instead of the
batch-pooled Dice used throughout this project's multi_train_adapt.py test() and
tta_inference.py (metrics.dc() called on a whole 5-image batch tensor at once, which
reads higher whenever a hard-failure image's error is diluted across the rest of its
batch). Per-image macro-averaging is the more standard aggregation and does not have
that dilution effect.

Stages (existing checkpoints only, no retraining):
  1. baseline    - no CLAHE, no FTL, no TTA (Image subdir)
  2. + CLAHE     - CLAHE, no FTL, no TTA (Image_clahe subdir)
  3. + CLAHE+FTL - CLAHE + compound Focal Tversky Loss, no TTA (Image_clahe subdir)
     (both CLAHE-alone and CLAHE+FTL are evaluated per-image for both networks; the
     one with the higher per-image mean becomes each network's stage-3/4 checkpoint,
     rather than assuming AViT=FTL / SwinUnet=CLAHE-alone as under the old numbers)
  4. + TTA       - 5-view soft-voted TTA on top of the stage-3 winner's checkpoint

Reuses the shared Utils/tta.py TTA implementation (via tta_inference.py's
build_model/tta_forward) and medpy.metric.binary.dc, for exact consistency with all
previously reported numbers modulo the aggregation-method fix.
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import yaml
import medpy.metric.binary as metrics

from Datasets.create_dataset import Dataset_wrap_csv
from Utils.pieces import DotDict
from tta_inference import build_model, forward_logits, tta_forward

BASELINE_CKPTS = {
    'SwinUnet': {
        0: '../results/isic2017_swinunet_official_baseline_k0_SwinUnet_20260721_1321/best.pth',
        1: '../results/isic2017_swinunet_official_baseline_k1_SwinUnet_20260721_1409/best.pth',
        2: '../results/isic2017_swinunet_official_baseline_k2_SwinUnet_20260721_1447/best.pth',
        3: '../results/isic2017_swinunet_official_baseline_k3_SwinUnet_20260721_1525/best.pth',
        4: '../results/isic2017_swinunet_official_baseline_k4_SwinUnet_20260721_1602/best.pth',
    },
    'AViT': {
        0: '../results/isic2017_avit_official_baseline_k0_SwinSeg_CNNprompt_adapt_20260721_1949/best.pth',
        1: '../results/isic2017_avit_official_baseline_k1_SwinSeg_CNNprompt_adapt_20260721_2039/best.pth',
        2: '../results/isic2017_avit_official_baseline_k2_SwinSeg_CNNprompt_adapt_20260721_2130/best.pth',
        3: '../results/isic2017_avit_official_baseline_k3_SwinSeg_CNNprompt_adapt_20260721_2220/best.pth',
        4: '../results/isic2017_avit_official_baseline_k4_SwinSeg_CNNprompt_adapt_20260721_2311/best.pth',
    },
}

CLAHE_CKPTS = {
    'SwinUnet': {
        0: '../results/isic2017_swinunet_clahe_k0_SwinUnet_20260728_0301/best.pth',
        1: '../results/isic2017_swinunet_clahe_k1_SwinUnet_20260728_1556/best.pth',
        2: '../results/isic2017_swinunet_clahe_k2_SwinUnet_20260728_1645/best.pth',
        3: '../results/isic2017_swinunet_clahe_k3_SwinUnet_20260728_1733/best.pth',
        4: '../results/isic2017_swinunet_clahe_k4_SwinUnet_20260728_1823/best.pth',
    },
    'AViT': {
        0: '../results/isic2017_avit_clahe_k0_SwinSeg_CNNprompt_adapt_20260728_0344/best.pth',
        1: '../results/isic2017_avit_clahe_k1_SwinSeg_CNNprompt_adapt_20260728_1914/best.pth',
        2: '../results/isic2017_avit_clahe_k2_SwinSeg_CNNprompt_adapt_20260728_2020/best.pth',
        3: '../results/isic2017_avit_clahe_k3_SwinSeg_CNNprompt_adapt_20260728_2134/best.pth',
        4: '../results/isic2017_avit_clahe_k4_SwinSeg_CNNprompt_adapt_20260728_2244/best.pth',
    },
}

CLAHE_FTL_CKPTS = {
    'SwinUnet': {
        0: '../results/isic2017_swinunet_clahe_ftl_k0_SwinUnet_20260730_0001/best.pth',
        1: '../results/isic2017_swinunet_clahe_ftl_k1_SwinUnet_20260730_1346/best.pth',
        2: '../results/isic2017_swinunet_clahe_ftl_k2_SwinUnet_20260730_1434/best.pth',
        3: '../results/isic2017_swinunet_clahe_ftl_k3_SwinUnet_20260730_1522/best.pth',
        4: '../results/isic2017_swinunet_clahe_ftl_k4_SwinUnet_20260730_1609/best.pth',
    },
    'AViT': {
        0: '../results/isic2017_avit_clahe_ftl_k0_SwinSeg_CNNprompt_adapt_20260730_1139/best.pth',
        1: '../results/isic2017_avit_clahe_ftl_k1_SwinSeg_CNNprompt_adapt_20260730_1656/best.pth',
        2: '../results/isic2017_avit_clahe_ftl_k2_SwinSeg_CNNprompt_adapt_20260730_1904/best.pth',
        3: '../results/isic2017_avit_clahe_ftl_k3_SwinSeg_CNNprompt_adapt_20260730_2012/best.pth',
        4: '../results/isic2017_avit_clahe_ftl_k4_SwinSeg_CNNprompt_adapt_20260730_2118/best.pth',
    },
}

NETWORK_MODEL_NAME = {'SwinUnet': 'SwinUnet', 'AViT': 'SwinSeg_CNNprompt_adapt'}

# old batch-pooled numbers (mean, std), for the before/after comparison table
OLD_BATCH_POOLED = {
    ('SwinUnet', 'baseline'):  (0.8254, 0.0045),
    ('SwinUnet', 'clahe'):     (0.8313, 0.0078),
    ('SwinUnet', 'clahe_ftl'): (0.8338, 0.0115),
    ('SwinUnet', 'tta_final'): (0.8352, 0.0077),
    ('AViT', 'baseline'):      (0.7514, 0.0089),
    ('AViT', 'clahe'):         (0.7622, 0.0096),
    ('AViT', 'clahe_ftl'):     (0.7721, 0.0070),
    ('AViT', 'tta_final'):     (0.7742, 0.0077),
}


def build_test_loader(config, image_subdir, dataset, meta_csv_name, fixed_test_csv_name):
    datas = Dataset_wrap_csv(k_fold='0', use_old_split=True, img_size=config.data.img_size,
                              dataset_name=dataset, split_ratio=config.data.split_ratio,
                              train_aug=False, data_folder=config.data.data_folder,
                              meta_csv_name=meta_csv_name, fixed_test_csv_name=fixed_test_csv_name,
                              image_subdir=image_subdir)
    test_data = datas['test']
    loader = torch.utils.data.DataLoader(test_data, batch_size=config.test.batch_size,
                                          shuffle=False, num_workers=config.test.num_workers,
                                          pin_memory=True, drop_last=False)
    return loader


def eval_per_image(model, model_name, loader, use_tta):
    rows = []
    for batch in loader:
        img = batch['image'].cuda().float()
        label = batch['label'].cuda().float()
        ids = batch['ID']

        with torch.no_grad():
            if use_tta:
                avg_prob, _ = tta_forward(model, model_name, img)
                prob = avg_prob
            else:
                logits = forward_logits(model, model_name, img)
                prob = torch.sigmoid(logits)

        pred = (prob.cpu().numpy() > 0.5)
        label_np = label.cpu().numpy() > 0.5

        for i in range(img.shape[0]):
            dice = metrics.dc(pred[i, 0], label_np[i, 0])
            rows.append({'image_id': ids[i], 'dice': dice})
    return rows


def summarize(df, network, stage):
    sub = df[(df['network'] == network) & (df['stage'] == stage)]
    fold_means = sub.groupby('fold')['dice'].mean()
    return fold_means.mean(), fold_means.std(ddof=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Per-image Dice recomputation of the full pipeline, both networks')
    parser.add_argument('--config_yml', type=str, default='Configs/multi_train_local.yml')
    parser.add_argument('--dataset', type=str, default='isic2017')
    parser.add_argument('--meta_csv_name', type=str, default='meta_isic2017_train2000.csv')
    parser.add_argument('--fixed_test_csv_name', type=str, default='meta_isic2017_test600.csv')
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = DotDict(yaml.load(open(args.config_yml), Loader=yaml.FullLoader))

    loader_raw = build_test_loader(config, 'Image', args.dataset, args.meta_csv_name, args.fixed_test_csv_name)
    loader_clahe = build_test_loader(config, 'Image_clahe', args.dataset, args.meta_csv_name, args.fixed_test_csv_name)
    print('Fixed test set loaded: {} samples (Image and Image_clahe)'.format(len(loader_raw.dataset)))

    all_rows = []

    # Stages 1-3: baseline, CLAHE, CLAHE+FTL, no TTA
    for network in ['SwinUnet', 'AViT']:
        model_name = NETWORK_MODEL_NAME[network]
        for stage, ckpts, loader in [
            ('baseline', BASELINE_CKPTS[network], loader_raw),
            ('clahe', CLAHE_CKPTS[network], loader_clahe),
            ('clahe_ftl', CLAHE_FTL_CKPTS[network], loader_clahe),
        ]:
            for fold, ckpt_path in ckpts.items():
                print('=== {} / {} / fold k{}: {} ==='.format(network, stage, fold, ckpt_path))
                model = build_model(model_name, config)
                model.load_state_dict(torch.load(ckpt_path))
                model.eval()
                rows = eval_per_image(model, model_name, loader, use_tta=False)
                for r in rows:
                    r.update({'network': network, 'stage': stage, 'fold': fold})
                all_rows.extend(rows)
                del model
                torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)

    # decide stage-3/4 winner per network: CLAHE vs CLAHE+FTL, per-image mean
    winners = {}
    for network in ['SwinUnet', 'AViT']:
        clahe_mean, _ = summarize(df, network, 'clahe')
        ftl_mean, _ = summarize(df, network, 'clahe_ftl')
        winner_stage = 'clahe_ftl' if ftl_mean > clahe_mean else 'clahe'
        winners[network] = (winner_stage, CLAHE_FTL_CKPTS[network] if winner_stage == 'clahe_ftl' else CLAHE_CKPTS[network])
        print('{}: CLAHE per-image mean={:.4f}, CLAHE+FTL per-image mean={:.4f} -> stage-3/4 winner: {}'.format(
            network, clahe_mean, ftl_mean, winner_stage))

    # Stage 4: TTA on top of the winning checkpoint
    tta_rows = []
    for network in ['SwinUnet', 'AViT']:
        model_name = NETWORK_MODEL_NAME[network]
        winner_stage, winner_ckpts = winners[network]
        for fold, ckpt_path in winner_ckpts.items():
            print('=== {} / tta_final (base={}) / fold k{}: {} ==='.format(network, winner_stage, fold, ckpt_path))
            model = build_model(model_name, config)
            model.load_state_dict(torch.load(ckpt_path))
            model.eval()
            rows = eval_per_image(model, model_name, loader_clahe, use_tta=True)
            for r in rows:
                r.update({'network': network, 'stage': 'tta_final', 'fold': fold})
            tta_rows.extend(rows)
            del model
            torch.cuda.empty_cache()

    df = pd.concat([df, pd.DataFrame(tta_rows)], ignore_index=True)

    per_image_csv = os.path.join(args.output_dir, 'per_image_final_pipeline.csv')
    df.to_csv(per_image_csv, index=False)
    print('Saved per-image results: {}'.format(per_image_csv))

    # before/after comparison table
    comparison_rows = []
    for network in ['SwinUnet', 'AViT']:
        for stage in ['baseline', 'clahe', 'clahe_ftl', 'tta_final']:
            new_mean, new_std = summarize(df, network, stage)
            old_mean, old_std = OLD_BATCH_POOLED[(network, stage)]
            comparison_rows.append({
                'network': network, 'stage': stage,
                'old_batch_pooled_mean': old_mean, 'old_batch_pooled_std': old_std,
                'new_per_image_mean': new_mean, 'new_per_image_std': new_std,
                'shift': new_mean - old_mean,
            })
    comp_df = pd.DataFrame(comparison_rows)
    comp_csv = os.path.join(args.output_dir, 'before_after_comparison.csv')
    comp_df.to_csv(comp_csv, index=False)

    print('========================================================================================')
    print('BEFORE/AFTER COMPARISON (old batch-pooled vs new per-image)')
    for _, r in comp_df.iterrows():
        print('{} / {}: old={:.4f}+/-{:.4f}  new={:.4f}+/-{:.4f}  shift={:+.4f}'.format(
            r['network'], r['stage'], r['old_batch_pooled_mean'], r['old_batch_pooled_std'],
            r['new_per_image_mean'], r['new_per_image_std'], r['shift']))
    print('Saved comparison: {}'.format(comp_csv))
    print('PIPELINE COMPLETE')
