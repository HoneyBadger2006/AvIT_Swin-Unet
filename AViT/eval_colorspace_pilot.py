"""
Evaluates an HSV or YCbCr color-space pilot checkpoint (SwinUnet, no CLAHE, true
fold-0 CV) against the existing RAW RGB baseline (network=SwinUnet, stage='baseline',
fold=0 in per_image_final_pipeline.csv) on two things:
  1. the full 600-image fixed test set
  2. the SAME 73 hard test images used throughout this project's CLAHE/augmentation
     investigation (network=SwinUnet, stage='clahe', fold=0, dice<0.7 in
     per_image_final_pipeline.csv) -- Prof. Samavi's stated interest is whether color
     space alone helps specifically on the cases already known to be difficult, not
     a fresh hard-subset defined by the raw baseline's own failures.

Paired one-sided t-test (pilot > raw baseline) on both populations, matching the
reporting format used for the augmentation pilots.
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats
import medpy.metric.binary as metrics

from Datasets.create_dataset import Dataset_wrap_csv
from Utils.pieces import DotDict
from tta_inference import build_model, forward_logits

FINAL_PIPELINE_CSV = '../per_image_analysis_v2/final_pipeline/per_image_final_pipeline.csv'


def eval_checkpoint_per_image(ckpt, model_name, image_subdir, config, norm_mean=None, norm_std=None):
    datas = Dataset_wrap_csv(k_fold='0', use_old_split=True, img_size=config.data.img_size,
                              dataset_name='isic2017', split_ratio=config.data.split_ratio,
                              train_aug=False, data_folder=config.data.data_folder,
                              meta_csv_name='meta_isic2017_train2000.csv',
                              fixed_test_csv_name='meta_isic2017_test600.csv',
                              image_subdir=image_subdir, norm_mean=norm_mean, norm_std=norm_std)
    loader = torch.utils.data.DataLoader(datas['test'], batch_size=config.test.batch_size, shuffle=False,
                                          num_workers=config.test.num_workers, pin_memory=True, drop_last=False)
    model = build_model(model_name, config)
    model.load_state_dict(torch.load(ckpt))
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in loader:
            img = batch['image'].cuda().float()
            label = batch['label'].cuda().float()
            ids = batch['ID']
            logits = forward_logits(model, model_name, img)
            prob = torch.sigmoid(logits)
            pred = (prob.cpu().numpy() > 0.5)
            label_np = label.cpu().numpy() > 0.5
            for i in range(img.shape[0]):
                dice = metrics.dc(pred[i, 0], label_np[i, 0])
                rows.append({'image_id': ids[i], 'dice': dice})
    return pd.DataFrame(rows)


def paired_report(pilot_dice, baseline_dice, label):
    diff = pilot_dice - baseline_dice
    t_stat, p_two = stats.ttest_rel(pilot_dice, baseline_dice)
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    report = {
        'label': label, 'n': len(diff),
        'pilot_mean': float(np.mean(pilot_dice)), 'baseline_mean': float(np.mean(baseline_dice)),
        'mean_diff': float(np.mean(diff)), 'std_diff': float(np.std(diff, ddof=1)),
        'pct_improved': float(np.mean(diff > 0)), 'pct_worsened': float(np.mean(diff < 0)),
        't_stat': float(t_stat), 'p_value_one_sided': float(p_one),
    }
    print('[{}] n={} pilot={:.4f} raw_baseline={:.4f} mean_diff={:+.4f} p_one_sided={:.4g} ({}% improved / {}% worsened)'.format(
        label, report['n'], report['pilot_mean'], report['baseline_mean'], report['mean_diff'],
        report['p_value_one_sided'], round(100 * report['pct_improved']), round(100 * report['pct_worsened'])))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--model_name', default='SwinUnet')
    p.add_argument('--image_subdir', required=True, help="'Image_hsv' or 'Image_ycbcr'")
    p.add_argument('--colorspace_name', required=True)
    p.add_argument('--out_json', required=True)
    p.add_argument('--out_csv', required=True)
    p.add_argument('--config_yml', default='Configs/multi_train_local.yml')
    p.add_argument('--norm_mean', type=float, nargs=3, default=None)
    p.add_argument('--norm_std', type=float, nargs=3, default=None)
    args = p.parse_args()

    config = DotDict(yaml.load(open(args.config_yml), Loader=yaml.FullLoader))

    base_full = pd.read_csv(FINAL_PIPELINE_CSV, dtype={'image_id': str})
    raw_base_sub = base_full[(base_full['network'] == 'SwinUnet') & (base_full['stage'] == 'baseline') & (base_full['fold'] == 0)][
        ['image_id', 'dice']].rename(columns={'dice': 'raw_baseline_dice'})
    print('Loaded {} raw-baseline per-image test600 rows'.format(len(raw_base_sub)))

    clahe_sub = base_full[(base_full['network'] == 'SwinUnet') & (base_full['stage'] == 'clahe') & (base_full['fold'] == 0)]
    hard73_ids = set(clahe_sub[clahe_sub['dice'] < 0.7]['image_id'])
    print('Existing hard-subset (CLAHE fold-0, dice<0.7): {} images'.format(len(hard73_ids)))
    assert len(hard73_ids) == 73

    pilot_df = eval_checkpoint_per_image(args.ckpt, args.model_name, args.image_subdir, config,
                                          norm_mean=args.norm_mean, norm_std=args.norm_std)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    pilot_df.rename(columns={'dice': 'pilot_dice'}).to_csv(args.out_csv, index=False)

    merged = pd.merge(pilot_df.rename(columns={'dice': 'pilot_dice'}), raw_base_sub, on='image_id', how='inner')
    assert len(merged) == 600, 'expected 600 matched images, got {}'.format(len(merged))

    full_report = paired_report(merged['pilot_dice'].values, merged['raw_baseline_dice'].values, 'full_test600')

    hard_sub = merged[merged['image_id'].isin(hard73_ids)]
    assert len(hard_sub) == 73, 'expected 73 matched hard-subset images, got {}'.format(len(hard_sub))
    hard_report = paired_report(hard_sub['pilot_dice'].values, hard_sub['raw_baseline_dice'].values, 'existing_hard73_subset')

    result = {
        'colorspace': args.colorspace_name, 'ckpt': args.ckpt, 'image_subdir': args.image_subdir,
        'full_test600': full_report, 'existing_hard73_subset': hard_report,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print('Saved: {}'.format(args.out_json))
    print('COLORSPACE_EVAL_DONE')


if __name__ == '__main__':
    main()
