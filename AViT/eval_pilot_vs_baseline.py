"""
Evaluates a pilot (bad-image-augmented) checkpoint against the existing baseline
checkpoint's ALREADY-COMPUTED per-image test600 results (from
per_image_analysis_v2/final_pipeline/per_image_final_pipeline.csv), per-image, on:
  - the full 600-image fixed test set
  - the "bad-image subset" of the test set = images where the BASELINE checkpoint's
    own per-image dice < 0.7 (i.e. hard cases for the model this pilot is trying to
    improve on)

Uses a paired one-sided t-test (pilot > baseline) on the per-image dice differences
to judge whether an improvement is a real signal vs noise, not just eyeballing the
mean delta. Writes a decision JSON consumed by the overnight orchestrator:
  decision = 'continue' if full-set mean_diff > MIN_MEAN_DIFF and one-sided p < ALPHA,
             else 'stop'.
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

MIN_MEAN_DIFF = 0.003
ALPHA = 0.05
FINAL_PIPELINE_CSV = '../per_image_analysis_v2/final_pipeline/per_image_final_pipeline.csv'


def eval_checkpoint_per_image(ckpt, model_name, image_subdir, config):
    datas = Dataset_wrap_csv(k_fold='0', use_old_split=True, img_size=config.data.img_size,
                              dataset_name='isic2017', split_ratio=config.data.split_ratio,
                              train_aug=False, data_folder=config.data.data_folder,
                              meta_csv_name='meta_isic2017_train2000.csv',
                              fixed_test_csv_name='meta_isic2017_test600.csv',
                              image_subdir=image_subdir)
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
    print('[{}] n={} pilot={:.4f} baseline={:.4f} mean_diff={:+.4f} p_one_sided={:.4g} ({}% improved / {}% worsened)'.format(
        label, report['n'], report['pilot_mean'], report['baseline_mean'], report['mean_diff'],
        report['p_value_one_sided'], round(100 * report['pct_improved']), round(100 * report['pct_worsened'])))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pilot_ckpt', required=True)
    p.add_argument('--model_name', required=True)
    p.add_argument('--network', required=True, help="'SwinUnet' or 'AViT', for filtering the baseline CSV")
    p.add_argument('--baseline_stage', required=True, help="'clahe' or 'clahe_ftl'")
    p.add_argument('--fold', type=int, required=True)
    p.add_argument('--image_subdir', default='Image_clahe')
    p.add_argument('--out_json', required=True)
    p.add_argument('--out_csv', required=True)
    p.add_argument('--config_yml', default='Configs/multi_train_local.yml')
    args = p.parse_args()

    config = DotDict(yaml.load(open(args.config_yml), Loader=yaml.FullLoader))

    base_df = pd.read_csv(FINAL_PIPELINE_CSV, dtype={'image_id': str})
    base_sub = base_df[(base_df['network'] == args.network) & (base_df['stage'] == args.baseline_stage)
                        & (base_df['fold'] == args.fold)][['image_id', 'dice']].rename(columns={'dice': 'baseline_dice'})
    print('Loaded {} baseline per-image test600 rows for {}/{}/fold{}'.format(
        len(base_sub), args.network, args.baseline_stage, args.fold))

    pilot_df = eval_checkpoint_per_image(args.pilot_ckpt, args.model_name, args.image_subdir, config)
    pilot_df = pilot_df.rename(columns={'dice': 'pilot_dice'})
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    pilot_df.to_csv(args.out_csv, index=False)

    merged = pd.merge(pilot_df, base_sub, on='image_id', how='inner')
    assert len(merged) == 600, 'expected 600 matched images, got {}'.format(len(merged))

    full_report = paired_report(merged['pilot_dice'].values, merged['baseline_dice'].values, 'full_test600')

    bad_sub = merged[merged['baseline_dice'] < 0.7]
    if len(bad_sub) >= 3:
        bad_report = paired_report(bad_sub['pilot_dice'].values, bad_sub['baseline_dice'].values, 'bad_subset')
    else:
        bad_report = {'label': 'bad_subset', 'n': len(bad_sub), 'note': 'too few images for a meaningful paired test'}
        print('[bad_subset] only {} images, skipping paired test'.format(len(bad_sub)))

    decision = 'continue' if (full_report['mean_diff'] > MIN_MEAN_DIFF and full_report['p_value_one_sided'] < ALPHA) else 'stop'
    result = {
        'network': args.network, 'fold': args.fold, 'pilot_ckpt': args.pilot_ckpt,
        'baseline_stage': args.baseline_stage, 'decision_criterion': 'mean_diff > {} and p_one_sided < {}'.format(MIN_MEAN_DIFF, ALPHA),
        'full_test600': full_report, 'bad_subset': bad_report, 'decision': decision,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print('Decision: {}'.format(decision.upper()))
    print('Saved: {}'.format(args.out_json))
    print('EVAL_DONE')


if __name__ == '__main__':
    main()
