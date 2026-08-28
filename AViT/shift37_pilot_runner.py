"""
Pilot retrain (single fold, no CV -- allin) using the 37 seed42 Dice<0.7 training
images augmented with the 5 named techniques (hflip, vflip, rot90, rot270, 10%
rightward shift with blurred reflect-101 padding), per Prof. Samavi's revised
spec. 2000 + 37x5 = 2185 training images total. Reports both recovery metrics:
  - how many of the 37 originally-bad TRAINING images are still <0.7 Dice
  - how many of the already-established 73 hard TEST images (SwinUnet/clahe/
    fold0 baseline, dice<0.7 on the fixed test600 set) are still <0.7 Dice
Also reports the full test600 vs. hard-subset comparison, same format as every
other pilot in this project, so this is directly comparable to the earlier
augmentation pilots.
"""
import json
import os
import subprocess
import sys
import time
import traceback

AVIT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(AVIT_DIR)

STATUS_PATH = '../per_image_analysis_v2/overnight_run/shift37_pilot_status.json'
STEP_LOG_DIR = '../kfold_logs/shift37_pilot_steps'
os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
os.makedirs(STEP_LOG_DIR, exist_ok=True)

PY = '../venv/Scripts/python.exe'

status = {'current': None, 'started_at': time.strftime('%Y-%m-%d %H:%M:%S')}


def save_status():
    with open(STATUS_PATH, 'w') as f:
        json.dump(status, f, indent=2)


def log(msg):
    line = '[{}] {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), msg)
    print(line, flush=True)


def run_step(name, cmd, log_file):
    log('RUN: {} -> {}'.format(name, ' '.join(cmd)))
    status['current'] = name
    save_status()
    with open(log_file, 'w') as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=AVIT_DIR)
    if proc.returncode != 0:
        raise RuntimeError('Step {} failed (exit {}). See {}'.format(name, proc.returncode, log_file))
    log('OK: {}'.format(name))


def find_exp_dir(prefix):
    candidates = [d for d in os.listdir('../results') if d.startswith(prefix)]
    if not candidates:
        raise RuntimeError('No results dir found with prefix {}'.format(prefix))
    candidates.sort()
    return '../results/' + candidates[-1]


def main():
    try:
        exp_name = 'isic2017_swinunet_clahe_seed42_dice37_shift5'
        train_cmd = [
            PY, '-u', 'multi_train_adapt.py',
            '--exp_name', exp_name, '--config_yml', 'Configs/multi_train_local.yml',
            '--model', 'SwinUnet', '--batch_size', '16', '--dataset', 'isic2017', '--k_fold', 'allin',
            '--num_epochs', '30',
            '--meta_csv_name', 'meta_isic2017_train2000_seed42_dice37_shift5.csv',
            '--fixed_test_csv_name', 'meta_isic2017_test600.csv',
            '--image_subdir', 'Image_clahe',
        ]
        run_step('p_train', train_cmd, os.path.join(STEP_LOG_DIR, 'train.log'))

        exp_dir = find_exp_dir('{}_SwinUnet_'.format(exp_name))
        final_ckpt = os.path.join(exp_dir, 'final.pth')
        if not os.path.exists(final_ckpt):
            raise RuntimeError('final.pth not found in {}'.format(exp_dir))
        log('Training complete. exp_dir={} final_ckpt={}'.format(exp_dir, final_ckpt))

        train37_csv = '../per_image_analysis_v2/bad_image_augmentation/shift37_train_recovery.csv'
        run_step('p_eval_train37', [
            PY, 'eval_shift37_train_recovery.py', '--ckpt', final_ckpt, '--model_name', 'SwinUnet',
            '--image_subdir', 'Image_clahe', '--out_csv', train37_csv,
        ], os.path.join(STEP_LOG_DIR, 'eval_train37.log'))

        decision_json = '../per_image_analysis_v2/overnight_run/shift37_test600_decision.json'
        test600_csv = '../per_image_analysis_v2/overnight_run/shift37_test600_per_image.csv'
        run_step('p_eval_test600', [
            PY, 'eval_pilot_vs_baseline.py',
            '--pilot_ckpt', final_ckpt, '--model_name', 'SwinUnet', '--network', 'SwinUnet',
            '--baseline_stage', 'clahe', '--fold', '0', '--image_subdir', 'Image_clahe',
            '--out_json', decision_json, '--out_csv', test600_csv,
        ], os.path.join(STEP_LOG_DIR, 'eval_test600.log'))

        import pandas as pd
        train37_df = pd.read_csv(train37_csv, dtype={'image_id': str})
        n_train37_still_bad = int((train37_df['retrain_dice'] < 0.7).sum())

        test_df = pd.read_csv(test600_csv, dtype={'image_id': str})
        base_full = pd.read_csv('../per_image_analysis_v2/final_pipeline/per_image_final_pipeline.csv', dtype={'image_id': str})
        base_sub = base_full[(base_full['network'] == 'SwinUnet') & (base_full['stage'] == 'clahe') & (base_full['fold'] == 0)][
            ['image_id', 'dice']].rename(columns={'dice': 'baseline_dice'})
        merged73 = pd.merge(test_df, base_sub, on='image_id', how='inner')
        bad73 = merged73[merged73['baseline_dice'] < 0.7]
        n_test73_still_bad = int((bad73['pilot_dice'] < 0.7).sum())

        with open(decision_json) as f:
            decision_result = json.load(f)

        summary = {
            'exp_dir': exp_dir, 'final_ckpt': final_ckpt,
            'train37_still_bad': n_train37_still_bad, 'train37_total': 37,
            'train37_mean_retrain_dice': float(train37_df['retrain_dice'].mean()),
            'train37_mean_orig_dice': float(train37_df['orig_dice'].mean()),
            'test73_still_bad': n_test73_still_bad, 'test73_total': int(len(bad73)),
            'test73_mean_pilot_dice': float(bad73['pilot_dice'].mean()),
            'test73_mean_baseline_dice': float(bad73['baseline_dice'].mean()),
            'full_test600': decision_result['full_test600'],
            'bad_subset_paired_test': decision_result['bad_subset'],
            'decision': decision_result['decision'],
        }
        summary_path = '../per_image_analysis_v2/overnight_run/shift37_pilot_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        log('PILOT SUMMARY: train37_still_bad={}/37  test73_still_bad={}/{}  full_test600 mean_diff={:+.4f} p={:.4g}  hard_subset mean_diff={:+.4f} p={:.4g}'.format(
            n_train37_still_bad, n_test73_still_bad, len(bad73),
            summary['full_test600']['mean_diff'], summary['full_test600']['p_value_one_sided'],
            summary['bad_subset_paired_test']['mean_diff'], summary['bad_subset_paired_test']['p_value_one_sided']))
        log('Saved: {}'.format(summary_path))
        status['summary'] = summary
        status['final'] = 'COMPLETE'
        save_status()
        log('SHIFT37_PILOT_RUNNER_COMPLETE')
    except Exception as e:
        status['final'] = 'FAILED'
        status['error'] = str(e)
        status['traceback'] = traceback.format_exc()
        save_status()
        log('SHIFT37_PILOT_RUNNER_FAILED: {}'.format(e))
        log(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
