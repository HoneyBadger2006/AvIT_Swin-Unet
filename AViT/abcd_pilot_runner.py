"""
Runs the A+B+C+D pilot (8 augmentation variants per bad training image: 4 standard
flips/rotations + Variant A tight-mask + Variant B tight-zoom + Variant C
expanded-bbox zoom + Variant D 70%-fill zoom) for the same 23 bad training images
used in the already-successful A+B-only corrected pilot, and compares three
conditions directly:
  1. baseline (no augmentation) -- existing SwinUnet+CLAHE fold-0 checkpoint
  2. A+B-only pilot (already run: results/isic2017_swinunet_clahe_pilot_badaug23_...,
     summary in per_image_analysis_v2/overnight_run/priority1_summary.json)
  3. A+B+C+D pilot (this run)

Same training setup as the successful A+B pilot otherwise: SwinUnet, CLAHE, no
fold splitting (k_fold='allin', all 2000 original images + augmented rows, single
run, final-epoch checkpoint), 30 epochs, same hyperparameters. Evaluated against
the same fold-0 CLAHE baseline stage in per_image_final_pipeline.csv, so all three
conditions are directly comparable on the same 600-image fixed test set.
"""
import json
import os
import subprocess
import sys
import time
import traceback

AVIT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(AVIT_DIR)

STATUS_PATH = '../per_image_analysis_v2/overnight_run/abcd_pilot_status.json'
STEP_LOG_DIR = '../kfold_logs/abcd_pilot_steps'
os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
os.makedirs(STEP_LOG_DIR, exist_ok=True)

PY = '../venv/Scripts/python.exe'
PRIORITY1_SUMMARY = '../per_image_analysis_v2/overnight_run/priority1_summary.json'
FINAL_PIPELINE_CSV = '../per_image_analysis_v2/final_pipeline/per_image_final_pipeline.csv'

status = {'stages': [], 'current': None, 'started_at': time.strftime('%Y-%m-%d %H:%M:%S')}


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
        exp_name = 'isic2017_swinunet_clahe_pilot_badaug23_abcd'
        train_cmd = [
            PY, '-u', 'multi_train_adapt.py',
            '--exp_name', exp_name, '--config_yml', 'Configs/multi_train_local.yml',
            '--model', 'SwinUnet', '--batch_size', '16', '--dataset', 'isic2017', '--k_fold', 'allin',
            '--num_epochs', '30',
            '--meta_csv_name', 'meta_isic2017_train2000_pilot_badaug23_abcd.csv',
            '--fixed_test_csv_name', 'meta_isic2017_test600.csv',
            '--image_subdir', 'Image_clahe',
        ]
        run_step('abcd_train', train_cmd, os.path.join(STEP_LOG_DIR, 'abcd_train.log'))

        exp_dir = find_exp_dir('{}_SwinUnet_'.format(exp_name))
        final_ckpt = os.path.join(exp_dir, 'final.pth')
        if not os.path.exists(final_ckpt):
            raise RuntimeError('final.pth not found in {}'.format(exp_dir))
        log('Training complete. exp_dir={} final_ckpt={}'.format(exp_dir, final_ckpt))

        train23_csv = '../per_image_analysis_v2/bad_image_augmentation/pilot_retrain_abcd_train23_recovery.csv'
        run_step('abcd_eval_train23', [
            PY, 'eval_train23_recovery.py', '--ckpt', final_ckpt, '--model_name', 'SwinUnet',
            '--image_subdir', 'Image_clahe', '--out_csv', train23_csv,
        ], os.path.join(STEP_LOG_DIR, 'abcd_eval_train23.log'))

        decision_json = '../per_image_analysis_v2/overnight_run/pilot_retrain_abcd_fold0_decision.json'
        test600_per_image_csv = '../per_image_analysis_v2/overnight_run/pilot_retrain_abcd_test600_per_image.csv'
        run_step('abcd_eval_test600', [
            PY, 'eval_pilot_vs_baseline.py',
            '--pilot_ckpt', final_ckpt, '--model_name', 'SwinUnet', '--network', 'SwinUnet',
            '--baseline_stage', 'clahe', '--fold', '0', '--image_subdir', 'Image_clahe',
            '--out_json', decision_json, '--out_csv', test600_per_image_csv,
        ], os.path.join(STEP_LOG_DIR, 'abcd_eval_test600.log'))

        import pandas as pd
        train23_df = pd.read_csv(train23_csv, dtype={'image_id': str})
        n_train23_still_bad = int((train23_df['retrain_dice'] < 0.7).sum())

        test_df = pd.read_csv(test600_per_image_csv, dtype={'image_id': str})
        base_full = pd.read_csv(FINAL_PIPELINE_CSV, dtype={'image_id': str})
        base_sub = base_full[(base_full['network'] == 'SwinUnet') & (base_full['stage'] == 'clahe') & (base_full['fold'] == 0)][
            ['image_id', 'dice']].rename(columns={'dice': 'baseline_dice'})
        merged73 = pd.merge(test_df, base_sub, on='image_id', how='inner')
        bad73 = merged73[merged73['baseline_dice'] < 0.7]
        n_test73_still_bad = int((bad73['pilot_dice'] < 0.7).sum())

        with open(decision_json) as f:
            decision_result = json.load(f)

        abcd_summary = {
            'exp_dir': exp_dir, 'final_ckpt': final_ckpt,
            'train23_still_bad': n_train23_still_bad, 'train23_total': 23,
            'train23_mean_retrain_dice': float(train23_df['retrain_dice'].mean()),
            'train23_mean_orig_dice': float(train23_df['orig_allin_dice'].mean()),
            'test73_still_bad': n_test73_still_bad, 'test73_total': int(len(bad73)),
            'test73_mean_pilot_dice': float(bad73['pilot_dice'].mean()),
            'test73_mean_baseline_dice': float(bad73['baseline_dice'].mean()),
            'full_test600': decision_result['full_test600'],
            'bad_subset_paired_test': decision_result['bad_subset'],
            'decision': decision_result['decision'],
        }

        # three-way comparison table
        with open(PRIORITY1_SUMMARY) as f:
            ab_summary = json.load(f)

        comparison = {
            'baseline': {
                'full_test600_mean': ab_summary['full_test600']['baseline_mean'],
                'hard_subset_mean': ab_summary['bad_subset_paired_test']['baseline_mean'],
            },
            'A+B_only_pilot': {
                'full_test600_mean': ab_summary['full_test600']['pilot_mean'],
                'full_test600_diff_vs_baseline': ab_summary['full_test600']['mean_diff'],
                'full_test600_p': ab_summary['full_test600']['p_value_one_sided'],
                'hard_subset_mean': ab_summary['bad_subset_paired_test']['pilot_mean'],
                'hard_subset_diff_vs_baseline': ab_summary['bad_subset_paired_test']['mean_diff'],
                'hard_subset_p': ab_summary['bad_subset_paired_test']['p_value_one_sided'],
                'train23_still_bad': ab_summary['train23_still_bad'],
                'test73_still_bad': ab_summary['test73_still_bad'],
            },
            'A+B+C+D_pilot': {
                'full_test600_mean': abcd_summary['full_test600']['pilot_mean'],
                'full_test600_diff_vs_baseline': abcd_summary['full_test600']['mean_diff'],
                'full_test600_p': abcd_summary['full_test600']['p_value_one_sided'],
                'hard_subset_mean': abcd_summary['bad_subset_paired_test']['pilot_mean'],
                'hard_subset_diff_vs_baseline': abcd_summary['bad_subset_paired_test']['mean_diff'],
                'hard_subset_p': abcd_summary['bad_subset_paired_test']['p_value_one_sided'],
                'train23_still_bad': abcd_summary['train23_still_bad'],
                'test73_still_bad': abcd_summary['test73_still_bad'],
            },
            'ABCD_vs_AB_full_test600_delta': abcd_summary['full_test600']['pilot_mean'] - ab_summary['full_test600']['pilot_mean'],
            'ABCD_vs_AB_hard_subset_delta': abcd_summary['bad_subset_paired_test']['pilot_mean'] - ab_summary['bad_subset_paired_test']['pilot_mean'],
        }

        summary_path = '../per_image_analysis_v2/overnight_run/abcd_pilot_summary.json'
        with open(summary_path, 'w') as f:
            json.dump({'abcd_summary': abcd_summary, 'three_way_comparison': comparison}, f, indent=2)

        log('ABCD PILOT SUMMARY: train23_still_bad={}/23  test73_still_bad={}/{}  full_test600 mean_diff={:+.4f} (AB was {:+.4f})  hard_subset mean_diff={:+.4f} (AB was {:+.4f})'.format(
            n_train23_still_bad, n_test73_still_bad, len(bad73),
            abcd_summary['full_test600']['mean_diff'], ab_summary['full_test600']['mean_diff'],
            abcd_summary['bad_subset_paired_test']['mean_diff'], ab_summary['bad_subset_paired_test']['mean_diff']))
        log('Saved: {}'.format(summary_path))
        status['abcd_summary'] = abcd_summary
        status['three_way_comparison'] = comparison
        save_status()

        status['final'] = 'COMPLETE'
        save_status()
        log('ABCD_PILOT_RUNNER_COMPLETE')
    except Exception as e:
        status['final'] = 'FAILED'
        status['error'] = str(e)
        status['traceback'] = traceback.format_exc()
        save_status()
        log('ABCD_PILOT_RUNNER_FAILED: {}'.format(e))
        log(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
