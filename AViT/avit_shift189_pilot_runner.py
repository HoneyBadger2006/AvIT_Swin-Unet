"""
Priority 2: pilot retrain (single fold, no CV -- allin) using AViT's own 189
seed42 Dice<0.7 training images, augmented with the shift-only 5-technique set
(hflip, vflip, rot90, rot270, 10% rightward shift + blurred reflect-101) --
explicitly NO mask-isolate (Variant A) or zoom-crop (Variant B/C/D) variants,
per instruction. 2000 + 924 = 2924 training images total (168/189 images got
the shift technique; 21 were skipped, lesion too close to the right edge --
see shift_skip_report_avit189.csv).

Kept methodologically symmetric with SwinUnet's own pipeline throughout: plain
CLAHE, no Focal Tversky Loss, for the identification run, this pilot retrain,
AND the baseline-comparison reference (AViT's OWN established hard test images
= network=AViT, stage='clahe' -- NOT 'clahe_ftl' -- fold=0, dice<0.7 in
per_image_final_pipeline.csv, 145 images), so this is a clean "does the
augmentation help AViT" test isolated from any loss-function difference.

Reports both recovery metrics: how many of AViT's 189 originally-bad TRAINING
images are still <0.7 Dice after retrain, and how many of AViT's own 145
established-hard TEST images are still <0.7 Dice after retrain, plus the same
full-test600/hard-subset comparison format used for every other pilot in this
project.
"""
import json
import os
import subprocess
import sys
import time
import traceback

AVIT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(AVIT_DIR)

STATUS_PATH = '../per_image_analysis_v2/overnight_run/avit_shift189_pilot_status.json'
STEP_LOG_DIR = '../kfold_logs/avit_shift189_pilot_steps'
os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
os.makedirs(STEP_LOG_DIR, exist_ok=True)

PY = '../venv/Scripts/python.exe'
MODEL_NAME = 'SwinSeg_CNNprompt_adapt'  # AViT
BAD189_CSV = '../per_image_analysis_v2/bad_image_augmentation/bad_images_avit_seed42_dice189.csv'
FINAL_PIPELINE_CSV = '../per_image_analysis_v2/final_pipeline/per_image_final_pipeline.csv'

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
        exp_name = 'isic2017_avit_clahe_dice189_shift5'
        train_cmd = [
            PY, '-u', 'multi_train_adapt.py',
            '--exp_name', exp_name, '--config_yml', 'Configs/multi_train_local.yml',
            '--model', MODEL_NAME, '--batch_size', '16', '--dataset', 'isic2017', '--k_fold', 'allin',
            '--num_epochs', '30',
            '--meta_csv_name', 'meta_isic2017_train2000_avit_dice189_shift5.csv',
            '--fixed_test_csv_name', 'meta_isic2017_test600.csv',
            '--image_subdir', 'Image_clahe',
            '--seed', '42',
        ]
        run_step('p_train', train_cmd, os.path.join(STEP_LOG_DIR, 'train.log'))

        exp_dir = find_exp_dir('{}_{}_'.format(exp_name, MODEL_NAME))
        final_ckpt = os.path.join(exp_dir, 'final.pth')
        if not os.path.exists(final_ckpt):
            raise RuntimeError('final.pth not found in {}'.format(exp_dir))
        log('Training complete. exp_dir={} final_ckpt={}'.format(exp_dir, final_ckpt))

        train189_csv = '../per_image_analysis_v2/bad_image_augmentation/avit_shift189_train_recovery.csv'
        run_step('p_eval_train189', [
            PY, 'eval_shift37_train_recovery.py', '--ckpt', final_ckpt, '--model_name', MODEL_NAME,
            '--image_subdir', 'Image_clahe', '--bad_csv', BAD189_CSV, '--expected_n', '189',
            '--out_csv', train189_csv,
        ], os.path.join(STEP_LOG_DIR, 'eval_train189.log'))

        decision_json = '../per_image_analysis_v2/overnight_run/avit_shift189_test600_decision.json'
        test600_csv = '../per_image_analysis_v2/overnight_run/avit_shift189_test600_per_image.csv'
        run_step('p_eval_test600', [
            PY, 'eval_pilot_vs_baseline.py',
            '--pilot_ckpt', final_ckpt, '--model_name', MODEL_NAME, '--network', 'AViT',
            '--baseline_stage', 'clahe', '--fold', '0', '--image_subdir', 'Image_clahe',
            '--out_json', decision_json, '--out_csv', test600_csv,
        ], os.path.join(STEP_LOG_DIR, 'eval_test600.log'))

        import pandas as pd
        train189_df = pd.read_csv(train189_csv, dtype={'image_id': str})
        n_train189_still_bad = int((train189_df['retrain_dice'] < 0.7).sum())

        test_df = pd.read_csv(test600_csv, dtype={'image_id': str})
        base_full = pd.read_csv(FINAL_PIPELINE_CSV, dtype={'image_id': str})
        base_sub = base_full[(base_full['network'] == 'AViT') & (base_full['stage'] == 'clahe') & (base_full['fold'] == 0)][
            ['image_id', 'dice']].rename(columns={'dice': 'baseline_dice'})
        merged145 = pd.merge(test_df, base_sub, on='image_id', how='inner')
        bad145 = merged145[merged145['baseline_dice'] < 0.7]
        n_test145_still_bad = int((bad145['pilot_dice'] < 0.7).sum())

        with open(decision_json) as f:
            decision_result = json.load(f)

        summary = {
            'exp_dir': exp_dir, 'final_ckpt': final_ckpt,
            'train189_still_bad': n_train189_still_bad, 'train189_total': 189,
            'train189_mean_retrain_dice': float(train189_df['retrain_dice'].mean()),
            'train189_mean_orig_dice': float(train189_df['orig_dice'].mean()),
            'test145_still_bad': n_test145_still_bad, 'test145_total': int(len(bad145)),
            'test145_mean_pilot_dice': float(bad145['pilot_dice'].mean()),
            'test145_mean_baseline_dice': float(bad145['baseline_dice'].mean()),
            'full_test600': decision_result['full_test600'],
            'bad_subset_paired_test': decision_result['bad_subset'],
            'decision': decision_result['decision'],
        }
        summary_path = '../per_image_analysis_v2/overnight_run/avit_shift189_pilot_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        log('PILOT SUMMARY: train189_still_bad={}/189  test145_still_bad={}/{}  full_test600 mean_diff={:+.4f} p={:.4g}  hard_subset mean_diff={:+.4f} p={:.4g}'.format(
            n_train189_still_bad, n_test145_still_bad, len(bad145),
            summary['full_test600']['mean_diff'], summary['full_test600']['p_value_one_sided'],
            summary['bad_subset_paired_test']['mean_diff'], summary['bad_subset_paired_test']['p_value_one_sided']))
        log('Saved: {}'.format(summary_path))
        status['summary'] = summary
        status['final'] = 'COMPLETE'
        save_status()
        log('AVIT_SHIFT189_PILOT_RUNNER_COMPLETE')
    except Exception as e:
        status['final'] = 'FAILED'
        status['error'] = str(e)
        status['traceback'] = traceback.format_exc()
        save_status()
        log('AVIT_SHIFT189_PILOT_RUNNER_FAILED: {}'.format(e))
        log(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
