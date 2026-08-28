"""
Step 3 of the data-integrity audit: an independent, genuinely NEW training run
(fresh checkpoint, not a re-check of the existing isic2017_swinunet_clahe_allin_*
checkpoint) that retrains SwinUnet+CLAHE on all 2000 training images (no CV,
k_fold='allin'), using the SAME Image_clahe/Label data that steps 1-2 of the audit
just verified is clean (exact ID match against the official raw ISIC-2017 training
set, no corruption/blanks/duplicates, no cross-contamination from any other
pipeline). If this independent run also lands at ~23 bad images (dice<0.7), that
confirms the original 23-count is a real, reproducible property of the data and
model, not an artifact of a specific checkpoint or a stale/corrupted file.
"""
import json
import os
import subprocess
import sys
import time
import traceback

AVIT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(AVIT_DIR)

STATUS_PATH = '../per_image_analysis_v2/overnight_run/dataintegrity_retrain_status.json'
STEP_LOG_DIR = '../kfold_logs/dataintegrity_retrain_steps'
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
        exp_name = 'isic2017_swinunet_clahe_allin_dataintegrity_verify'
        train_cmd = [
            PY, '-u', 'multi_train_adapt.py',
            '--exp_name', exp_name, '--config_yml', 'Configs/multi_train_local.yml',
            '--model', 'SwinUnet', '--batch_size', '16', '--dataset', 'isic2017', '--k_fold', 'allin',
            '--num_epochs', '30',
            '--meta_csv_name', 'meta_isic2017_train2000.csv',
            '--image_subdir', 'Image_clahe',
        ]
        run_step('train', train_cmd, os.path.join(STEP_LOG_DIR, 'train.log'))

        exp_dir = find_exp_dir('{}_SwinUnet_'.format(exp_name))
        final_ckpt = os.path.join(exp_dir, 'final.pth')
        if not os.path.exists(final_ckpt):
            raise RuntimeError('final.pth not found in {}'.format(exp_dir))
        log('Training complete. exp_dir={} final_ckpt={}'.format(exp_dir, final_ckpt))

        eval_csv = '../per_image_analysis_v2/bad_image_augmentation/dataintegrity_verify_per_image_dice.csv'
        run_step('eval', [
            PY, 'eval_train_full_recovery.py', '--ckpt', final_ckpt, '--model_name', 'SwinUnet',
            '--image_subdir', 'Image_clahe', '--out_csv', eval_csv,
        ], os.path.join(STEP_LOG_DIR, 'eval.log'))

        import pandas as pd
        df = pd.read_csv(eval_csv, dtype={'image_id': str})
        n_bad = int((df['dice'] < 0.7).sum())
        mean_dice = float(df['dice'].mean())

        # cross-check against the ORIGINAL allin run's per-image list: same underlying
        # images, or a genuinely different failure set from a different training run?
        orig = pd.read_csv('../per_image_analysis_v2/bad_image_augmentation/allin_final_per_image_dice.csv',
                            dtype={'image_id': str})
        orig_bad_ids = set(orig[orig['dice'] < 0.7]['image_id'])
        new_bad_ids = set(df[df['dice'] < 0.7]['image_id'])
        overlap = orig_bad_ids & new_bad_ids

        result = {
            'exp_dir': exp_dir, 'final_ckpt': final_ckpt, 'n': len(df),
            'mean_dice': mean_dice, 'n_bad_dice_lt_0.7': n_bad,
            'orig_bad_count': len(orig_bad_ids), 'new_bad_count': len(new_bad_ids),
            'overlap_count': len(overlap),
            'orig_bad_ids': sorted(orig_bad_ids), 'new_bad_ids': sorted(new_bad_ids),
        }
        summary_path = '../per_image_analysis_v2/overnight_run/dataintegrity_retrain_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(result, f, indent=2)

        log('INDEPENDENT RETRAIN RESULT: n_bad={} (orig was {}), mean_dice={:.4f}, overlap_with_orig={}/{}'.format(
            n_bad, len(orig_bad_ids), mean_dice, len(overlap), len(orig_bad_ids)))
        log('Saved: {}'.format(summary_path))
        status['result'] = result
        status['final'] = 'COMPLETE'
        save_status()
        log('DATAINTEGRITY_RETRAIN_COMPLETE')
    except Exception as e:
        status['final'] = 'FAILED'
        status['error'] = str(e)
        status['traceback'] = traceback.format_exc()
        save_status()
        log('DATAINTEGRITY_RETRAIN_FAILED: {}'.format(e))
        log(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
