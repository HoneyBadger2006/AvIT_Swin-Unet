"""
Fixes the reproducibility issue found in the data-integrity audit: multi_train_adapt.py
now supports an explicit --seed (model init, data shuffling, augmentation, cuDNN
determinism -- see set_seed()/seed_worker() there). This script re-runs the "train
on all 2000, no CV" SwinUnet+CLAHE setup 3 times with 3 different FIXED seeds, and
reports the Dice<0.7 and IoU<0.7 bad-image counts for each, plus the mean and range
across the 3 runs -- a genuine, honest uncertainty estimate instead of one run's
number presented as exact.
"""
import json
import os
import subprocess
import sys
import time
import traceback

AVIT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(AVIT_DIR)

STATUS_PATH = '../per_image_analysis_v2/overnight_run/seeded_reproducibility_status.json'
STEP_LOG_DIR = '../kfold_logs/seeded_reproducibility_steps'
os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
os.makedirs(STEP_LOG_DIR, exist_ok=True)

PY = '../venv/Scripts/python.exe'
SEEDS = [42, 123, 2024]

status = {'runs': {}, 'current': None, 'started_at': time.strftime('%Y-%m-%d %H:%M:%S')}


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


def run_seed(seed):
    label = 'seed{}'.format(seed)
    log('=== {} ==='.format(label))
    exp_name = 'isic2017_swinunet_clahe_allin_{}'.format(label)
    train_cmd = [
        PY, '-u', 'multi_train_adapt.py',
        '--exp_name', exp_name, '--config_yml', 'Configs/multi_train_local.yml',
        '--model', 'SwinUnet', '--batch_size', '16', '--dataset', 'isic2017', '--k_fold', 'allin',
        '--num_epochs', '30',
        '--meta_csv_name', 'meta_isic2017_train2000.csv',
        '--image_subdir', 'Image_clahe',
        '--seed', str(seed),
    ]
    run_step('{}_train'.format(label), train_cmd, os.path.join(STEP_LOG_DIR, '{}_train.log'.format(label)))

    exp_dir = find_exp_dir('{}_SwinUnet_'.format(exp_name))
    final_ckpt = os.path.join(exp_dir, 'final.pth')
    if not os.path.exists(final_ckpt):
        raise RuntimeError('final.pth not found in {}'.format(exp_dir))
    log('{} training complete. exp_dir={}'.format(label, exp_dir))

    eval_csv = '../per_image_analysis_v2/bad_image_augmentation/{}_per_image_dice_iou.csv'.format(label)
    run_step('{}_eval'.format(label), [
        PY, 'eval_seeded_run.py', '--ckpt', final_ckpt, '--seed_label', label, '--out_csv', eval_csv,
    ], os.path.join(STEP_LOG_DIR, '{}_eval.log'.format(label)))

    import pandas as pd
    df = pd.read_csv(eval_csv, dtype={'image_id': str})
    n_dice_bad = int((df['dice'] < 0.7).sum())
    n_iou_bad = int((df['iou'] < 0.7).sum())
    result = {
        'seed': seed, 'exp_dir': exp_dir, 'ckpt': final_ckpt,
        'mean_dice': float(df['dice'].mean()), 'mean_iou': float(df['iou'].mean()),
        'dice_lt_0.7_count': n_dice_bad, 'iou_lt_0.7_count': n_iou_bad,
    }
    log('{} RESULT: Dice<0.7={}  IoU<0.7={}  mean_dice={:.4f}  mean_iou={:.4f}'.format(
        label, n_dice_bad, n_iou_bad, result['mean_dice'], result['mean_iou']))
    status['runs'][label] = result
    save_status()
    return result


def main():
    try:
        results = [run_seed(s) for s in SEEDS]

        dice_counts = [r['dice_lt_0.7_count'] for r in results]
        iou_counts = [r['iou_lt_0.7_count'] for r in results]
        summary = {
            'seeds': SEEDS,
            'dice_lt_0.7_counts': dice_counts,
            'dice_mean': sum(dice_counts) / len(dice_counts),
            'dice_range': [min(dice_counts), max(dice_counts)],
            'iou_lt_0.7_counts': iou_counts,
            'iou_mean': sum(iou_counts) / len(iou_counts),
            'iou_range': [min(iou_counts), max(iou_counts)],
            'per_run': results,
        }
        summary_path = '../per_image_analysis_v2/overnight_run/seeded_reproducibility_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        log('FINAL SUMMARY: Dice<0.7 counts={} mean={:.1f} range={}  |  IoU<0.7 counts={} mean={:.1f} range={}'.format(
            dice_counts, summary['dice_mean'], summary['dice_range'],
            iou_counts, summary['iou_mean'], summary['iou_range']))
        log('Saved: {}'.format(summary_path))

        status['final'] = 'COMPLETE'
        save_status()
        log('SEEDED_REPRODUCIBILITY_RUNNER_COMPLETE')
    except Exception as e:
        status['final'] = 'FAILED'
        status['error'] = str(e)
        status['traceback'] = traceback.format_exc()
        save_status()
        log('SEEDED_REPRODUCIBILITY_RUNNER_FAILED: {}'.format(e))
        log(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
