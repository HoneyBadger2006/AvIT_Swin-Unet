"""
Replicates the exact "allin" bad-image identification process already run for
SwinUnet, but for AViT (model_name='SwinSeg_CNNprompt_adapt'): all 2000 official
training images, no cross-validation, CLAHE, no FTL (matching SwinUnet's own
identification run exactly, so architecture is the only variable that changes),
fixed seed=42 (the same primary seed used for SwinUnet's seed42 run), same
hyperparameters (batch_size=16, num_epochs=30). Evaluates the resulting
final.pth checkpoint against the same 2000 training images and reports the
Dice<0.7 bad-image count plus mean training Dice, for direct comparison against
SwinUnet's 37 (mean ~0.92) and FAT-Net's reported 222 (mean ~0.86).
"""
import json
import os
import subprocess
import sys
import time
import traceback

AVIT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(AVIT_DIR)

STATUS_PATH = '../per_image_analysis_v2/overnight_run/avit_allin_seed42_status.json'
STEP_LOG_DIR = '../kfold_logs/avit_allin_seed42_steps'
os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
os.makedirs(STEP_LOG_DIR, exist_ok=True)

PY = '../venv/Scripts/python.exe'
MODEL_NAME = 'SwinSeg_CNNprompt_adapt'  # AViT

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
        exp_name = 'isic2017_avit_clahe_allin_seed42'
        train_cmd = [
            PY, '-u', 'multi_train_adapt.py',
            '--exp_name', exp_name, '--config_yml', 'Configs/multi_train_local.yml',
            '--model', MODEL_NAME, '--batch_size', '16', '--dataset', 'isic2017', '--k_fold', 'allin',
            '--num_epochs', '30',
            '--meta_csv_name', 'meta_isic2017_train2000.csv',
            '--image_subdir', 'Image_clahe',
            '--seed', '42',
        ]
        run_step('train', train_cmd, os.path.join(STEP_LOG_DIR, 'train.log'))

        exp_dir = find_exp_dir('{}_{}_'.format(exp_name, MODEL_NAME))
        final_ckpt = os.path.join(exp_dir, 'final.pth')
        if not os.path.exists(final_ckpt):
            raise RuntimeError('final.pth not found in {}'.format(exp_dir))
        log('Training complete. exp_dir={} final_ckpt={}'.format(exp_dir, final_ckpt))

        eval_csv = '../per_image_analysis_v2/bad_image_augmentation/avit_allin_seed42_per_image_dice.csv'
        run_step('eval', [
            PY, 'eval_train_full_recovery.py', '--ckpt', final_ckpt, '--model_name', MODEL_NAME,
            '--image_subdir', 'Image_clahe', '--out_csv', eval_csv,
        ], os.path.join(STEP_LOG_DIR, 'eval.log'))

        import pandas as pd
        df = pd.read_csv(eval_csv, dtype={'image_id': str})
        n_bad = int((df['dice'] < 0.7).sum())
        mean_dice = float(df['dice'].mean())

        result = {
            'model': 'AViT', 'model_name': MODEL_NAME, 'seed': 42,
            'exp_dir': exp_dir, 'final_ckpt': final_ckpt, 'n': len(df),
            'mean_dice': mean_dice, 'n_bad_dice_lt_0.7': n_bad,
            'comparison': {
                'AViT': {'n_bad': n_bad, 'mean_dice': mean_dice},
                'SwinUnet_seed42': {'n_bad': 37, 'mean_dice': 0.92},
                'FAT-Net': {'n_bad': 222, 'mean_dice': 0.86},
            },
        }
        summary_path = '../per_image_analysis_v2/overnight_run/avit_allin_seed42_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(result, f, indent=2)

        log('AVIT ALLIN RESULT: n_bad(dice<0.7)={} / {}  mean_dice={:.4f}'.format(n_bad, len(df), mean_dice))
        log('Saved: {}'.format(summary_path))
        status['result'] = result
        status['final'] = 'COMPLETE'
        save_status()
        log('AVIT_ALLIN_SEED42_RUNNER_COMPLETE')
    except Exception as e:
        status['final'] = 'FAILED'
        status['error'] = str(e)
        status['traceback'] = traceback.format_exc()
        save_status()
        log('AVIT_ALLIN_SEED42_RUNNER_FAILED: {}'.format(e))
        log(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
