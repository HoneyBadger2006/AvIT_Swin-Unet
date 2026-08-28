"""
Re-runs the fold-0 HSV pilot (SwinUnet, raw baseline setup, no CLAHE, true fold-0
CV) with HSV-specific normalization stats computed directly from the HSV-converted
training set (see compute_colorspace_norm_stats.py), instead of the RGB ImageNet
constants used in the original HSV pilot. Same everything else, so the only
variable that changes vs. the original HSV pilot is normalization.
"""
import json
import os
import subprocess
import sys
import time
import traceback

AVIT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(AVIT_DIR)

STATUS_PATH = '../per_image_analysis_v2/overnight_run/hsv_corrected_status.json'
STEP_LOG_DIR = '../kfold_logs/hsv_corrected_steps'
os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
os.makedirs(STEP_LOG_DIR, exist_ok=True)

PY = '../venv/Scripts/python.exe'
NORM_MEAN = ['0.1372', '0.2540', '0.7138']
NORM_STD = ['0.2187', '0.1727', '0.1507']

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
        exp_name = 'isic2017_swinunet_hsv_normfix_k0'
        train_cmd = [
            PY, '-u', 'multi_train_adapt.py',
            '--exp_name', exp_name, '--config_yml', 'Configs/multi_train_local.yml',
            '--model', 'SwinUnet', '--batch_size', '16', '--dataset', 'isic2017', '--k_fold', '0',
            '--num_epochs', '30',
            '--meta_csv_name', 'meta_isic2017_train2000.csv',
            '--fixed_test_csv_name', 'meta_isic2017_test600.csv',
            '--image_subdir', 'Image_hsv',
            '--norm_mean'] + NORM_MEAN + ['--norm_std'] + NORM_STD
        run_step('hsv_normfix_train', train_cmd, os.path.join(STEP_LOG_DIR, 'train.log'))

        exp_dir = find_exp_dir('{}_SwinUnet_'.format(exp_name))
        pilot_ckpt = os.path.join(exp_dir, 'best.pth')
        if not os.path.exists(pilot_ckpt):
            raise RuntimeError('best.pth not found in {}'.format(exp_dir))
        log('Training complete. exp_dir={} pilot_ckpt={}'.format(exp_dir, pilot_ckpt))

        eval_json = '../per_image_analysis_v2/overnight_run/colorspace_hsv_normfix_decision.json'
        eval_csv = '../per_image_analysis_v2/overnight_run/colorspace_hsv_normfix_per_image.csv'
        run_step('hsv_normfix_eval', [
            PY, 'eval_colorspace_pilot.py',
            '--ckpt', pilot_ckpt, '--model_name', 'SwinUnet', '--image_subdir', 'Image_hsv',
            '--colorspace_name', 'HSV_normfix', '--out_json', eval_json, '--out_csv', eval_csv,
            '--norm_mean'] + NORM_MEAN + ['--norm_std'] + NORM_STD,
            os.path.join(STEP_LOG_DIR, 'eval.log'))

        with open(eval_json) as f:
            result = json.load(f)
        result['exp_dir'] = exp_dir
        result['pilot_ckpt'] = pilot_ckpt
        log('HSV_NORMFIX RESULT: full_test600 diff={:+.4f} p={:.4g}  |  existing_hard73 diff={:+.4f} p={:.4g}'.format(
            result['full_test600']['mean_diff'], result['full_test600']['p_value_one_sided'],
            result['existing_hard73_subset']['mean_diff'], result['existing_hard73_subset']['p_value_one_sided']))
        status['result'] = result
        status['final'] = 'COMPLETE'
        save_status()
        log('HSV_CORRECTED_RUNNER_COMPLETE')
    except Exception as e:
        status['final'] = 'FAILED'
        status['error'] = str(e)
        status['traceback'] = traceback.format_exc()
        save_status()
        log('HSV_CORRECTED_RUNNER_FAILED: {}'.format(e))
        log(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
