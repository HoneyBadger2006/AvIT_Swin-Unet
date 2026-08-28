"""
Runs the HSV and YCbCr color-space pilots (Prof. Samavi's spec): SwinUnet, RAW
(no CLAHE) baseline architecture/hyperparameters, true fold-0 CV (matching the
existing raw baseline exactly: batch_size=16, num_epochs=30, meta_isic2017_train2000
+ fixed test600), just swapping image_subdir to Image_hsv / Image_ycbcr. Sequential:
HSV first, then YCbCr. Each is evaluated against the existing raw RGB baseline on
the full test600 set and on the existing 73 hard test images (see
eval_colorspace_pilot.py for why that specific fixed 73-image list, not a fresh
hard-subset derived from the raw baseline).
"""
import json
import os
import subprocess
import sys
import time
import traceback

AVIT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(AVIT_DIR)

STATUS_PATH = '../per_image_analysis_v2/overnight_run/colorspace_pilot_status.json'
STEP_LOG_DIR = '../kfold_logs/colorspace_pilot_steps'
os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
os.makedirs(STEP_LOG_DIR, exist_ok=True)

PY = '../venv/Scripts/python.exe'

COLORSPACES = [
    {'name': 'HSV', 'image_subdir': 'Image_hsv', 'exp_name': 'isic2017_swinunet_hsv_k0'},
    {'name': 'YCbCr', 'image_subdir': 'Image_ycbcr', 'exp_name': 'isic2017_swinunet_ycbcr_k0'},
]

status = {'colorspaces': {}, 'current': None, 'started_at': time.strftime('%Y-%m-%d %H:%M:%S')}


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


def run_colorspace(spec):
    name = spec['name']
    log('=== {} PILOT ==='.format(name))
    train_cmd = [
        PY, '-u', 'multi_train_adapt.py',
        '--exp_name', spec['exp_name'], '--config_yml', 'Configs/multi_train_local.yml',
        '--model', 'SwinUnet', '--batch_size', '16', '--dataset', 'isic2017', '--k_fold', '0',
        '--num_epochs', '30',
        '--meta_csv_name', 'meta_isic2017_train2000.csv',
        '--fixed_test_csv_name', 'meta_isic2017_test600.csv',
        '--image_subdir', spec['image_subdir'],
    ]
    run_step('{}_train'.format(name), train_cmd, os.path.join(STEP_LOG_DIR, '{}_train.log'.format(name)))

    exp_dir = find_exp_dir('{}_SwinUnet_'.format(spec['exp_name']))
    pilot_ckpt = os.path.join(exp_dir, 'best.pth')
    if not os.path.exists(pilot_ckpt):
        raise RuntimeError('best.pth not found in {}'.format(exp_dir))
    log('{} training complete. exp_dir={} pilot_ckpt={}'.format(name, exp_dir, pilot_ckpt))

    eval_json = '../per_image_analysis_v2/overnight_run/colorspace_{}_decision.json'.format(name.lower())
    eval_csv = '../per_image_analysis_v2/overnight_run/colorspace_{}_per_image.csv'.format(name.lower())
    run_step('{}_eval'.format(name), [
        PY, 'eval_colorspace_pilot.py',
        '--ckpt', pilot_ckpt, '--model_name', 'SwinUnet', '--image_subdir', spec['image_subdir'],
        '--colorspace_name', name, '--out_json', eval_json, '--out_csv', eval_csv,
    ], os.path.join(STEP_LOG_DIR, '{}_eval.log'.format(name)))

    with open(eval_json) as f:
        result = json.load(f)
    result['exp_dir'] = exp_dir
    result['pilot_ckpt'] = pilot_ckpt
    log('{} RESULT: full_test600 diff={:+.4f} p={:.4g}  |  existing_hard73 diff={:+.4f} p={:.4g}'.format(
        name, result['full_test600']['mean_diff'], result['full_test600']['p_value_one_sided'],
        result['existing_hard73_subset']['mean_diff'], result['existing_hard73_subset']['p_value_one_sided']))
    status['colorspaces'][name] = result
    save_status()
    return result


def main():
    try:
        for spec in COLORSPACES:
            run_colorspace(spec)

        status['final'] = 'COMPLETE'
        save_status()
        log('COLORSPACE_PILOT_RUNNER_COMPLETE')
    except Exception as e:
        status['final'] = 'FAILED'
        status['error'] = str(e)
        status['traceback'] = traceback.format_exc()
        save_status()
        log('COLORSPACE_PILOT_RUNNER_FAILED: {}'.format(e))
        log(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
