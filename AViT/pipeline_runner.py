"""
Resilient, resumable multi-run driver for multi_train_adapt.py.

Runs a list of experiment specs sequentially via subprocess, one python process
per run. Each run is wrapped in try/except so a crash in one run is logged and
does not kill the whole pipeline. Already-completed runs (a results dir matching
the exp_name/model prefix that contains test_results.csv) are skipped, so this
script can be re-launched after an interruption without redoing finished folds.

Usage: python pipeline_runner.py --phase {baseline,swinunet_focal,avit_focal}
"""
import argparse
import glob
import os
import subprocess
import sys
import traceback
from datetime import datetime

ROOT = 'C:/Users/quanp/Downloads/ISIC 2017'
AVIT_DIR = os.path.join(ROOT, 'AViT')
PY = os.path.join(ROOT, 'venv', 'Scripts', 'python.exe')
RESULTS_DIR = os.path.join(ROOT, 'results')
LOG_DIR = os.path.join(ROOT, 'kfold_logs')
CONFIG_YML = 'Configs/multi_train_local.yml'

os.makedirs(LOG_DIR, exist_ok=True)


def log(master_log_path, msg):
    line = '[{}] {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg)
    print(line, flush=True)
    with open(master_log_path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
        f.flush()


def already_done(exp_name, model):
    pattern = os.path.join(RESULTS_DIR, '{}_{}_*'.format(exp_name, model))
    for d in glob.glob(pattern):
        if os.path.exists(os.path.join(d, 'test_results.csv')):
            return d
    return None


def run_one(spec, master_log_path):
    exp_name = spec['exp_name']
    model = spec['model']

    existing = already_done(exp_name, model)
    if existing:
        log(master_log_path, 'SKIP {} (already complete: {})'.format(exp_name, existing))
        return True

    log(master_log_path, 'START {} (model={} k_fold={} focal={})'.format(
        exp_name, model, spec.get('k_fold'), spec.get('use_focal_loss', False)))

    cmd = [
        PY, '-u', 'multi_train_adapt.py',
        '--exp_name', exp_name,
        '--config_yml', CONFIG_YML,
        '--model', model,
        '--batch_size', str(spec.get('batch_size', 16)),
        '--dataset', 'isic2017',
        '--k_fold', str(spec['k_fold']),
    ]
    if spec.get('num_epochs') is not None:
        cmd += ['--num_epochs', str(spec['num_epochs'])]
    if spec.get('use_focal_loss'):
        cmd += ['--use_focal_loss', '--focal_lambda', str(spec.get('focal_lambda', 1.0))]
    if spec.get('only_test'):
        cmd += ['--only_test', '--test_model_dir', spec['test_model_dir']]
    if spec.get('meta_csv_name'):
        cmd += ['--meta_csv_name', spec['meta_csv_name']]
    if spec.get('fixed_test_csv_name'):
        cmd += ['--fixed_test_csv_name', spec['fixed_test_csv_name']]

    out_path = os.path.join(LOG_DIR, '{}.log'.format(exp_name))
    err_path = os.path.join(LOG_DIR, '{}_err.log'.format(exp_name))

    try:
        with open(out_path, 'w', encoding='utf-8') as out_f, \
             open(err_path, 'w', encoding='utf-8') as err_f:
            result = subprocess.run(cmd, cwd=AVIT_DIR, stdout=out_f, stderr=err_f)
        if result.returncode == 0:
            log(master_log_path, 'FINISH {} exit_code=0'.format(exp_name))
            return True
        else:
            log(master_log_path, 'FAIL {} exit_code={} (see {})'.format(
                exp_name, result.returncode, err_path))
            return False
    except Exception:
        log(master_log_path, 'EXCEPTION while running {}:\n{}'.format(
            exp_name, traceback.format_exc()))
        return False


def build_specs(phase):
    if phase == 'baseline':
        k1_ckpt = os.path.join(
            RESULTS_DIR, 'isic2017_swinunet_bceLogits_dice_k1_SwinUnet_20260718_0414', 'best.pth')
        return [
            {'exp_name': 'isic2017_swinunet_bceLogits_dice_k1_testresume', 'model': 'SwinUnet',
             'k_fold': 1, 'only_test': True, 'test_model_dir': k1_ckpt},
            {'exp_name': 'isic2017_swinunet_bceLogits_dice_k2', 'model': 'SwinUnet',
             'k_fold': 2, 'batch_size': 16, 'num_epochs': 30},
            {'exp_name': 'isic2017_swinunet_bceLogits_dice_k3', 'model': 'SwinUnet',
             'k_fold': 3, 'batch_size': 16, 'num_epochs': 30},
            {'exp_name': 'isic2017_swinunet_bceLogits_dice_k4', 'model': 'SwinUnet',
             'k_fold': 4, 'batch_size': 16, 'num_epochs': 30},
        ]
    elif phase == 'swinunet_focal':
        specs = []
        for k in ['No', 1, 2, 3, 4]:
            tag = 'k0' if k == 'No' else 'k{}'.format(k)
            specs.append({
                'exp_name': 'isic2017_swinunet_focal_{}'.format(tag), 'model': 'SwinUnet',
                'k_fold': k, 'batch_size': 16, 'num_epochs': 30,
                'use_focal_loss': True, 'focal_lambda': 1.0,
            })
        return specs
    elif phase == 'avit_focal':
        specs = []
        for k in ['No', 1, 2, 3, 4]:
            tag = 'k0' if k == 'No' else 'k{}'.format(k)
            specs.append({
                'exp_name': 'isic2017_avit_focal_{}'.format(tag), 'model': 'SwinSeg_CNNprompt_adapt',
                'k_fold': k, 'batch_size': 16, 'num_epochs': 30,
                'use_focal_loss': True, 'focal_lambda': 1.0,
            })
        return specs
    elif phase in ('official_swinunet_baseline', 'official_swinunet_focal',
                   'official_avit_baseline', 'official_avit_focal'):
        model = 'SwinUnet' if 'swinunet' in phase else 'SwinSeg_CNNprompt_adapt'
        use_focal = phase.endswith('focal')
        exp_prefix = 'isic2017_{}_official_{}'.format(
            'swinunet' if model == 'SwinUnet' else 'avit', 'focal' if use_focal else 'baseline')
        specs = []
        # Genuine 5-way partition within the 2000 official training images (k=0..4),
        # NOT the 'No' pseudo-fold used in the old pooled-split pipeline.
        for k in [0, 1, 2, 3, 4]:
            spec = {
                'exp_name': '{}_k{}'.format(exp_prefix, k), 'model': model,
                'k_fold': k, 'batch_size': 16, 'num_epochs': 30,
                'meta_csv_name': 'meta_isic2017_train2000.csv',
                'fixed_test_csv_name': 'meta_isic2017_test600.csv',
            }
            if use_focal:
                spec['use_focal_loss'] = True
                spec['focal_lambda'] = 1.0
            specs.append(spec)
        return specs
    else:
        raise ValueError('Unknown phase: {}'.format(phase))


def run_phase(phase):
    master_log_path = os.path.join(LOG_DIR, 'pipeline_{}.log'.format(phase))
    log(master_log_path, '=== PHASE {} STARTED ==='.format(phase))

    specs = build_specs(phase)
    n_ok, n_fail = 0, 0
    for spec in specs:
        ok = run_one(spec, master_log_path)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    log(master_log_path, '=== PHASE {} COMPLETE: {} ok, {} failed (of {}) ==='.format(
        phase, n_ok, n_fail, len(specs)))


COMPOSITE_PHASES = {
    'all': ['baseline', 'swinunet_focal', 'avit_focal'],
    'official_all': ['official_swinunet_baseline', 'official_swinunet_focal',
                      'official_avit_baseline', 'official_avit_focal'],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True,
                        choices=['baseline', 'swinunet_focal', 'avit_focal',
                                 'official_swinunet_baseline', 'official_swinunet_focal',
                                 'official_avit_baseline', 'official_avit_focal',
                                 'all', 'official_all'])
    args = parser.parse_args()

    is_composite = args.phase in COMPOSITE_PHASES
    all_log_path = os.path.join(LOG_DIR, 'pipeline_{}.log'.format(args.phase))
    phases = COMPOSITE_PHASES[args.phase] if is_composite else [args.phase]

    if is_composite:
        log(all_log_path, '=== {} PIPELINE STARTED (pid={}) ==='.format(args.phase, os.getpid()))

    for phase in phases:
        try:
            run_phase(phase)
        except Exception:
            # A whole-phase crash (e.g. bad spec) still shouldn't take down later phases.
            if is_composite:
                log(all_log_path, 'PHASE {} CRASHED:\n{}'.format(phase, traceback.format_exc()))

    if is_composite:
        log(all_log_path, '=== {} PIPELINE COMPLETE ==='.format(args.phase))


if __name__ == '__main__':
    main()
