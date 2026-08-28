"""
Commits the validated A+B-only corrected pilot (23 bad training images, tight GT
bboxes, 6 augmentation variants: hflip/vflip/rot90/rot270/cropA/cropB) to a full
5-fold sweep. Fold 0 is already done (the successful corrected pilot from
priority1_summary.json). This script retrains folds 1-4 with the SAME fixed
23-image/6-variant augmentation set added to each fold's own ~1600-image training
split (true k-fold CV, genuine ~400-image held-out val per fold -- unlike fold 0's
no-CV 'allin' run), then aggregates the full 5-fold mean+/-std.

Checkpoint choice note: fold 0 used final.pth (literal last epoch) because its
'allin' run has no genuine held-out validation (val=train), so best.pth there would
really mean "best-on-train", not legitimate model selection. Folds 1-4 DO have a
genuine held-out split, so best.pth (val-IOU-selected checkpoint) is used instead --
methodologically consistent with how the baseline checkpoints themselves were
selected. Everything else (data recipe, hyperparameters) is identical to fold 0.
"""
import json
import os
import subprocess
import sys
import time
import traceback

AVIT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(AVIT_DIR)

STATUS_PATH = '../per_image_analysis_v2/overnight_run/fivefold_pilot_status.json'
STEP_LOG_DIR = '../kfold_logs/fivefold_pilot_steps'
os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
os.makedirs(STEP_LOG_DIR, exist_ok=True)

PY = '../venv/Scripts/python.exe'
PRIORITY1_SUMMARY = '../per_image_analysis_v2/overnight_run/priority1_summary.json'
MANIFEST_CSV = '../per_image_analysis_v2/bad_image_augmentation/manifest_pilot_badaug23.csv'
META_TAG = 'pilot_badaug23_5fold'

status = {'folds': {}, 'current': None, 'started_at': time.strftime('%Y-%m-%d %H:%M:%S')}


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


def run_fold(fold):
    log('=== FOLD {} ==='.format(fold))
    run_step('k{}_build_csv'.format(fold), [
        PY, 'build_fold_csv_generic.py', '--fold', str(fold),
        '--manifest_csv', MANIFEST_CSV, '--meta_tag', META_TAG,
    ], os.path.join(STEP_LOG_DIR, 'k{}_build_csv.log'.format(fold)))

    exp_name = 'isic2017_swinunet_clahe_pilot_badaug23_5fold_k{}'.format(fold)
    train_cmd = [
        PY, '-u', 'multi_train_adapt.py',
        '--exp_name', exp_name, '--config_yml', 'Configs/multi_train_local.yml',
        '--model', 'SwinUnet', '--batch_size', '16', '--dataset', 'isic2017', '--k_fold', str(fold),
        '--num_epochs', '30',
        '--meta_csv_name', 'meta_isic2017_train2000_{}.csv'.format(META_TAG),
        '--fixed_test_csv_name', 'meta_isic2017_test600.csv',
        '--image_subdir', 'Image_clahe',
    ]
    run_step('k{}_train'.format(fold), train_cmd, os.path.join(STEP_LOG_DIR, 'k{}_train.log'.format(fold)))

    exp_dir = find_exp_dir('{}_SwinUnet_'.format(exp_name))
    pilot_ckpt = os.path.join(exp_dir, 'best.pth')
    if not os.path.exists(pilot_ckpt):
        raise RuntimeError('best.pth not found in {}'.format(exp_dir))
    log('Fold {} training complete. exp_dir={} pilot_ckpt={}'.format(fold, exp_dir, pilot_ckpt))

    decision_json = '../per_image_analysis_v2/overnight_run/pilot_5fold_k{}_decision.json'.format(fold)
    per_image_csv = '../per_image_analysis_v2/overnight_run/pilot_5fold_k{}_per_image.csv'.format(fold)
    run_step('k{}_eval'.format(fold), [
        PY, 'eval_pilot_vs_baseline.py',
        '--pilot_ckpt', pilot_ckpt, '--model_name', 'SwinUnet', '--network', 'SwinUnet',
        '--baseline_stage', 'clahe', '--fold', str(fold), '--image_subdir', 'Image_clahe',
        '--out_json', decision_json, '--out_csv', per_image_csv,
    ], os.path.join(STEP_LOG_DIR, 'k{}_eval.log'.format(fold)))

    with open(decision_json) as f:
        result = json.load(f)

    fold_summary = {
        'fold': fold, 'exp_dir': exp_dir, 'pilot_ckpt': pilot_ckpt,
        'full_test600': result['full_test600'], 'bad_subset': result['bad_subset'],
        'decision': result['decision'],
    }
    log('FOLD {} RESULT: full_test600 pilot={:.4f} baseline={:.4f} diff={:+.4f} p={:.4g}  |  hard_subset(n={}) pilot={:.4f} baseline={:.4f} diff={:+.4f} p={:.4g}'.format(
        fold, result['full_test600']['pilot_mean'], result['full_test600']['baseline_mean'],
        result['full_test600']['mean_diff'], result['full_test600']['p_value_one_sided'],
        result['bad_subset']['n'], result['bad_subset']['pilot_mean'], result['bad_subset']['baseline_mean'],
        result['bad_subset']['mean_diff'], result['bad_subset']['p_value_one_sided']))
    status['folds'][str(fold)] = fold_summary
    save_status()
    return fold_summary


def main():
    try:
        with open(PRIORITY1_SUMMARY) as f:
            fold0 = json.load(f)
        fold0_summary = {
            'fold': 0, 'exp_dir': fold0['exp_dir'], 'pilot_ckpt': fold0['final_ckpt'],
            'full_test600': fold0['full_test600'], 'bad_subset': fold0['bad_subset_paired_test'],
            'decision': fold0['decision'], 'note': 'no-CV allin run, final.pth (not a true fold split)',
        }
        status['folds']['0'] = fold0_summary
        save_status()
        log('Fold 0 (already done): full_test600 diff={:+.4f}, hard_subset diff={:+.4f}'.format(
            fold0_summary['full_test600']['mean_diff'], fold0_summary['bad_subset']['mean_diff']))

        for fold in [1, 2, 3, 4]:
            run_fold(fold)

        import statistics
        all_folds = [status['folds'][str(f)] for f in range(5)]
        full_pilot_means = [f['full_test600']['pilot_mean'] for f in all_folds]
        full_baseline_means = [f['full_test600']['baseline_mean'] for f in all_folds]
        hard_pilot_means = [f['bad_subset']['pilot_mean'] for f in all_folds]
        hard_baseline_means = [f['bad_subset']['baseline_mean'] for f in all_folds]

        summary = {
            'full_test600_pilot_5fold_mean': statistics.mean(full_pilot_means),
            'full_test600_pilot_5fold_std': statistics.stdev(full_pilot_means),
            'full_test600_baseline_5fold_mean': statistics.mean(full_baseline_means),
            'full_test600_baseline_5fold_std': statistics.stdev(full_baseline_means),
            'hard_subset_pilot_5fold_mean': statistics.mean(hard_pilot_means),
            'hard_subset_pilot_5fold_std': statistics.stdev(hard_pilot_means),
            'hard_subset_baseline_5fold_mean': statistics.mean(hard_baseline_means),
            'hard_subset_baseline_5fold_std': statistics.stdev(hard_baseline_means),
            'per_fold': {str(f['fold']): f for f in all_folds},
        }
        summary_path = '../per_image_analysis_v2/overnight_run/fivefold_pilot_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        log('5-FOLD SUMMARY: full_test600 pilot={:.4f}+/-{:.4f} baseline={:.4f}+/-{:.4f}  |  hard_subset pilot={:.4f}+/-{:.4f} baseline={:.4f}+/-{:.4f}'.format(
            summary['full_test600_pilot_5fold_mean'], summary['full_test600_pilot_5fold_std'],
            summary['full_test600_baseline_5fold_mean'], summary['full_test600_baseline_5fold_std'],
            summary['hard_subset_pilot_5fold_mean'], summary['hard_subset_pilot_5fold_std'],
            summary['hard_subset_baseline_5fold_mean'], summary['hard_subset_baseline_5fold_std']))
        log('Saved: {}'.format(summary_path))

        status['final'] = 'COMPLETE'
        save_status()
        log('FIVEFOLD_PILOT_RUNNER_COMPLETE')
    except Exception as e:
        status['final'] = 'FAILED'
        status['error'] = str(e)
        status['traceback'] = traceback.format_exc()
        save_status()
        log('FIVEFOLD_PILOT_RUNNER_FAILED: {}'.format(e))
        log(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
