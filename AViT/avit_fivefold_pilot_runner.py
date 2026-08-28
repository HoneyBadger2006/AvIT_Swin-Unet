"""
Commits AViT's shift-only augmentation pilot (189 hard training images, hflip/
vflip/rot90/rot270/shiftR10blur -- NO isolate/zoom crop variants) to a full
5-fold sweep, mirroring SwinUnet's own fivefold_pilot_runner.py exactly:
  - Fold 0 reused from the already-completed single-fold "allin" pilot
    (avit_shift189_pilot_summary.json), same as SwinUnet's fold 0 reuse.
  - Folds 1-4: true k-fold CV (each fold's own ~1600-image split + the same
    189-image/924-row shift-only manifest added), genuine held-out validation,
    best.pth checkpoint selection (NOT final.pth -- that was only needed for
    fold 0's degenerate allin/no-CV val=train setup). Unseeded, matching
    SwinUnet's fold 1-4 precedent exactly (seed was only used for fold 0 and
    the identification runs, not the true-CV folds).
  - Each fold evaluated against ITS OWN AViT clahe baseline checkpoint's
    per-image test600 dice (network=AViT, stage='clahe', fold=N in
    per_image_final_pipeline.csv) -- hard-subset composition and size varies
    naturally per fold (145/150/141/150/132), same as SwinUnet's sweep.
"""
import json
import os
import subprocess
import sys
import time
import traceback

AVIT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(AVIT_DIR)

STATUS_PATH = '../per_image_analysis_v2/overnight_run/avit_fivefold_pilot_status.json'
STEP_LOG_DIR = '../kfold_logs/avit_fivefold_pilot_steps'
os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
os.makedirs(STEP_LOG_DIR, exist_ok=True)

PY = '../venv/Scripts/python.exe'
MODEL_NAME = 'SwinSeg_CNNprompt_adapt'  # AViT
PILOT0_SUMMARY = '../per_image_analysis_v2/overnight_run/avit_shift189_pilot_summary.json'
MANIFEST_CSV = '../per_image_analysis_v2/bad_image_augmentation/manifest_avit_seed42_dice189_shift5.csv'
META_TAG = 'avit_dice189_shift5_5fold'

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

    exp_name = 'isic2017_avit_clahe_dice189_shift5_5fold_k{}'.format(fold)
    train_cmd = [
        PY, '-u', 'multi_train_adapt.py',
        '--exp_name', exp_name, '--config_yml', 'Configs/multi_train_local.yml',
        '--model', MODEL_NAME, '--batch_size', '16', '--dataset', 'isic2017', '--k_fold', str(fold),
        '--num_epochs', '30',
        '--meta_csv_name', 'meta_isic2017_train2000_{}.csv'.format(META_TAG),
        '--fixed_test_csv_name', 'meta_isic2017_test600.csv',
        '--image_subdir', 'Image_clahe',
    ]
    run_step('k{}_train'.format(fold), train_cmd, os.path.join(STEP_LOG_DIR, 'k{}_train.log'.format(fold)))

    exp_dir = find_exp_dir('{}_{}_'.format(exp_name, MODEL_NAME))
    pilot_ckpt = os.path.join(exp_dir, 'best.pth')
    if not os.path.exists(pilot_ckpt):
        raise RuntimeError('best.pth not found in {}'.format(exp_dir))
    log('Fold {} training complete. exp_dir={} pilot_ckpt={}'.format(fold, exp_dir, pilot_ckpt))

    decision_json = '../per_image_analysis_v2/overnight_run/avit_pilot_5fold_k{}_decision.json'.format(fold)
    per_image_csv = '../per_image_analysis_v2/overnight_run/avit_pilot_5fold_k{}_per_image.csv'.format(fold)
    run_step('k{}_eval'.format(fold), [
        PY, 'eval_pilot_vs_baseline.py',
        '--pilot_ckpt', pilot_ckpt, '--model_name', MODEL_NAME, '--network', 'AViT',
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
        with open(PILOT0_SUMMARY) as f:
            pilot0 = json.load(f)
        fold0_summary = {
            'fold': 0, 'exp_dir': pilot0['exp_dir'], 'pilot_ckpt': pilot0['final_ckpt'],
            'full_test600': pilot0['full_test600'], 'bad_subset': pilot0['bad_subset_paired_test'],
            'decision': pilot0['decision'], 'note': 'no-CV allin run, final.pth (not a true fold split)',
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
        summary_path = '../per_image_analysis_v2/overnight_run/avit_fivefold_pilot_summary.json'
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
        log('AVIT_FIVEFOLD_PILOT_RUNNER_COMPLETE')
    except Exception as e:
        status['final'] = 'FAILED'
        status['error'] = str(e)
        status['traceback'] = traceback.format_exc()
        save_status()
        log('AVIT_FIVEFOLD_PILOT_RUNNER_FAILED: {}'.format(e))
        log(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
