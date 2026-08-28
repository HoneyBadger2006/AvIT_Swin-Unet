"""
Overnight autonomous orchestrator for the hard-example augmentation study.

Sequence:
  1. Waits for the already-launched SwinUnet fold-0 pilot (CLAHE, no FTL) to finish.
  2. Evaluates it vs the existing SwinUnet+CLAHE fold-0 baseline (full test600 +
     bad-image subset, paired one-sided t-test). Decision: continue or stop.
  3. If continue: runs the same bad-image-ID -> augment -> build-CSV -> train -> eval
     pipeline for SwinUnet folds 1-4, then aggregates a 5-fold summary.
  4. If the SwinUnet sweep completed, repeats the whole pilot-then-sweep process for
     AViT (CLAHE + compound FTL, its own bad-image list from its own checkpoints).

Every step's outcome is written to a running JSON status file
(../per_image_analysis_v2/overnight_run/status.json) and a human-readable log
(../kfold_logs/overnight_orchestrator.log), so progress survives independent of
any particular conversation/process watching it. A failure at any step marks the
whole run FAILED with full detail rather than crashing silently.
"""
import json
import os
import subprocess
import sys
import time
import traceback

AVIT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(AVIT_DIR)

STATUS_PATH = '../per_image_analysis_v2/overnight_run/status.json'
STEP_LOG_DIR = '../kfold_logs/overnight_steps'
os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
os.makedirs(STEP_LOG_DIR, exist_ok=True)

PY = '../venv/Scripts/python.exe'

MODEL_NAME = {'SwinUnet': 'SwinUnet', 'AViT': 'SwinSeg_CNNprompt_adapt'}
BASELINE_STAGE = {'SwinUnet': 'clahe', 'AViT': 'clahe_ftl'}
USE_FTL = {'SwinUnet': False, 'AViT': True}
META_TAG = {'SwinUnet': 'swinunet_badaug', 'AViT': 'avit_badaug'}

CLAHE_CKPTS = {
    'SwinUnet': {
        0: '../results/isic2017_swinunet_clahe_k0_SwinUnet_20260728_0301/best.pth',
        1: '../results/isic2017_swinunet_clahe_k1_SwinUnet_20260728_1556/best.pth',
        2: '../results/isic2017_swinunet_clahe_k2_SwinUnet_20260728_1645/best.pth',
        3: '../results/isic2017_swinunet_clahe_k3_SwinUnet_20260728_1733/best.pth',
        4: '../results/isic2017_swinunet_clahe_k4_SwinUnet_20260728_1823/best.pth',
    },
}
CLAHE_FTL_CKPTS = {
    'AViT': {
        0: '../results/isic2017_avit_clahe_ftl_k0_SwinSeg_CNNprompt_adapt_20260730_1139/best.pth',
        1: '../results/isic2017_avit_clahe_ftl_k1_SwinSeg_CNNprompt_adapt_20260730_1656/best.pth',
        2: '../results/isic2017_avit_clahe_ftl_k2_SwinSeg_CNNprompt_adapt_20260730_1904/best.pth',
        3: '../results/isic2017_avit_clahe_ftl_k3_SwinSeg_CNNprompt_adapt_20260730_2012/best.pth',
        4: '../results/isic2017_avit_clahe_ftl_k4_SwinSeg_CNNprompt_adapt_20260730_2118/best.pth',
    },
}
BASELINE_CKPTS = {'SwinUnet': CLAHE_CKPTS['SwinUnet'], 'AViT': CLAHE_FTL_CKPTS['AViT']}

# fold-0 pilot for SwinUnet was already launched by hand before this orchestrator
# started; this points at that already-running exp dir instead of relaunching it.
PRELAUNCHED_FOLD0 = {
    'SwinUnet': '../results/isic2017_swinunet_clahe_pilot_badaug_k0_SwinUnet_20260811_2220',
}

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


def wait_for_prelaunched_training(exp_dir_glob_prefix, network):
    """Poll for the already-running fold-0 training process to finish."""
    log('Waiting for pre-launched {} fold-0 pilot training to finish...'.format(network))
    log_path = '../kfold_logs/pilot_badaug_swinunet_clahe_k0.log'
    while True:
        if os.path.exists(log_path):
            with open(log_path, 'r', errors='ignore') as f:
                content = f.read()
            if 'Complete training' in content and 'Test ||' in content:
                log('Pre-launched training finished.')
                return
            if 'Traceback' in content:
                raise RuntimeError('Pre-launched fold-0 training crashed. See {}'.format(log_path))
        time.sleep(30)


def find_exp_dir(prefix):
    candidates = [d for d in os.listdir('../results') if d.startswith(prefix)]
    if not candidates:
        raise RuntimeError('No results dir found with prefix {}'.format(prefix))
    candidates.sort()
    return '../results/' + candidates[-1]


def run_fold(network, fold, is_prelaunched_fold0=False):
    model_name = MODEL_NAME[network]
    baseline_ckpt = BASELINE_CKPTS[network][fold]
    meta_tag = META_TAG[network]
    stage_result = {'network': network, 'fold': fold}

    bad_csv = '../per_image_analysis_v2/overnight_run/{}_bad_images_fold{}.csv'.format(network, fold)
    manifest_csv = '../per_image_analysis_v2/overnight_run/{}_manifest_fold{}.csv'.format(network, fold)

    if not is_prelaunched_fold0:
        run_step('{}_fold{}_identify_bad'.format(network, fold), [
            PY, 'identify_bad_images_generic.py',
            '--ckpt', baseline_ckpt, '--model_name', model_name, '--fold', str(fold),
            '--image_subdir', 'Image_clahe', '--out_csv', bad_csv,
        ], os.path.join(STEP_LOG_DIR, '{}_fold{}_identify_bad.log'.format(network, fold)))

        run_step('{}_fold{}_augment'.format(network, fold), [
            PY, 'build_augmentations_generic.py', '--bad_csv', bad_csv, '--out_manifest', manifest_csv,
        ], os.path.join(STEP_LOG_DIR, '{}_fold{}_augment.log'.format(network, fold)))

        run_step('{}_fold{}_build_csv'.format(network, fold), [
            PY, 'build_fold_csv_generic.py', '--fold', str(fold), '--manifest_csv', manifest_csv, '--meta_tag', meta_tag,
        ], os.path.join(STEP_LOG_DIR, '{}_fold{}_build_csv.log'.format(network, fold)))

        exp_name = 'isic2017_{}_badaug_k{}'.format(network.lower(), fold)
        train_cmd = [
            PY, '-u', 'multi_train_adapt.py',
            '--exp_name', exp_name, '--config_yml', 'Configs/multi_train_local.yml',
            '--model', model_name, '--batch_size', '16', '--dataset', 'isic2017', '--k_fold', str(fold),
            '--meta_csv_name', 'meta_isic2017_train2000_{}.csv'.format(meta_tag),
            '--fixed_test_csv_name', 'meta_isic2017_test600.csv', '--image_subdir', 'Image_clahe',
        ]
        if USE_FTL[network]:
            train_cmd.append('--use_focal_tversky_loss')
        run_step('{}_fold{}_train'.format(network, fold), train_cmd,
                  os.path.join(STEP_LOG_DIR, '{}_fold{}_train.log'.format(network, fold)))
        exp_dir = find_exp_dir('{}_{}_'.format(exp_name, model_name))
    else:
        exp_dir = PRELAUNCHED_FOLD0[network]

    pilot_ckpt = os.path.join(exp_dir, 'best.pth')
    decision_json = '../per_image_analysis_v2/overnight_run/{}_fold{}_decision.json'.format(network, fold)
    per_image_csv = '../per_image_analysis_v2/overnight_run/{}_fold{}_pilot_per_image.csv'.format(network, fold)
    run_step('{}_fold{}_eval'.format(network, fold), [
        PY, 'eval_pilot_vs_baseline.py',
        '--pilot_ckpt', pilot_ckpt, '--model_name', model_name, '--network', network,
        '--baseline_stage', BASELINE_STAGE[network], '--fold', str(fold), '--image_subdir', 'Image_clahe',
        '--out_json', decision_json, '--out_csv', per_image_csv,
    ], os.path.join(STEP_LOG_DIR, '{}_fold{}_eval.log'.format(network, fold)))

    with open(decision_json) as f:
        stage_result['result'] = json.load(f)
    stage_result['exp_dir'] = exp_dir
    return stage_result


def run_network_sweep(network):
    log('=== Starting {} pilot (fold 0) ==='.format(network))
    fold0 = run_fold(network, 0, is_prelaunched_fold0=(network in PRELAUNCHED_FOLD0))
    status['stages'].append(fold0)
    save_status()

    decision = fold0['result']['decision']
    full = fold0['result']['full_test600']
    bad = fold0['result']['bad_subset']
    log('{} PILOT DECISION: {} (full test600 mean_diff={:+.4f}, p={:.4g}; bad-subset mean_diff={})'.format(
        network, decision.upper(), full['mean_diff'], full['p_value_one_sided'],
        bad.get('mean_diff', 'n/a')))

    if decision != 'continue':
        log('{}: pilot flat/negative. STOPPING sweep for {} per instructions.'.format(network, network))
        return False

    log('{}: pilot positive. Proceeding to full 5-fold sweep.'.format(network))
    for fold in [1, 2, 3, 4]:
        log('=== {} fold {} ==='.format(network, fold))
        res = run_fold(network, fold)
        status['stages'].append(res)
        save_status()

    # aggregate 5-fold summary
    pilot_means = [s['result']['full_test600']['pilot_mean'] for s in status['stages'] if s['network'] == network]
    baseline_means = [s['result']['full_test600']['baseline_mean'] for s in status['stages'] if s['network'] == network]
    import statistics
    summary = {
        'network': network, 'pilot_5fold_mean': statistics.mean(pilot_means),
        'pilot_5fold_std': statistics.stdev(pilot_means), 'baseline_5fold_mean': statistics.mean(baseline_means),
        'baseline_5fold_std': statistics.stdev(baseline_means),
    }
    log('{} 5-FOLD SUMMARY: pilot={:.4f}+/-{:.4f} baseline={:.4f}+/-{:.4f}'.format(
        network, summary['pilot_5fold_mean'], summary['pilot_5fold_std'],
        summary['baseline_5fold_mean'], summary['baseline_5fold_std']))
    status.setdefault('summaries', []).append(summary)
    save_status()
    return True


def main():
    try:
        wait_for_prelaunched_training('isic2017_swinunet_clahe_pilot_badaug_k0', 'SwinUnet')
        swinunet_full_sweep = run_network_sweep('SwinUnet')

        if swinunet_full_sweep:
            log('SwinUnet full sweep complete. Proceeding to AViT.')
            run_network_sweep('AViT')
        else:
            log('SwinUnet sweep stopped at pilot; per instructions, NOT proceeding to AViT.')

        status['final'] = 'COMPLETE'
        save_status()
        log('ORCHESTRATOR COMPLETE')
    except Exception as e:
        status['final'] = 'FAILED'
        status['error'] = str(e)
        status['traceback'] = traceback.format_exc()
        save_status()
        log('ORCHESTRATOR FAILED: {}'.format(e))
        log(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
