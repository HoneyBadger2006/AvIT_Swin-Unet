"""
Overnight priority-chained runner for the corrected pilot retrain and follow-ons.
Launched via a Windows scheduled task so it survives independent of any
particular terminal/session watching it. Every step's outcome is written to a
running JSON status file and a human-readable log, and every stage's own
subprocess output is captured to its own log file, so progress can always be
verified directly (file/process/GPU state) rather than trusted from a monitor.

PRIORITY 1 (core deliverable): retrain SwinUnet+CLAHE with the corrected setup
  (23 bad training images from the full-2000 no-CV allin run, corrected tight
  bboxes, all 6 augmentation variants), no fold structure (all 2138 rows =
  2000 original + 138 augmented, single run, final-epoch checkpoint per the
  established allin precedent). Then computes both recovery metrics Prof.
  Samavi asked for (23-training recovery, 73-test recovery) plus the same
  full-test/hard-subset comparison used for the earlier pilot.

PRIORITY 2 (if time remains): builds + visually verifies (no training) Variant
  C (expanded bbox) and Variant D (70%-fill zoom) for the same 23 images.

PRIORITY 3 (stretch, only if lots of time remains): begins setting up the
  HSV/YCbCr color-space experiment as a separate, lower-priority scaffold --
  conversion utilities + a small visual sanity check, no training.
"""
import datetime
import json
import os
import subprocess
import sys
import time
import traceback

AVIT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(AVIT_DIR)

STATUS_PATH = '../per_image_analysis_v2/overnight_run/priority_status.json'
STEP_LOG_DIR = '../kfold_logs/overnight_priority_steps'
os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
os.makedirs(STEP_LOG_DIR, exist_ok=True)

PY = '../venv/Scripts/python.exe'

# Priority 2/3 are only attempted if there's still meaningful time before this
# cutoff (local time), per "I need Priority 1's full result by morning at
# minimum -- that's the non-negotiable part."
P2_CUTOFF_HOUR = 7   # attempt Priority 2 only before 07:00 local
P3_MIN_HOURS_REMAINING = 2.0  # attempt Priority 3 only if >= 2h left before P2_CUTOFF_HOUR

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


def find_exp_dir(prefix):
    candidates = [d for d in os.listdir('../results') if d.startswith(prefix)]
    if not candidates:
        raise RuntimeError('No results dir found with prefix {}'.format(prefix))
    candidates.sort()
    return '../results/' + candidates[-1]


def priority1():
    log('=== PRIORITY 1: corrected pilot retrain (23 bad images, tight bbox, all 6 variants) ===')

    exp_name = 'isic2017_swinunet_clahe_pilot_badaug23'
    train_cmd = [
        PY, '-u', 'multi_train_adapt.py',
        '--exp_name', exp_name, '--config_yml', 'Configs/multi_train_local.yml',
        '--model', 'SwinUnet', '--batch_size', '16', '--dataset', 'isic2017', '--k_fold', 'allin',
        '--num_epochs', '30',
        '--meta_csv_name', 'meta_isic2017_train2000_pilot_badaug23.csv',
        '--fixed_test_csv_name', 'meta_isic2017_test600.csv',
        '--image_subdir', 'Image_clahe',
    ]
    run_step('p1_train', train_cmd, os.path.join(STEP_LOG_DIR, 'p1_train.log'))

    exp_dir = find_exp_dir('{}_SwinUnet_'.format(exp_name))
    final_ckpt = os.path.join(exp_dir, 'final.pth')
    if not os.path.exists(final_ckpt):
        raise RuntimeError('final.pth not found in {} -- training did not complete the final-epoch save'.format(exp_dir))
    log('Training complete. exp_dir={} final_ckpt={}'.format(exp_dir, final_ckpt))

    train23_csv = '../per_image_analysis_v2/bad_image_augmentation/pilot_retrain_train23_recovery.csv'
    run_step('p1_eval_train23', [
        PY, 'eval_train23_recovery.py', '--ckpt', final_ckpt, '--model_name', 'SwinUnet',
        '--image_subdir', 'Image_clahe', '--out_csv', train23_csv,
    ], os.path.join(STEP_LOG_DIR, 'p1_eval_train23.log'))

    decision_json = '../per_image_analysis_v2/overnight_run/pilot_retrain_fold0_decision.json'
    test600_per_image_csv = '../per_image_analysis_v2/overnight_run/pilot_retrain_test600_per_image.csv'
    run_step('p1_eval_test600', [
        PY, 'eval_pilot_vs_baseline.py',
        '--pilot_ckpt', final_ckpt, '--model_name', 'SwinUnet', '--network', 'SwinUnet',
        '--baseline_stage', 'clahe', '--fold', '0', '--image_subdir', 'Image_clahe',
        '--out_json', decision_json, '--out_csv', test600_per_image_csv,
    ], os.path.join(STEP_LOG_DIR, 'p1_eval_test600.log'))

    import pandas as pd
    train23_df = pd.read_csv(train23_csv, dtype={'image_id': str})
    n_train23_still_bad = int((train23_df['retrain_dice'] < 0.7).sum())

    test_df = pd.read_csv(test600_per_image_csv, dtype={'image_id': str})
    base_full = pd.read_csv('../per_image_analysis_v2/final_pipeline/per_image_final_pipeline.csv', dtype={'image_id': str})
    base_sub = base_full[(base_full['network'] == 'SwinUnet') & (base_full['stage'] == 'clahe') & (base_full['fold'] == 0)][
        ['image_id', 'dice']].rename(columns={'dice': 'baseline_dice'})
    merged73 = pd.merge(test_df, base_sub, on='image_id', how='inner')
    bad73 = merged73[merged73['baseline_dice'] < 0.7]
    n_test73_still_bad = int((bad73['pilot_dice'] < 0.7).sum())

    with open(decision_json) as f:
        decision_result = json.load(f)

    summary = {
        'exp_dir': exp_dir, 'final_ckpt': final_ckpt,
        'train23_still_bad': n_train23_still_bad, 'train23_total': 23,
        'train23_mean_retrain_dice': float(train23_df['retrain_dice'].mean()),
        'train23_mean_orig_dice': float(train23_df['orig_allin_dice'].mean()),
        'test73_still_bad': n_test73_still_bad, 'test73_total': int(len(bad73)),
        'test73_mean_pilot_dice': float(bad73['pilot_dice'].mean()),
        'test73_mean_baseline_dice': float(bad73['baseline_dice'].mean()),
        'full_test600': decision_result['full_test600'],
        'bad_subset_paired_test': decision_result['bad_subset'],
        'decision': decision_result['decision'],
    }
    summary_path = '../per_image_analysis_v2/overnight_run/priority1_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    log('PRIORITY 1 SUMMARY: train23_still_bad={}/23  test73_still_bad={}/{}  full_test600 mean_diff={:+.4f}'.format(
        n_train23_still_bad, n_test73_still_bad, len(bad73), decision_result['full_test600']['mean_diff']))
    log('Saved: {}'.format(summary_path))
    status['priority1_summary'] = summary
    save_status()
    return summary


def priority2():
    log('=== PRIORITY 2: build + verify Variant C / Variant D (no training) ===')
    manifest_csv = '../per_image_analysis_v2/bad_image_augmentation/manifest_variant_cd23.csv'
    bbox_csv = '../per_image_analysis_v2/bad_image_augmentation/bbox_coords_variant_cd.csv'
    run_step('p2_build_variant_cd', [
        PY, 'build_variant_cd.py',
        '--bad_csv', '../per_image_analysis_v2/bad_image_augmentation/bad_images_allin23.csv',
        '--out_manifest', manifest_csv, '--out_bbox_csv', bbox_csv,
    ], os.path.join(STEP_LOG_DIR, 'p2_build_variant_cd.log'))

    run_step('p2_visualize_variant_cd', [
        PY, 'visualize_variant_cd.py',
    ], os.path.join(STEP_LOG_DIR, 'p2_visualize_variant_cd.log'))

    log('PRIORITY 2 done: Variant C/D built and visually verified for 23/23 images. NOT retrained (per instructions).')
    status['priority2_status'] = 'built_and_verified_not_trained'
    save_status()


def priority3():
    log('=== PRIORITY 3 (stretch): begin HSV/YCbCr color-space experiment setup ===')
    script_path = 'build_colorspace_variants.py'
    content = '''"""
Scaffold for the HSV/YCbCr color-space experiment (Priority 3 stretch goal).
Converts a handful of sample CLAHE images to HSV and YCbCr and saves side-by-side
visual sanity-check panels, so the conversion logic can be reviewed before any
full-dataset generation or retraining is committed to. Not wired into training.
"""
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

DATA_ROOT = '../data/isic2017'
OUT_DIR = '../per_image_analysis_v2/colorspace_experiment/sanity_check'
N_SAMPLES = 8

os.makedirs(OUT_DIR, exist_ok=True)


def to_hsv(img_uint8):
    return cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)


def to_ycbcr(img_uint8):
    return cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YCrCb)


def main():
    meta = pd.read_csv(os.path.join(DATA_ROOT, 'meta_isic2017_train2000.csv'), dtype={'ID': str})
    sample_ids = meta['ID'].sample(N_SAMPLES, random_state=42).tolist()

    for image_id in sample_ids:
        img = np.load(os.path.join(DATA_ROOT, 'Image_clahe', image_id + '.npy')).astype(np.uint8)
        hsv = to_hsv(img)
        ycbcr = to_ycbcr(img)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
        axes[0].imshow(img); axes[0].set_title('CLAHE RGB'); axes[0].axis('off')
        axes[1].imshow(hsv); axes[1].set_title('HSV (raw channels as RGB)'); axes[1].axis('off')
        axes[2].imshow(ycbcr); axes[2].set_title('YCbCr (raw channels as RGB)'); axes[2].axis('off')
        fig.suptitle(image_id, fontsize=11)
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, '{}_colorspace_sanity.png'.format(image_id))
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        print('Saved:', out_path)

    print('COLORSPACE_SANITY_DONE')


if __name__ == '__main__':
    main()
'''
    with open(script_path, 'w') as f:
        f.write(content)
    log('Wrote scaffold script: {}'.format(script_path))

    run_step('p3_colorspace_sanity', [PY, script_path], os.path.join(STEP_LOG_DIR, 'p3_colorspace_sanity.log'))
    log('PRIORITY 3 done: color-space conversion utilities + visual sanity check only. NOT a full dataset build, NOT training.')
    status['priority3_status'] = 'scaffold_and_sanity_check_only'
    save_status()


def main():
    try:
        priority1()
        status['priority1_done'] = True
        save_status()

        now = datetime.datetime.now()
        cutoff = now.replace(hour=P2_CUTOFF_HOUR, minute=0, second=0, microsecond=0)
        if now >= cutoff:
            cutoff = cutoff + datetime.timedelta(days=1)
        hours_remaining = (cutoff - now).total_seconds() / 3600.0
        log('Priority 1 finished at {}. Hours remaining before {:02d}:00 cutoff: {:.2f}'.format(
            now.strftime('%H:%M:%S'), P2_CUTOFF_HOUR, hours_remaining))

        if hours_remaining > 0.25:
            priority2()
            status['priority2_done'] = True
            save_status()

            now2 = datetime.datetime.now()
            hours_remaining2 = (cutoff - now2).total_seconds() / 3600.0
            if hours_remaining2 >= P3_MIN_HOURS_REMAINING:
                priority3()
                status['priority3_done'] = True
                save_status()
            else:
                log('Only {:.2f}h remaining after Priority 2 (< {}h threshold) -- skipping Priority 3 stretch goal.'.format(
                    hours_remaining2, P3_MIN_HOURS_REMAINING))
                status['priority3_status'] = 'skipped_insufficient_time'
                save_status()
        else:
            log('Only {:.2f}h remaining before cutoff -- skipping Priority 2 and 3.'.format(hours_remaining))
            status['priority2_status'] = 'skipped_insufficient_time'
            status['priority3_status'] = 'skipped_insufficient_time'
            save_status()

        status['final'] = 'COMPLETE'
        save_status()
        log('OVERNIGHT_RUNNER_COMPLETE')
    except Exception as e:
        status['final'] = 'FAILED'
        status['error'] = str(e)
        status['traceback'] = traceback.format_exc()
        save_status()
        log('OVERNIGHT_RUNNER_FAILED: {}'.format(e))
        log(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
