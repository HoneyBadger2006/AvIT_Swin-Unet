"""
Post-processing pipeline (Prof. Samavi's spec), applied to TTA-averaged predictions
from each network's current best checkpoint config:
  SwinUnet: CLAHE + TTA                 (isic2017_swinunet_clahe_k{0-4})
  AViT:     CLAHE + compound FTL + TTA  (isic2017_avit_clahe_ftl_k{0-4})

Stages, each applied on top of the previous mask unless noted:
  1. raw      - TTA-averaged (5-view soft-voted) probability map, thresholded at 0.5.
                No postprocessing. This IS the current best-config prediction.
  2. +morph   - cv2.morphologyEx: MORPH_OPEN then MORPH_CLOSE (5x5 ellipse kernel by
                default), applied to the raw binary mask. Removes small spurious
                detections, fills small holes.
  3. +Method A - starting from the +morph mask: connected-component filtering that
                keeps ONLY the single largest component (justified since ISIC has
                exactly one lesion per image).
  4. +Method B - starting from the +morph mask (alternative to Method A, not stacked
                on top of it): discard any connected component with area less than
                0.1x the mean ground-truth lesion area, computed once over the 2000
                training images (in the same 224x224 resized coordinate space used
                for prediction/label comparison throughout this project).

Inference-only: loads existing best.pth checkpoints for all 5 folds of both networks,
evaluates on the fixed 600-image official test set (Image_clahe subdir, since both
current-best configs use CLAHE). No training. Reuses the shared Utils/tta.py TTA
implementation (via tta_inference.py's build_model/tta_forward) for exact consistency
with all previously reported TTA numbers.
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import cv2
import yaml
import medpy.metric.binary as metrics

from Datasets.create_dataset import Dataset_wrap_csv
from Utils.pieces import DotDict
from tta_inference import build_model, tta_forward

SWINUNET_CLAHE_CKPTS = {
    0: '../results/isic2017_swinunet_clahe_k0_SwinUnet_20260728_0301/best.pth',
    1: '../results/isic2017_swinunet_clahe_k1_SwinUnet_20260728_1556/best.pth',
    2: '../results/isic2017_swinunet_clahe_k2_SwinUnet_20260728_1645/best.pth',
    3: '../results/isic2017_swinunet_clahe_k3_SwinUnet_20260728_1733/best.pth',
    4: '../results/isic2017_swinunet_clahe_k4_SwinUnet_20260728_1823/best.pth',
}

AVIT_CLAHE_FTL_CKPTS = {
    0: '../results/isic2017_avit_clahe_ftl_k0_SwinSeg_CNNprompt_adapt_20260730_1139/best.pth',
    1: '../results/isic2017_avit_clahe_ftl_k1_SwinSeg_CNNprompt_adapt_20260730_1656/best.pth',
    2: '../results/isic2017_avit_clahe_ftl_k2_SwinSeg_CNNprompt_adapt_20260730_1904/best.pth',
    3: '../results/isic2017_avit_clahe_ftl_k3_SwinSeg_CNNprompt_adapt_20260730_2012/best.pth',
    4: '../results/isic2017_avit_clahe_ftl_k4_SwinSeg_CNNprompt_adapt_20260730_2118/best.pth',
}

NETWORKS = {
    'SwinUnet': {'model_name': 'SwinUnet', 'ckpts': SWINUNET_CLAHE_CKPTS},
    'AViT': {'model_name': 'SwinSeg_CNNprompt_adapt', 'ckpts': AVIT_CLAHE_FTL_CKPTS},
}


def morph_open_close(mask01, kernel_size=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask01, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def keep_largest_component(mask01):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
    if num <= 1:
        return mask01  # no foreground component at all
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    return (labels == largest_label).astype(np.uint8)


def filter_by_min_area(mask01, min_area):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
    out = np.zeros_like(mask01)
    for lbl in range(1, num):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            out[labels == lbl] = 1
    return out


def compute_avg_train_gt_area(data_root, img_size):
    df = pd.read_csv(os.path.join(data_root, 'meta_isic2017_train2000.csv'), dtype={'ID': str})
    areas = []
    for image_id in df['ID']:
        label_raw = (np.load(os.path.join(data_root, 'Label', '{}.npy'.format(image_id))) > 0.5).astype('uint8')
        label_resized = cv2.resize(label_raw, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        areas.append(int(label_resized.sum()))
    return float(np.mean(areas))


def process_stages(raw_pred01, gt_bool, kernel_size, min_area):
    dice_raw = metrics.dc(raw_pred01.astype(bool), gt_bool)

    morph_pred = morph_open_close(raw_pred01, kernel_size)
    dice_morph = metrics.dc(morph_pred.astype(bool), gt_bool)

    methodA_pred = keep_largest_component(morph_pred)
    dice_A = metrics.dc(methodA_pred.astype(bool), gt_bool)

    methodB_pred = filter_by_min_area(morph_pred, min_area)
    dice_B = metrics.dc(methodB_pred.astype(bool), gt_bool)

    return dice_raw, dice_morph, dice_A, dice_B


def summarize(df, label):
    print('--- {} ---'.format(label))
    for network in df['network'].unique():
        sub = df[df['network'] == network]
        fold_means = sub.groupby('fold')[['dice_raw', 'dice_morph', 'dice_methodA', 'dice_methodB']].mean()
        mean = fold_means.mean()
        std = fold_means.std(ddof=1)
        n_images = len(sub) // sub['fold'].nunique()
        print('{} (n={} images/fold, {} folds):'.format(network, n_images, sub['fold'].nunique()))
        print('  raw:        {:.4f} +/- {:.4f}'.format(mean['dice_raw'], std['dice_raw']))
        print('  +morph:     {:.4f} +/- {:.4f}'.format(mean['dice_morph'], std['dice_morph']))
        print('  +Method A:  {:.4f} +/- {:.4f}'.format(mean['dice_methodA'], std['dice_methodA']))
        print('  +Method B:  {:.4f} +/- {:.4f}'.format(mean['dice_methodB'], std['dice_methodB']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Post-processing pipeline on TTA predictions, both networks, 5-fold')
    parser.add_argument('--config_yml', type=str, default='Configs/multi_train_local.yml')
    parser.add_argument('--data_root', type=str, default='C:/Users/quanp/Downloads/ISIC 2017/data/isic2017')
    parser.add_argument('--dataset', type=str, default='isic2017')
    parser.add_argument('--meta_csv_name', type=str, default='meta_isic2017_train2000.csv')
    parser.add_argument('--fixed_test_csv_name', type=str, default='meta_isic2017_test600.csv')
    parser.add_argument('--image_subdir', type=str, default='Image_clahe')
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--bad_dice_threshold', type=float, default=0.7)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = DotDict(yaml.load(open(args.config_yml), Loader=yaml.FullLoader))
    img_size = config.data.img_size

    print('Computing mean GT lesion area over 2000 training images...')
    avg_area = compute_avg_train_gt_area(args.data_root, img_size)
    min_area = 0.1 * avg_area
    print('Mean train GT lesion area (224x224 px): {:.1f}  ->  Method B min_area threshold: {:.1f}'.format(
        avg_area, min_area))

    datas = Dataset_wrap_csv(k_fold='0', use_old_split=True, img_size=img_size,
                              dataset_name=args.dataset, split_ratio=config.data.split_ratio,
                              train_aug=False, data_folder=config.data.data_folder,
                              meta_csv_name=args.meta_csv_name, fixed_test_csv_name=args.fixed_test_csv_name,
                              image_subdir=args.image_subdir)
    test_data = datas['test']
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.test.batch_size,
                                               shuffle=False, num_workers=config.test.num_workers,
                                               pin_memory=True, drop_last=False)
    print('Fixed test set: {} samples, batch_size={}'.format(len(test_data), config.test.batch_size))

    rows = []
    for network_name, spec in NETWORKS.items():
        model_name = spec['model_name']
        for fold, ckpt_path in spec['ckpts'].items():
            print('=== {} fold k{}: {} ==='.format(network_name, fold, ckpt_path))
            model = build_model(model_name, config)
            model.load_state_dict(torch.load(ckpt_path))
            model.eval()

            for batch in test_loader:
                img = batch['image'].cuda().float()
                label = batch['label'].cuda().float()
                ids = batch['ID']

                avg_prob, _ = tta_forward(model, model_name, img)
                avg_prob_np = avg_prob.cpu().numpy()
                label_np = label.cpu().numpy()

                for i in range(img.shape[0]):
                    raw_pred = (avg_prob_np[i, 0] > 0.5).astype(np.uint8)
                    gt = (label_np[i, 0] > 0.5)
                    dice_raw, dice_morph, dice_A, dice_B = process_stages(
                        raw_pred, gt, args.kernel_size, min_area)
                    rows.append({
                        'network': network_name, 'fold': fold, 'image_id': ids[i],
                        'dice_raw': dice_raw, 'dice_morph': dice_morph,
                        'dice_methodA': dice_A, 'dice_methodB': dice_B,
                    })

            del model
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    per_image_csv = os.path.join(args.output_dir, 'postprocess_per_image.csv')
    df.to_csv(per_image_csv, index=False)
    print('Saved per-image results: {}'.format(per_image_csv))

    print('========================================================================================')
    summarize(df, 'ALL 600 TEST IMAGES')

    bad_df = df[df['dice_raw'] < args.bad_dice_threshold]
    bad_csv = os.path.join(args.output_dir, 'postprocess_bad_subset.csv')
    bad_df.to_csv(bad_csv, index=False)
    print('========================================================================================')
    summarize(bad_df, 'BAD-IMAGE SUBSET (raw Dice < {})'.format(args.bad_dice_threshold))
    for network in bad_df['network'].unique():
        sub = bad_df[bad_df['network'] == network]
        print('{}: {} (image, fold) pairs below threshold out of {}'.format(
            network, len(sub), len(df[df['network'] == network])))

    print('POSTPROCESSING PIPELINE COMPLETE')
