"""
Tests the "Dice vs IoU threshold mix-up" hypothesis for why our hard-image count
differs from Prof. Samavi's expectation. Re-runs inference with the SAME allin
final.pth checkpoint (the one that originally produced the 23-bad-image, full-2000,
no-CV, in-sample result) and computes BOTH Dice (medpy.metric.binary.dc) and IoU
(medpy.metric.binary.jc) from the SAME prediction/label array pair in the SAME loop
iteration -- so there is no possibility of the two metrics drifting out of pairing
with each other or with the image IDs.

Reports:
  - count(IoU < 0.7)
  - count(Dice < 0.824)  (the exact Dice-equivalent of IoU=0.7, since for binary
    masks IoU = Dice/(2-Dice) exactly, so IoU<0.7 <=> Dice<0.8235... elementwise)
  - cross-check that recomputed per-image Dice matches the previously saved
    allin_final_per_image_dice.csv (reproducibility check)
  - cross-check that the IoU<0.7 set and Dice<0.824 set are IDENTICAL (they must
    be, by the exact algebraic relationship, for every individual image -- any
    mismatch would indicate a real bug, not just threshold semantics)
"""
import numpy as np
import pandas as pd
import torch
import yaml
import medpy.metric.binary as metrics

from Datasets.create_dataset import Dataset_wrap_csv
from Utils.pieces import DotDict
from tta_inference import build_model, forward_logits

CKPT = '../results/isic2017_swinunet_clahe_allin_SwinUnet_20260817_0324/final.pth'
PREV_DICE_CSV = '../per_image_analysis_v2/bad_image_augmentation/allin_final_per_image_dice.csv'
OUT_CSV = '../per_image_analysis_v2/bad_image_augmentation/iou_vs_dice_check_per_image.csv'
IOU_EQUIV_THRESHOLD_DICE = 2 * 0.7 / (1 + 0.7)  # exact Dice equivalent of IoU=0.7


def main():
    config = DotDict(yaml.load(open('Configs/multi_train_local.yml'), Loader=yaml.FullLoader))
    datas = Dataset_wrap_csv(k_fold='allin', use_old_split=True, img_size=config.data.img_size,
                              dataset_name='isic2017', split_ratio=config.data.split_ratio,
                              train_aug=False, data_folder=config.data.data_folder,
                              meta_csv_name='meta_isic2017_train2000.csv', fixed_test_csv_name=None,
                              image_subdir='Image_clahe')
    train_ds = datas['train']
    print('Pool: {} images'.format(len(train_ds)))
    print('Exact Dice-equivalent of IoU=0.7: {:.6f} (rounds to 0.824)'.format(IOU_EQUIV_THRESHOLD_DICE))

    loader = torch.utils.data.DataLoader(train_ds, batch_size=config.test.batch_size, shuffle=False,
                                          num_workers=config.test.num_workers, pin_memory=True, drop_last=False)

    model = build_model('SwinUnet', config)
    model.load_state_dict(torch.load(CKPT))
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in loader:
            img = batch['image'].cuda().float()
            label = batch['label'].cuda().float()
            ids = batch['ID']
            logits = forward_logits(model, 'SwinUnet', img)
            prob = torch.sigmoid(logits)
            pred = (prob.cpu().numpy() > 0.5)
            label_np = label.cpu().numpy() > 0.5
            for i in range(img.shape[0]):
                p = pred[i, 0]
                g = label_np[i, 0]
                dice = metrics.dc(p, g)
                iou = metrics.jc(p, g)
                # manual recompute from raw pixel counts, as an independent triple-check
                tp = np.logical_and(p, g).sum()
                fp = np.logical_and(p, np.logical_not(g)).sum()
                fn = np.logical_and(np.logical_not(p), g).sum()
                dice_manual = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
                iou_manual = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
                rows.append({'image_id': ids[i], 'dice': dice, 'iou': iou,
                             'dice_manual': dice_manual, 'iou_manual': iou_manual,
                             'pred_area': int(p.sum()), 'gt_area': int(g.sum())})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print('n =', len(df))

    # --- sanity check 1: medpy vs manual pixel-count recomputation ---
    dice_match = np.allclose(df['dice'], df['dice_manual'], atol=1e-6)
    iou_match = np.allclose(df['iou'], df['iou_manual'], atol=1e-6)
    print('\n[Sanity check 1] medpy dc()/jc() matches manual TP/FP/FN recomputation: dice={} iou={}'.format(dice_match, iou_match))
    assert dice_match and iou_match

    # --- sanity check 2: exact algebraic Dice<->IoU relationship holds per-image ---
    predicted_iou_from_dice = df['dice'] / (2 - df['dice'])
    algebra_match = np.allclose(predicted_iou_from_dice, df['iou'], atol=1e-6)
    print('[Sanity check 2] IoU == Dice/(2-Dice) holds for every image: {}'.format(algebra_match))
    assert algebra_match

    # --- sanity check 3: reproducibility vs the previously saved allin dice CSV ---
    prev = pd.read_csv(PREV_DICE_CSV, dtype={'image_id': str}).rename(columns={'dice': 'prev_dice'})
    merged = pd.merge(df, prev, on='image_id', how='inner')
    assert len(merged) == 2000, 'expected 2000 matched images, got {}'.format(len(merged))
    reproduced = np.allclose(merged['dice'], merged['prev_dice'], atol=1e-6)
    max_diff = (merged['dice'] - merged['prev_dice']).abs().max()
    print('[Sanity check 3] Recomputed Dice reproduces the original saved allin_final_per_image_dice.csv: {} (max abs diff={:.2e})'.format(reproduced, max_diff))

    # --- sanity check 4: no image/label pairing issue -- pred and gt areas both nonzero/sane ---
    n_empty_gt = (df['gt_area'] == 0).sum()
    n_empty_pred = (df['pred_area'] == 0).sum()
    print('[Sanity check 4] Images with empty GT mask: {} | Images with empty prediction: {}'.format(n_empty_gt, n_empty_pred))

    # --- the actual hypothesis test ---
    n_iou_bad = (df['iou'] < 0.7).sum()
    n_dice824_bad = (df['dice'] < IOU_EQUIV_THRESHOLD_DICE).sum()
    n_dice70_bad = (df['dice'] < 0.7).sum()
    same_set = set(df[df['iou'] < 0.7]['image_id']) == set(df[df['dice'] < IOU_EQUIV_THRESHOLD_DICE]['image_id'])

    print('\n=== HYPOTHESIS TEST RESULTS ===')
    print('IoU < 0.7 count:      {} / {}'.format(n_iou_bad, len(df)))
    print('Dice < 0.824 count:   {} / {}  (exact threshold used: {:.6f})'.format(n_dice824_bad, len(df), IOU_EQUIV_THRESHOLD_DICE))
    print('[reference] Dice < 0.7 count (the original metric/threshold used throughout this project): {} / {}'.format(n_dice70_bad, len(df)))
    print('IoU<0.7 and Dice<0.824 identify the EXACT SAME set of images: {}'.format(same_set))
    print('\nEVAL_IOU_VS_DICE_DONE')


if __name__ == '__main__':
    main()
