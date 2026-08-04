"""
Test-Time Augmentation (TTA) inference-only evaluation.

For each test image: run the network 5 times (original, horizontal flip, vertical
flip, 90-degree rotation, 270-degree rotation), take the sigmoid probability map
from each pass, apply the exact inverse transform to bring each probability map
back into the original image's orientation, average the 5 probability maps (soft
voting), then threshold once at 0.5 to get the final binary mask.

Evaluates against the SAME fixed 600-image official test set used throughout this
project, using the same batch_size=5 (from Configs/multi_train_local.yml test.batch_size)
and the same medpy.metric.binary dice/iou accumulation scheme as multi_train_adapt.py's
test(), so the recomputed "without TTA" numbers here are directly comparable to
(and should closely match) the historical 5-fold mean baselines.

No training. Takes 5 already-trained fold checkpoints (best.pth) for one network and
reports per-fold and 5-fold-mean Dice/IOU with vs. without TTA.
"""
import argparse
import numpy as np
import torch
import yaml
import medpy.metric.binary as metrics

from Datasets.create_dataset import Dataset_wrap_csv
from Utils.pieces import DotDict
from Utils.tta import TTA_VIEWS, tta_forward as _shared_tta_forward


def build_model(model_name, config):
    if model_name == 'SwinUnet':
        from Models.Transformer.SwinUnet import SwinUnet
        model = SwinUnet(img_size=config.data.img_size)
    elif model_name == 'SwinSeg_CNNprompt_adapt':
        from Models.Transformer.Swin_adapters import SwinSimpleSeg_CNNprompt_adapt
        model = SwinSimpleSeg_CNNprompt_adapt(
            img_size=config.data.img_size, pretrained=False,
            pretrained_swin_name=config.swin.name, pretrained_folder=config.pretrained_folder,
            embed_dim=config.swin.EMBED_DIM, drop_path_rate=config.swin.DROP_PATH_RATE,
            depths=config.swin.DEPTHS, num_heads=config.swin.NUM_HEADS,
            window_size=config.swin.WINDOW_SIZE, debug=False,
            adapt_method=config.model_adapt.adapt_method, num_domains=1)
    else:
        raise ValueError('Unsupported model for TTA script: {}'.format(model_name))
    return model.cuda().eval()


def forward_logits(model, model_name, x):
    if model_name == 'SwinUnet':
        return model(x)
    return model(x, d='0')['seg']


def tta_forward(model, model_name, img):
    """Thin wrapper preserving this script's original (model, model_name, img) call
    signature, delegating the actual 5-view soft-voting to the shared Utils.tta
    implementation (also used by multi_train_adapt.py's --use_tta), so there is a
    single source of truth for the TTA view/inverse logic."""
    return _shared_tta_forward(lambda x: forward_logits(model, model_name, x), img)


def evaluate_fold(model, model_name, test_loader):
    """Single pass over the test set. Accumulates both TTA and no-TTA dice/iou using
    the same per-batch-aggregate-then-weight-by-batch_len scheme as multi_train_adapt.py's
    test(), so results are apples-to-apples with the historical recorded baselines."""
    dice_tta_sum = iou_tta_sum = 0.0
    dice_notta_sum = iou_notta_sum = 0.0
    num_test = 0

    for batch in test_loader:
        img = batch['image'].cuda().float()
        label = batch['label'].cuda().float()
        batch_len = img.shape[0]

        avg_prob, orig_prob = tta_forward(model, model_name, img)

        label_np = label.cpu().numpy()

        pred_tta = (avg_prob.cpu().numpy() > 0.5)
        dice_tta_sum += metrics.dc(pred_tta, label_np) * batch_len
        iou_tta_sum += metrics.jc(pred_tta, label_np) * batch_len

        pred_notta = (orig_prob.cpu().numpy() > 0.5)
        dice_notta_sum += metrics.dc(pred_notta, label_np) * batch_len
        iou_notta_sum += metrics.jc(pred_notta, label_np) * batch_len

        num_test += batch_len

    return {
        'dice_tta': dice_tta_sum / num_test,
        'iou_tta': iou_tta_sum / num_test,
        'dice_notta': dice_notta_sum / num_test,
        'iou_notta': iou_notta_sum / num_test,
        'num_test': num_test,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TTA inference-only evaluation over 5 fold checkpoints')
    parser.add_argument('--model', type=str, required=True, choices=['SwinUnet', 'SwinSeg_CNNprompt_adapt'])
    parser.add_argument('--checkpoints', type=str, nargs=5, required=True,
                         help='5 best.pth paths in fold order k0 k1 k2 k3 k4')
    parser.add_argument('--config_yml', type=str, default='Configs/multi_train_local.yml')
    parser.add_argument('--dataset', type=str, default='isic2017')
    parser.add_argument('--meta_csv_name', type=str, default='meta_isic2017_train2000.csv')
    parser.add_argument('--fixed_test_csv_name', type=str, default='meta_isic2017_test600.csv')
    parser.add_argument('--image_subdir', type=str, default='Image_clahe')
    parser.add_argument('--output_csv', type=str, required=True)
    args = parser.parse_args()

    config = yaml.load(open(args.config_yml), Loader=yaml.FullLoader)
    config = DotDict(config)

    # Test set is fixed/identical across folds (fixed_test_csv_name), so load it once.
    datas = Dataset_wrap_csv(k_fold='0', use_old_split=True, img_size=config.data.img_size,
                              dataset_name=args.dataset, split_ratio=config.data.split_ratio,
                              train_aug=False, data_folder=config.data.data_folder,
                              meta_csv_name=args.meta_csv_name, fixed_test_csv_name=args.fixed_test_csv_name,
                              image_subdir=args.image_subdir)
    test_data = datas['test']
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.test.batch_size,
                                               shuffle=False, num_workers=config.test.num_workers,
                                               pin_memory=True, drop_last=False)
    print('Fixed test set: {} samples, batch_size={}'.format(len(test_data), config.test.batch_size))

    fold_results = []
    for k, ckpt_path in enumerate(args.checkpoints):
        print('=== Fold k{}: {} ==='.format(k, ckpt_path))
        model = build_model(args.model, config)
        model.load_state_dict(torch.load(ckpt_path))
        model.eval()
        result = evaluate_fold(model, args.model, test_loader)
        result['fold'] = k
        result['checkpoint'] = ckpt_path
        fold_results.append(result)
        print('  no-TTA: Dice={:.4f} IOU={:.4f} | TTA: Dice={:.4f} IOU={:.4f}'.format(
            result['dice_notta'], result['iou_notta'], result['dice_tta'], result['iou_tta']))
        del model
        torch.cuda.empty_cache()

    dice_tta = np.array([r['dice_tta'] for r in fold_results])
    iou_tta = np.array([r['iou_tta'] for r in fold_results])
    dice_notta = np.array([r['dice_notta'] for r in fold_results])
    iou_notta = np.array([r['iou_notta'] for r in fold_results])

    print('========================================================================================')
    print('{} 5-fold summary ({} test images):'.format(args.model, fold_results[0]['num_test']))
    print('  no-TTA: Dice {:.4f}+/-{:.4f}  IOU {:.4f}+/-{:.4f}'.format(
        dice_notta.mean(), dice_notta.std(ddof=1), iou_notta.mean(), iou_notta.std(ddof=1)))
    print('  TTA:    Dice {:.4f}+/-{:.4f}  IOU {:.4f}+/-{:.4f}'.format(
        dice_tta.mean(), dice_tta.std(ddof=1), iou_tta.mean(), iou_tta.std(ddof=1)))

    import pandas as pd
    rows = [{'fold': r['fold'], 'checkpoint': r['checkpoint'],
             'dice_notta': r['dice_notta'], 'iou_notta': r['iou_notta'],
             'dice_tta': r['dice_tta'], 'iou_tta': r['iou_tta']} for r in fold_results]
    rows.append({'fold': 'mean', 'checkpoint': '',
                 'dice_notta': dice_notta.mean(), 'iou_notta': iou_notta.mean(),
                 'dice_tta': dice_tta.mean(), 'iou_tta': iou_tta.mean()})
    rows.append({'fold': 'std', 'checkpoint': '',
                 'dice_notta': dice_notta.std(ddof=1), 'iou_notta': iou_notta.std(ddof=1),
                 'dice_tta': dice_tta.std(ddof=1), 'iou_tta': iou_tta.std(ddof=1)})
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print('Saved: {}'.format(args.output_csv))
