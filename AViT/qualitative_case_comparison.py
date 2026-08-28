"""
Qualitative pipeline-stage comparison for a single test image: runs one network
through 4 checkpoint stages (baseline, +CLAHE, +CLAHE+compound-FTL,
+CLAHE+FTL+TTA), saves each stage's binary predicted mask, and composes a single
labeled comparison figure (original image, ground truth, 4 predictions).

Uses the exact same preprocessing (Resize 224x224, norm01, ImageNet normalize) as
Datasets/create_dataset.py's SkinDataset_csv, and the same shared Utils/tta.py TTA
implementation used by multi_train_adapt.py's --use_tta and tta_inference.py, so
results are directly comparable to all prior reported numbers.
"""
import argparse
import os
import numpy as np
import torch
import yaml
import cv2
import matplotlib.pyplot as plt
import medpy.metric.binary as metrics
import albumentations as A
from torchvision import transforms as T

from Utils.pieces import DotDict
from Utils.tta import tta_forward
from Datasets.create_dataset import norm01
from tta_inference import build_model

IMG_SIZE = 224
NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
RESIZE = A.Resize(IMG_SIZE, IMG_SIZE)


def load_tensor_and_display(data_root, image_subdir, image_id):
    img_path = os.path.join(data_root, image_subdir, '{}.npy'.format(image_id))
    label_path = os.path.join(data_root, 'Label', '{}.npy'.format(image_id))
    img_raw = np.load(img_path)
    label_raw = np.load(label_path) > 0.5

    tsf = RESIZE(image=img_raw.astype('uint8'), mask=label_raw.astype('uint8'))
    img_resized, label_resized = tsf['image'], tsf['mask']

    img_tensor = torch.from_numpy(norm01(img_resized)).float().permute(2, 0, 1)
    img_tensor = NORMALIZE(img_tensor).unsqueeze(0)  # add batch dim

    return img_tensor, label_resized, img_resized


def forward_model(model, model_name, img_tensor):
    if model_name == 'SwinUnet':
        return model(img_tensor)
    return model(img_tensor, d='0')['seg']


def predict(model, model_name, img_tensor, use_tta=False):
    img_tensor = img_tensor.cuda()
    with torch.no_grad():
        if use_tta:
            avg_prob, _ = tta_forward(lambda x: forward_model(model, model_name, x), img_tensor)
            prob = avg_prob
        else:
            logits = forward_model(model, model_name, img_tensor)
            prob = torch.sigmoid(logits)
    return prob.squeeze().cpu().numpy()


def overlay_panel(ax, display_img, pred_mask=None, gt_mask=None, title=''):
    ax.imshow(display_img)
    if gt_mask is not None:
        ax.contour(gt_mask, levels=[0.5], colors='lime', linewidths=1.5)
    if pred_mask is not None:
        ax.contour(pred_mask, levels=[0.5], colors='red', linewidths=1.5)
    ax.set_title(title, fontsize=10)
    ax.axis('off')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single-image 4-stage pipeline comparison')
    parser.add_argument('--image_id', type=str, required=True)
    parser.add_argument('--model', type=str, default='SwinSeg_CNNprompt_adapt')
    parser.add_argument('--baseline_ckpt', type=str, required=True)
    parser.add_argument('--clahe_ckpt', type=str, required=True)
    parser.add_argument('--clahe_ftl_ckpt', type=str, required=True)
    parser.add_argument('--config_yml', type=str, default='Configs/multi_train_local.yml')
    parser.add_argument('--data_root', type=str, default='C:/Users/quanp/Downloads/ISIC 2017/data/isic2017')
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = DotDict(yaml.load(open(args.config_yml), Loader=yaml.FullLoader))
    config.data.img_size = IMG_SIZE

    img_tensor_raw, label_resized, display_raw = load_tensor_and_display(args.data_root, 'Image', args.image_id)
    img_tensor_clahe, label_resized2, display_clahe = load_tensor_and_display(args.data_root, 'Image_clahe', args.image_id)
    assert np.array_equal(label_resized, label_resized2), 'label mismatch between Image/ and Image_clahe/ subdirs'

    stages = []  # (title, display_img_for_panel, pred_mask, dice)

    model = build_model(args.model, config)
    model.load_state_dict(torch.load(args.baseline_ckpt))
    model.eval()
    pred = predict(model, args.model, img_tensor_raw, use_tta=False) > 0.5
    dice = metrics.dc(pred, label_resized)
    stages.append(('1. Baseline\n(no CLAHE, no FTL)', display_raw, pred, dice))
    del model; torch.cuda.empty_cache()

    model = build_model(args.model, config)
    model.load_state_dict(torch.load(args.clahe_ckpt))
    model.eval()
    pred = predict(model, args.model, img_tensor_clahe, use_tta=False) > 0.5
    dice = metrics.dc(pred, label_resized)
    stages.append(('2. + CLAHE', display_clahe, pred, dice))
    del model; torch.cuda.empty_cache()

    model = build_model(args.model, config)
    model.load_state_dict(torch.load(args.clahe_ftl_ckpt))
    model.eval()
    pred = predict(model, args.model, img_tensor_clahe, use_tta=False) > 0.5
    dice = metrics.dc(pred, label_resized)
    stages.append(('3. + CLAHE + compound FTL', display_clahe, pred, dice))

    pred_tta = predict(model, args.model, img_tensor_clahe, use_tta=True) > 0.5
    dice_tta = metrics.dc(pred_tta, label_resized)
    stages.append(('4. + CLAHE + FTL + TTA', display_clahe, pred_tta, dice_tta))
    del model; torch.cuda.empty_cache()

    for title, _, pred, dice in stages:
        safe_name = title.split('\n')[0].split('. ', 1)[1].replace(' ', '_').replace('+', 'plus')
        cv2.imwrite(os.path.join(args.output_dir, '{}_stage{}.png'.format(args.image_id, safe_name)),
                    (pred.astype(np.uint8) * 255))

    fig, axes = plt.subplots(1, 6, figsize=(24, 4.5))
    overlay_panel(axes[0], display_raw, title='Original image\n({})'.format(args.image_id))
    overlay_panel(axes[1], display_raw, gt_mask=label_resized, title='Ground truth\n(green outline)')
    for i, (title, disp, pred, dice) in enumerate(stages):
        overlay_panel(axes[2 + i], disp, pred_mask=pred, gt_mask=label_resized,
                      title='{}\nDice = {:.4f}'.format(title, dice))
    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, '{}_comparison.png'.format(args.image_id))
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print('=== {} Dice progression ==='.format(args.image_id))
    for title, _, pred, dice in stages:
        print('{}: {:.4f}'.format(title.replace(chr(10), ' '), dice))
    print('Saved comparison figure: {}'.format(fig_path))
