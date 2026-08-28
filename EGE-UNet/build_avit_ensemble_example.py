"""
Presentation figure: one test image (0012903) where SwinUnet and EGE-UNet both
predict well and agree closely, but AViT diverges badly -- illustrating why
AViT hurts the 3-way ensemble (per README's "AViT's weaker, higher-variance
predictions... dragging the combined probability map down").

Uses the already-saved TTA-averaged probability maps (results/prob_maps/{model}/
0012903.npy) and the exact same Method A postprocessing (morph_open_close +
keep_largest_component, reused unmodified from AViT/postprocess_pipeline.py) that
produced the reported dice_tta_methodA numbers -- confirmed by recomputation to
match the per-image CSVs to full float precision before building this figure.
"""
import sys
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, '../AViT')
from postprocess_pipeline import morph_open_close, keep_largest_component

IMAGE_ID = '0012903'
DATA_ROOT = '../data/isic2017'
OUT_PATH = 'results/presentation_avit_ensemble_example_0012903.png'


def methodA_pred(prob_map):
    raw = (prob_map > 0.5).astype(np.uint8)
    morphed = morph_open_close(raw, 5)
    return keep_largest_component(morphed)


def main():
    orig_img = np.load('{}/Image_clahe/{}.npy'.format(DATA_ROOT, IMAGE_ID))
    gt = (np.load('{}/Label/{}.npy'.format(DATA_ROOT, IMAGE_ID)) > 0.5).astype(np.uint8)
    gt224 = cv2.resize(gt, (224, 224), interpolation=cv2.INTER_NEAREST)
    orig224 = cv2.resize(orig_img, (224, 224), interpolation=cv2.INTER_LINEAR)

    scores = {}
    for f, name in [('results/swinunet_reference_per_image.csv', 'SwinUnet'),
                     ('results/avit_reference_per_image.csv', 'AViT'),
                     ('results/official_seed42/test_per_image.csv', 'EGE-UNet')]:
        df = pd.read_csv(f, dtype={'image_id': str})
        scores[name] = df[df['image_id'] == IMAGE_ID].iloc[0]['dice_tta_methodA']

    preds = {}
    for model_dir, name in [('swinunet', 'SwinUnet'), ('egeunet', 'EGE-UNet'), ('avit', 'AViT')]:
        prob = np.load('results/prob_maps/{}/{}.npy'.format(model_dir, IMAGE_ID))
        preds[name] = methodA_pred(prob)

    fig, axes = plt.subplots(1, 5, figsize=(24, 5.2))

    axes[0].imshow(orig224)
    axes[0].set_title('Original (CLAHE)', fontsize=12)
    axes[0].axis('off')

    def overlay(ax, mask, color, title):
        ax.imshow(orig224)
        rgba = np.zeros((*mask.shape, 4))
        rgba[mask > 0.5] = color
        ax.imshow(rgba)
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    overlay(axes[1], gt224, [0.15, 1.0, 0.15, 0.5], 'Ground truth')
    overlay(axes[2], preds['SwinUnet'], [0.2, 0.4, 1.0, 0.5],
            'SwinUnet (TTA+MethodA)\nDice = {:.3f}'.format(scores['SwinUnet']))
    overlay(axes[3], preds['EGE-UNet'], [0.2, 0.4, 1.0, 0.5],
            'EGE-UNet (TTA+MethodA)\nDice = {:.3f}'.format(scores['EGE-UNet']))
    overlay(axes[4], preds['AViT'], [1.0, 0.2, 0.2, 0.5],
            'AViT (TTA+MethodA)\nDice = {:.3f}'.format(scores['AViT']))

    fig.suptitle('Why AViT hurts the ensemble -- test image {}: SwinUnet and EGE-UNet agree closely '
                  'and predict well; AViT\'s prediction locks onto the wrong region entirely'.format(IMAGE_ID),
                  fontsize=13, y=1.04)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_PATH, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print('Saved:', OUT_PATH)
    print('Scores used:', scores)
    print('DONE')


if __name__ == '__main__':
    main()
