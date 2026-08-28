"""
Presentation figure: one hard image (from AViT's 189-image list) and all 5 of
its augmented versions, side by side -- original + hflip + vflip + rot90 +
rot270 + shift(10%, blurred reflect-101), 6 columns. Row 1 = images, row 2 =
each panel's own ground truth mask (not overlaid -- shown standalone so the
mask's own transform is directly visible, independent of the image).
"""
import numpy as np
import matplotlib.pyplot as plt

DATA_ROOT = '../data/isic2017'
IMAGE_ID = '0013196'
OUT_PATH = '../per_image_analysis_v2/bad_image_augmentation/presentation_augmentation_demo_grid_{}.png'.format(IMAGE_ID)

PANELS = [
    (None, 'Original'),
    ('hflip', 'Horizontal flip'),
    ('vflip', 'Vertical flip'),
    ('rot90', '90° rotation'),
    ('rot270', '270° rotation'),
    ('shiftR10blur', 'Shift 10% right\n(blurred reflect-101)'),
]


def load(suffix):
    if suffix is None:
        img = np.load('{}/Image_clahe/{}.npy'.format(DATA_ROOT, IMAGE_ID))
        lbl = np.load('{}/Label/{}.npy'.format(DATA_ROOT, IMAGE_ID)) > 0.5
    else:
        img = np.load('{}/Image_clahe_aug/{}_{}.npy'.format(DATA_ROOT, IMAGE_ID, suffix))
        lbl = np.load('{}/Label_aug/{}_{}.npy'.format(DATA_ROOT, IMAGE_ID, suffix)) > 0.5
    return img, lbl


def main():
    fig, axes = plt.subplots(2, 6, figsize=(24, 8.5))

    for col, (suffix, title) in enumerate(PANELS):
        img, lbl = load(suffix)

        axes[0, col].imshow(img)
        axes[0, col].set_title(title, fontsize=13)
        axes[0, col].axis('off')

        axes[1, col].imshow(lbl, cmap='gray')
        axes[1, col].axis('off')

    axes[0, 0].set_ylabel('Image', fontsize=13)
    axes[1, 0].set_ylabel('Ground truth\nmask', fontsize=13)
    # matplotlib hides ylabel when axis('off') is called -- re-enable just the label
    for row, label in [(0, 'Image'), (1, 'Ground truth mask')]:
        axes[row, 0].axis('on')
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        for spine in axes[row, 0].spines.values():
            spine.set_visible(False)
        axes[row, 0].set_ylabel(label, fontsize=13)

    fig.suptitle('Augmentation demo -- image {} (AViT hard-training-image list, source dice=0.000)'.format(IMAGE_ID),
                  fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_PATH, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print('Saved:', OUT_PATH)
    print('DONE')


if __name__ == '__main__':
    main()
