"""
Presentation figure: before/after comparison of the shift-augmentation
reflection-padding fix, for images that visibly showed the mirrored-duplicate
artifact under the original 20%-shift/no-blur version. Uses the actual .npy
files already on disk from both versions (shiftR20 = original, shiftR10blur =
corrected) -- no regeneration.
"""
import numpy as np
import matplotlib.pyplot as plt

DATA_ROOT = '../data/isic2017'
IMAGE_IDS = ['0013777', '0012706']


def load(image_id, suffix):
    img = np.load('{}/Image_clahe_aug/{}_{}.npy'.format(DATA_ROOT, image_id, suffix))
    lbl = np.load('{}/Label_aug/{}_{}.npy'.format(DATA_ROOT, image_id, suffix)) > 0.5
    return img, lbl


def overlay_mask(ax, img, lbl, title):
    ax.imshow(img)
    rgba = np.zeros((*lbl.shape, 4))
    rgba[lbl > 0.5] = [0.15, 1.0, 0.15, 0.4]
    ax.imshow(rgba)
    ax.set_title(title, fontsize=12)
    ax.axis('off')


def main():
    for image_id in IMAGE_IDS:
        orig_img = np.load('{}/Image_clahe/{}.npy'.format(DATA_ROOT, image_id))
        orig_lbl = np.load('{}/Label/{}.npy'.format(DATA_ROOT, image_id)) > 0.5
        before_img, before_lbl = load(image_id, 'shiftR20')
        after_img, after_lbl = load(image_id, 'shiftR10blur')

        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
        overlay_mask(axes[0], orig_img, orig_lbl, 'Original (unshifted)')
        overlay_mask(axes[1], before_img, before_lbl,
                     'BEFORE FIX\n20% shift, plain reflect-101\n(visible mirrored ruler/ink duplicate)')
        overlay_mask(axes[2], after_img, after_lbl,
                     'AFTER FIX\n10% shift, blurred reflect-101 source\n(no visible duplicate)')

        fig.suptitle('Shift-augmentation reflection-padding fix -- image {}'.format(image_id), fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        out_path = '../per_image_analysis_v2/bad_image_augmentation/presentation_shift_before_after_{}.png'.format(image_id)
        fig.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print('Saved:', out_path)

    print('DONE')


if __name__ == '__main__':
    main()
