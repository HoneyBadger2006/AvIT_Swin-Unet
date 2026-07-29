'''
Input: downloaded ISIC 2017 dataset (Training + Test_v2, Part1 segmentation ground truth)
Process: resize, change jpg to npy, store images and labels to Image/, Label/, build meta_isic2017.csv
Modeled on process_isic2018() in process_resize.py
'''

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


isic2017_raw_folder = 'C:/Users/quanp/Downloads/ISIC 2017/isic2017_raw'
isic2017_proceeded_folder = 'C:/Users/quanp/Downloads/ISIC 2017/data/isic2017'


def process_isic2017(
    isic2017_raw_folder=isic2017_raw_folder,
    dim=(512, 512),
    isic2017_proceeded_folder=isic2017_proceeded_folder):

    splits = [
        (isic2017_raw_folder + '/ISIC-2017_Training_Data',
         isic2017_raw_folder + '/ISIC-2017_Training_Part1_GroundTruth'),
        (isic2017_raw_folder + '/ISIC-2017_Test_v2_Data',
         isic2017_raw_folder + '/ISIC-2017_Test_v2_Part1_GroundTruth'),
    ]

    image_save_dir = isic2017_proceeded_folder + '/Image'
    label_save_dir = isic2017_proceeded_folder + '/Label'
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(label_save_dir, exist_ok=True)

    id_list = []
    for image_dir, mask_dir in splits:
        image_path_list = sorted(f for f in os.listdir(image_dir) if f.endswith('.jpg'))
        mask_path_list = sorted(f for f in os.listdir(mask_dir) if f.endswith('_segmentation.png'))
        assert len(image_path_list) == len(mask_path_list), \
            '{} images vs {} masks in {}'.format(len(image_path_list), len(mask_path_list), image_dir)
        print('{}: {} images, {} masks'.format(image_dir, len(image_path_list), len(mask_path_list)))

        for image_name, mask_name in tqdm(list(zip(image_path_list, mask_path_list))):
            _id = image_name[:-4].split('_')[1]
            assert mask_name.split('_')[1] == _id, '{} != {}'.format(mask_name, _id)

            image = plt.imread(os.path.join(image_dir, image_name))
            mask = plt.imread(os.path.join(mask_dir, mask_name))

            image_new = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
            mask_new = cv2.resize(mask, dim, interpolation=cv2.INTER_NEAREST)

            np.save(os.path.join(image_save_dir, _id + '.npy'), image_new)
            np.save(os.path.join(label_save_dir, _id + '.npy'), mask_new)
            id_list.append(_id)

    df = pd.DataFrame({'ID': id_list})
    df['dataset'] = 'isic2017'
    # ISIC-2017 classification ground truth (Part 3) was not downloaded for this project;
    # diagnosis/diagnosis_id are unused by the segmentation loss, only carried as metadata.
    df['diagnosis'] = 'unknown'
    df['diagnosis_id'] = 0
    df.to_csv(os.path.join(isic2017_proceeded_folder, 'meta_isic2017.csv'), index=False)
    print('Finished: {} images processed, meta_isic2017.csv written with {} rows.'.format(len(id_list), len(df)))


if __name__ == '__main__':
    process_isic2017()
