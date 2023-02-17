import os
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from copy import deepcopy
from argparse import ArgumentParser
import json
import sys 
from tqdm import tqdm

np.random.seed(1000)


def copy_images2final_dir(origin_dir, target_dir, list_file_names):
    '''
    copy final images (after all preprocessing and cleaning) to the final dir for sailon task
    origin_dir: directory containing the images to be copied
    target_dir: directory where the images are to be copied
    list_file_names: list containing the file name of the images
    '''
    assert os.path.exists(origin_dir)
    assert os.path.exists(target_dir)

    # print(f'\n>>> Length of images being copied:', len(list_file_names))

    for fname in tqdm(list_file_names):
        if os.path.exists(target_dir+fname):
            continue
        shutil.copy(origin_dir+fname, target_dir+fname)
        

def split_cal_val(initial_train_fname, initial_val_fname, size_prenovelty_val=3000, size_novety_type_val=300):
    '''
    Split the validation set into validation and calibration sets
    Arguments:
        initial_train_fname: initial training csv and json filename from which the validation prenovelty is will be sampled
        initial_val_fname: initial validation csv and json filename from which the calibration set will be sampled
        size_prenovelty_val: number of images to leave out of the training set for validation (internal testing) prenovelty examples
        size_novety_type_val: maximum number of examples to leave out for validation (internal testing) per novelty/subnovelty type
    '''

    output_dir = '/nfs/hpc/share/sail_on3/final/'

    train_df = pd.read_csv(initial_train_fname + '.csv', index_col=0)
    valid_df = pd.read_csv(initial_val_fname + '.csv', index_col=0)
    
    f1 = open(initial_train_fname + '.json')
    train_json = json.load(f1)
    
    f2 = open(initial_val_fname + '.json')
    valid_json = json.load(f2)

    n_capture_id = int(size_prenovelty_val / 2.6)
    selected_capture_ids = np.random.choice(train_df['capture_id'].unique(), size=n_capture_id, replace=False).tolist()

    new_val_bboxes = {}
    prenov_val_df = deepcopy(train_df.loc[train_df['capture_id'].isin(selected_capture_ids)])
    train_df = train_df.loc[~train_df['capture_id'].isin(selected_capture_ids)]

    for img in prenov_val_df.loc[prenov_val_df['agent1_name']!='blank', 'filename']:
        new_val_bboxes[img] = train_json[img].copy()
        del train_json[img]

    # copy the selected images for prenovelty validation into a separate dir (new validation dir)
    copy_images2final_dir(
        origin_dir=os.path.join(output_dir, 'dataset/train/'), 
        target_dir=os.path.join(output_dir, 'osu_train_cal_val/validation_images/'), 
        list_file_names=prenov_val_df['filename'].to_list()
        )

    cal_df, new_val_df = pd.DataFrame(columns=valid_df.columns), pd.DataFrame(columns=valid_df.columns)
    for novelty in valid_df['novelty_type'].unique():
        environment_ids = valid_df.loc[valid_df['novelty_type']==novelty, 'environment_id'].unique()
        for env_id in environment_ids:
            novelty_val_df = deepcopy(valid_df.loc[(valid_df['novelty_type']==novelty) & (valid_df['environment_id']==env_id)])
            len_all_novelty_val = len(novelty_val_df)
            # len_unique_capture_id = len(novelty_val_df['capture_id'].unique())

            # if the number of images available for the novelty is less that 2 times size_novelty_type_val, the use half as target number
            n_target_val_imgs = min(size_novety_type_val, int(len_all_novelty_val/2))
            if novelty == 6 and env_id in [3, 4]:
                # all images in subnovelty type 3 and 4 of novelty type 6 come from the different capture ids
                n_target_val_capture_ids = n_target_val_imgs
            else:
                n_target_val_capture_ids = int(n_target_val_imgs / 2.2)  # because each capture id is about 3 images

            val_capture_ids = np.random.choice(novelty_val_df['capture_id'].unique(), size=n_target_val_capture_ids, replace=False).tolist()
            novelty_cal_df = novelty_val_df.loc[~novelty_val_df['capture_id'].isin(val_capture_ids)]
            new_novelty_val_df = novelty_val_df.loc[novelty_val_df['capture_id'].isin(val_capture_ids)]

            print('\nNovelty type:', novelty)
            print(f'>>> size calibration set [env: {env_id}]:', len(novelty_cal_df))
            print(f'>>> size valibration set [env: {env_id}]:', len(new_novelty_val_df))

            cal_df = pd.concat([cal_df, novelty_cal_df], ignore_index=True)
            new_val_df = pd.concat([new_val_df, new_novelty_val_df], ignore_index=True)

    for img in new_val_df['filename']:
        new_val_bboxes[img] = valid_json[img].copy()
        del valid_json[img]

    # copy the calidation imges (supset of the initial validation set) into the calibration dir
    copy_images2final_dir(
        origin_dir=os.path.join(output_dir, 'dataset/valid/'), 
        target_dir=os.path.join(output_dir, 'osu_train_cal_val/calibration_images/'), 
        list_file_names=cal_df['filename'].to_list()
        )

    # copy the new validation imges (supset of the initial validation set) into a separate dir (new validation dir)
    copy_images2final_dir(
        origin_dir=os.path.join(output_dir, 'dataset/valid/'), 
        target_dir=os.path.join(output_dir, 'osu_train_cal_val/validation_images/'), 
        list_file_names=new_val_df['filename'].to_list()
        )

    # *** Save the csv files *******
    new_val_df = pd.concat([new_val_df, prenov_val_df], ignore_index=True)
    new_val_df['image_path'] = new_val_df['filename'].apply(lambda x: 'final/osu_train_cal_val/validation_images/' + x)
    cal_df['image_path'] = cal_df['filename'].apply(lambda x: 'final/osu_train_cal_val/calibration_images/' + x)

    # check that validation and calibration sets are disjoint
    assert len(cal_df.loc[cal_df['filename'].isin(new_val_df['filename'])]) == 0
    assert len(new_val_df.loc[new_val_df['filename'].isin(cal_df['filename'])]) == 0

    train_df.to_csv(os.path.join(output_dir, 'osu_train_cal_val/train.csv'), index = None)
    new_val_df.to_csv(os.path.join(output_dir, 'osu_train_cal_val/valid.csv'), index = None)
    cal_df.to_csv(os.path.join(output_dir, 'osu_train_cal_val/calib.csv'), index = None)
    
    # *** Save the json files *****
    with open(os.path.join(output_dir, 'osu_train_cal_val/train.json'), 'w') as fp:
        json.dump(train_json, fp)

    with open(os.path.join(output_dir, 'osu_train_cal_val/valid.json'), 'w') as fp:
        json.dump(new_val_bboxes, fp)

    with open(os.path.join(output_dir, 'osu_train_cal_val/calib.json'), 'w') as fp:
        json.dump(valid_json, fp)



if __name__ == "__main__":

    split_cal_val(
        initial_train_fname='/nfs/hpc/share/sail_on3/final/train', 
        initial_val_fname='/nfs/hpc/share/sail_on3/final/valid', 
        size_prenovelty_val=3000, 
        size_novety_type_val=300
        )
        