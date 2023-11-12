"""
Takes a tests directory and a results directory.
Writes log files for each test and a summary scores file to an output logs directory.
"""
#########################################################################
# Copyright 2021-2022 by Raytheon BBN Technologies.  All Rights Reserved
#########################################################################


import os
from pathlib import Path
from argparse import ArgumentParser
import json
import pickle
import pandas as pd
from class_file_reader import ClassFileReader
# from stats import Stats, Instance
import shutil
import numpy as np
import sklearn.metrics as metrics
import ast

from write_results import write_results_to_log, print_confusion_matrices


epsilon = 1e-6

def percent_string(num, denom=None):
    if denom is None:
        return f'{100 * num:6.2f}%'
    return f'{100 * num / denom:6.2f}%'


def species_count_error(grd_trth_vec, pred_vec, metric='MSE'):
    '''
    Computes the squared error, absolute error, mean squared error, 
    mean absolute error or root mean squared error between the ground truth and the predicion
    '''
    assert metric in ('SE', 'AE', 'RMSE', 'MSE', 'MAE')
    assert len(grd_trth_vec) == len(pred_vec)

    grd_trth_vec = np.array(grd_trth_vec).ravel()
    pred_vec = np.array(pred_vec).ravel()
    diff = grd_trth_vec - pred_vec
    sqrt_diff = np.power(diff, 2)

    if metric == 'SE':
        return sum(sqrt_diff)
    elif metric == 'AE':
        return sum(np.abs(diff))
    elif metric == 'MSE':
        return np.mean(sqrt_diff)
    elif metric == 'MAE':
        return np.mean(np.abs(diff))
    return np.sqrt(np.mean(sqrt_diff))


def score_test(
    test_id, metadata, test_df, detect_lines, class_lines, class_file_reader, 
    log, all_performances, detection_threshold, spe_presence_threshold=0.5, 
    act_presence_threshold=0.5
):
    # The metadata file is not currently used.
    # import ipdb; ipdb.set_trace()
    total_novel, total = 0, 0
    test_tuples = test_df.itertuples()
    len_test = len(test_df)
    red_button = False
    red_button_pos = -1
    sys_declare_pos = -1

    nbr_known_species = 11

    spe_counts_cols = [x for x in class_lines[0].split(',') if 'species_' in x and '_count' in x]

    spe_presence_cols = [x for x in class_lines[0].split(',') if 'species_' in x and '_presence' in x]
    total_nbr_species = len(spe_presence_cols)

    act_presence_cols = [x for x in class_lines[0].split(',') if 'activity_' in x]
    total_nbr_activities = len(act_presence_cols)

    class_lines = pd.DataFrame([class_line.split(',') for class_line in class_lines[1:]], columns=class_lines[0].split(','))
    all_pred_spe_counts = class_lines[spe_counts_cols].astype(float).copy(deep=True)
    all_pred_spe_presence = class_lines[spe_presence_cols].astype(float).copy(deep=True)
    all_pred_act_presence = class_lines[act_presence_cols].astype(float).copy(deep=True)

    all_grd_truth_spe_counts = np.zeros((len_test, total_nbr_species))
    all_grd_truth_spe_presence = np.zeros((len_test, total_nbr_species))
    all_grd_truth_act_presence = np.zeros((len_test, total_nbr_activities))

    pre_red_per_spe_precision, pre_red_per_spe_recall, pre_red_per_spe_f1, pre_red_per_spe_auc = {}, {}, {}, {}
    post_red_per_spe_precision, post_red_per_spe_recall, post_red_per_spe_f1, post_red_per_spe_auc = {}, {}, {}, {}

    pre_red_per_act_precision, pre_red_per_act_recall, pre_red_per_act_f1, pre_red_per_act_auc = {}, {}, {}, {}
    post_red_per_act_precision, post_red_per_act_recall, post_red_per_act_f1, post_red_per_act_auc = {}, {}, {}, {}

    total_pre_red_btn, total_post_red_btn = 0, 0

    species_id2name_mapping = {}
    activity_id2name_mapping = {}

    pre_red_count_abs_err, post_red_count_abs_err, pre_red_count_rel_err, post_red_count_rel_err = {}, {}, {}, {}
    for pos, (test_tuple, detect_line) in enumerate(zip(test_tuples, detect_lines[1:])):
        if pd.isnull(test_tuple.agent1_id):
            print(f'** Empty image:', test_tuple.image_path)
            continue

        if test_tuple.novel:
            red_button = True
            if red_button_pos == -1:
                red_button_pos = pos
            total_novel += 1
        
        total += 1
        if red_button:
            total_post_red_btn += 1
        else:
            total_pre_red_btn += 1
                
        ans_nov = float(detect_line.split(',')[1]) > detection_threshold
        if sys_declare_pos == -1 and ans_nov:
            sys_declare_pos = pos      

        # ground truth vector of species counts
        ground_truth_species_counts = np.zeros((1, total_nbr_species)).ravel()
        ground_truth_species_presence = np.zeros((1, total_nbr_species)).ravel()

        assert not pd.isnull(test_tuple.agent1_id)

        ground_truth_species_counts[int(test_tuple.agent1_id)] = int(test_tuple.agent1_count)
        ground_truth_species_presence[int(test_tuple.agent1_id)] = 1

        if int(test_tuple.agent1_id) not in species_id2name_mapping:
            species_id2name_mapping[int(test_tuple.agent1_id)] = test_tuple.agent1_name

        # if 0 not in species_id2name_mapping:
        #     species_id2name_mapping[0] = 'blank'

        if not pd.isnull(test_tuple.agent2_id):
            ground_truth_species_counts[int(test_tuple.agent2_id)] = int(test_tuple.agent2_count)
            ground_truth_species_presence[int(test_tuple.agent2_id)] = 1

            if int(test_tuple.agent2_id) not in species_id2name_mapping:
                species_id2name_mapping[int(test_tuple.agent2_id)] = test_tuple.agent2_name

        if not pd.isnull(test_tuple.agent3_id):
            ground_truth_species_counts[int(test_tuple.agent3_id)] = int(test_tuple.agent3_count)
            ground_truth_species_presence[int(test_tuple.agent3_id)] = 1

            if int(test_tuple.agent3_id) not in species_id2name_mapping:
                species_id2name_mapping[int(test_tuple.agent3_id)] = test_tuple.agent3_name

        # update array of all ground truth species presence
        all_grd_truth_spe_counts[pos,:] = ground_truth_species_counts
        
        # update array of all ground truth species presence
        all_grd_truth_spe_presence[pos,:] = ground_truth_species_presence

        # ground truth boolean vector of activity presence
        ground_truth_activity_presence = np.zeros((1, total_nbr_activities)).ravel()
        # list_activities = ast.literal_eval(test_tuple.activities)
        list_activities_id = ast.literal_eval(test_tuple.activities_id)
        list_activities = ast.literal_eval(test_tuple.activities)

        for idx, act_id in enumerate(list_activities_id):
            ground_truth_activity_presence[act_id] = 1

            if act_id not in activity_id2name_mapping:
                activity_id2name_mapping[act_id] = list_activities[idx]

        all_grd_truth_act_presence[pos, :] = ground_truth_activity_presence


    # ** new column names
    new_spe_count_col_names = {}
    for spe_col in spe_counts_cols:
        spe_id = int(spe_col.split('_')[1])
        if spe_id in species_id2name_mapping:
            new_spe_count_col_names[spe_col] = species_id2name_mapping[spe_id]
        else:
            new_spe_count_col_names[spe_col] = 'blank' if spe_id == 0 else 'unknown_' + str(spe_id)
    
    new_spe_presence_col_names = {}
    for spe_col in spe_presence_cols:
        spe_id = int(spe_col.split('_')[1])
        if spe_id in species_id2name_mapping:
            new_spe_presence_col_names[spe_col] = species_id2name_mapping[spe_id]
        else:
            new_spe_presence_col_names[spe_col] = 'blank' if spe_id == 0 else 'unknown_' + str(spe_id)

    new_act_col_names = {}
    for act_col in act_presence_cols:
        act_id = int(act_col.split('_')[1])
        if act_id in activity_id2name_mapping:
            new_act_col_names[act_col] = activity_id2name_mapping[act_id]
        else:
            new_act_col_names[act_col] = 'unknown_' + str(act_id)

    # ** renaming columns 
    all_pred_spe_counts.rename(columns=new_spe_count_col_names, inplace=True)
    all_pred_spe_presence.rename(columns=new_spe_presence_col_names, inplace=True)
    all_pred_act_presence.rename(columns=new_act_col_names, inplace=True)

    all_grd_truth_spe_counts = pd.DataFrame(all_grd_truth_spe_counts, columns=all_pred_spe_counts.columns)
    all_grd_truth_spe_presence = pd.DataFrame(all_grd_truth_spe_presence, columns=all_pred_spe_presence.columns)
    all_grd_truth_act_presence = pd.DataFrame(all_grd_truth_act_presence, columns=all_pred_act_presence.columns)

    # ** Drop empty species from ground truth and predicted data
    for df in [
        all_pred_spe_counts, all_pred_spe_presence, all_pred_act_presence, 
        all_grd_truth_spe_counts, all_grd_truth_spe_presence, all_grd_truth_act_presence
        ]:
        if 'blank' in df.columns:
            df.drop(columns='blank', inplace=True)


    # *****************************  ACTIVITY PRESENCE PERFORMANCE  ********************************

    for act in all_grd_truth_act_presence.columns:
        if act not in activity_id2name_mapping.values():
            pre_red_per_act_auc[act], post_red_per_act_auc[act] = -1, -1
            pre_red_per_act_precision[act], post_red_per_act_precision[act] = -1, -1
            pre_red_per_act_recall[act], post_red_per_act_recall[act] = -1, -1
            pre_red_per_act_f1[act], post_red_per_act_f1[act] = -1, -1
            continue

        # ---------->>  Activity presence AUC  <<----------
        try:
            pre_red_per_act_auc[act] = metrics.roc_auc_score(
                    all_grd_truth_act_presence[act].iloc[:total_pre_red_btn], 
                    all_pred_act_presence[act].iloc[:total_pre_red_btn]
                )
            post_red_per_act_auc[act] = metrics.roc_auc_score(
                    all_grd_truth_act_presence[act].iloc[total_pre_red_btn:], 
                    all_pred_act_presence[act].iloc[total_pre_red_btn:]
                )
        except Exception as ex:
            print('\n\n**** The following exception happened:', ex, '\n\n')
            pre_red_per_act_auc[act] = -1
            post_red_per_act_auc[act] = -1
        
        # ---------->>  Activity presence Precision, Recall, F1  <<----------
        # if act_presence_threshold:
        all_pred_act_presence_yn = all_pred_act_presence.copy(deep=True)
        all_pred_act_presence_yn = (all_pred_act_presence_yn >= act_presence_threshold).astype(int)
        try:
            pre_red_precision, pre_red_rec, pre_red_f1, _ = metrics.precision_recall_fscore_support(
                all_grd_truth_act_presence[act].iloc[:total_pre_red_btn], 
                all_pred_act_presence_yn[act].iloc[:total_pre_red_btn],
                average='binary'
            )
            post_red_precision, post_red_rec, post_red_f1, _ = metrics.precision_recall_fscore_support(
                all_grd_truth_act_presence[act].iloc[total_pre_red_btn:], 
                all_pred_act_presence_yn[act].iloc[total_pre_red_btn:],
                average='binary'
            )

            # Precision
            pre_red_per_act_precision[act] = pre_red_precision
            post_red_per_act_precision[act] = post_red_precision

            # Recall
            pre_red_per_act_recall[act] = pre_red_rec
            post_red_per_act_recall[act] = post_red_rec

            # F1 score
            pre_red_per_act_f1[act] = pre_red_f1
            post_red_per_act_f1[act] = post_red_f1
        except:
            pre_red_per_act_precision[act], post_red_per_act_precision[act] = -1, -1
            pre_red_per_act_recall[act], post_red_per_act_recall[act] = -1, -1
            pre_red_per_act_f1[act], post_red_per_act_f1[act] = -1, -1
            continue


    # *****************************  SPECIES PRESENCE PERFORMANCE  ********************************

    for spe in all_grd_truth_spe_presence.columns:

        if spe not in species_id2name_mapping.values():
            pre_red_per_spe_auc[spe], post_red_per_spe_auc[spe] = -1, -1
            pre_red_per_spe_precision[spe], post_red_per_spe_precision[spe] = -1, -1
            pre_red_per_spe_recall[spe], post_red_per_spe_recall[spe] = -1, -1
            pre_red_per_spe_f1[spe], post_red_per_spe_f1[spe] = -1, -1
            continue

        # ---------->>  Species presence AUC  <<----------
        try:
            pre_red_per_spe_auc[spe] = metrics.roc_auc_score(
                    all_grd_truth_spe_presence[spe].iloc[:total_pre_red_btn], 
                    all_pred_spe_presence[spe].iloc[:total_pre_red_btn]
                )
            post_red_per_spe_auc[spe] = metrics.roc_auc_score(
                    all_grd_truth_spe_presence[spe].iloc[total_pre_red_btn:], 
                    all_pred_spe_presence[spe].iloc[total_pre_red_btn:]
                )
        except:
            pre_red_per_spe_auc[spe] = -1
            post_red_per_spe_auc[spe] = -1
        
        # ---------->>  Activity presence Precision, Recall, F1  <<----------
        # if spe_presence_threshold:
        all_pred_spe_presence_yn = all_pred_spe_presence.copy(deep=True)
        all_pred_spe_presence_yn = (all_pred_spe_presence_yn >= spe_presence_threshold).astype(int)

        try:
            pre_red_precision, pre_red_rec, pre_red_f1, _ = metrics.precision_recall_fscore_support(
                all_grd_truth_spe_presence[spe].iloc[:total_pre_red_btn], 
                all_pred_spe_presence_yn[spe].iloc[:total_pre_red_btn],
                average='binary'
            )
            post_red_precision, post_red_rec, post_red_f1, _ = metrics.precision_recall_fscore_support(
                all_grd_truth_spe_presence[spe].iloc[total_pre_red_btn:], 
                all_pred_spe_presence_yn[spe].iloc[total_pre_red_btn:],
                average='binary'
            )
        
            # Precision
            pre_red_per_spe_precision[spe] = pre_red_precision
            post_red_per_spe_precision[spe] = post_red_precision

            # Recall
            pre_red_per_spe_recall[spe] = pre_red_rec
            post_red_per_spe_recall[spe] = post_red_rec

            # F1 score
            pre_red_per_spe_f1[spe] = pre_red_f1
            post_red_per_spe_f1[spe] = post_red_f1
        except:
            pre_red_per_spe_auc[spe], post_red_per_spe_auc[spe] = -1, -1
            pre_red_per_spe_precision[spe], post_red_per_spe_precision[spe] = -1, -1
            pre_red_per_spe_recall[spe], post_red_per_spe_recall[spe] = -1, -1
            pre_red_per_spe_f1[spe], post_red_per_spe_f1[spe] = -1, -1


    # *****************************  SPECIES COUNT PERFORMANCE  ********************************

    num_pre_red_img_w_spe = all_grd_truth_spe_counts.iloc[:total_pre_red_btn].astype(bool).sum(axis=0)
    num_post_red_img_w_spe = all_grd_truth_spe_counts.iloc[total_pre_red_btn:].astype(bool).sum(axis=0)
    for spe in all_grd_truth_spe_counts.columns:
        if spe not in species_id2name_mapping.values():
            pre_red_count_abs_err[spe], post_red_count_abs_err[spe] = -1, -1
            pre_red_count_rel_err[spe], post_red_count_rel_err[spe] = -1, -1
            continue
        
        # ---------->>  Absolute error  <<----------
        if num_pre_red_img_w_spe[spe] > 0:
            pre_red_count_abs_err[spe] = species_count_error(
                all_grd_truth_spe_counts[spe].iloc[:total_pre_red_btn], 
                all_pred_spe_counts[spe].iloc[:total_pre_red_btn], 
                metric='AE'
            ) / num_pre_red_img_w_spe[spe]
        else:
            pre_red_count_abs_err[spe] = -1

        if num_post_red_img_w_spe[spe] > 0:
            post_red_count_abs_err[spe] = species_count_error(
                all_grd_truth_spe_counts[spe].iloc[total_pre_red_btn:], 
                all_pred_spe_counts[spe].iloc[total_pre_red_btn:], 
                metric='AE'
            ) / num_post_red_img_w_spe[spe]
        else:
            post_red_count_abs_err[spe] = -1
        
        # ---------->>  Relative error  <<----------
        if all_grd_truth_spe_counts[spe].iloc[:total_pre_red_btn].sum() > 0:
            pre_red_count_rel_err[spe] = species_count_error(
                all_grd_truth_spe_counts[spe].iloc[:total_pre_red_btn], 
                all_pred_spe_counts[spe].iloc[:total_pre_red_btn], 
                metric='AE'
            ) / all_grd_truth_spe_counts[spe].iloc[:total_pre_red_btn].sum()
        else:
            pre_red_count_rel_err[spe] = -1
        
        if all_grd_truth_spe_counts[spe].iloc[total_pre_red_btn:].sum() > 0:
            post_red_count_rel_err[spe] = species_count_error(
                all_grd_truth_spe_counts[spe].iloc[total_pre_red_btn:], 
                all_pred_spe_counts[spe].iloc[total_pre_red_btn:], 
                metric='AE'
            ) / all_grd_truth_spe_counts[spe].iloc[total_pre_red_btn:].sum()
        else:
            post_red_count_rel_err[spe] = -1

    # log.write(f'{" "*86} {total_top_1_score}  {total_top_3_score}\n')
    if sys_declare_pos == -1:
        detect = 'Miss'
        delay = ' NA'
    elif sys_declare_pos < red_button_pos:
        detect = ' FA '
        delay = ' NA'
    else:
        detect = 'HIT '
        delay = f'{sys_declare_pos - red_button_pos:3}'


    # ************ Write all performance for current test_id *****************
    all_performances['Red_button'][test_id] = red_button_pos
    all_performances['Red_button_declared'][test_id] = sys_declare_pos
    all_performances['Red_button_result'][test_id] = detect
    all_performances['Delay'][test_id] = delay
    all_performances['Total'][test_id] = total
    all_performances['Novel'][test_id] = total_novel

    all_performances['Species_id2name'][test_id] = species_id2name_mapping
    all_performances['Activity_id2name'][test_id] = activity_id2name_mapping

    # species counts
    counts = {
        'pre_red_btn':{
            'abs_err': pre_red_count_abs_err,
            'rel_err': pre_red_count_rel_err
        },
        'post_red_btn':{
            'abs_err': post_red_count_abs_err,
            'rel_err': post_red_count_rel_err
        }
    }
    all_performances['Species_counts'][test_id] = counts

    # Aggregate species counts
    pre_red_count_abs_err_arr = np.fromiter(pre_red_count_abs_err.values(), dtype=float)
    post_red_count_abs_err_arr = np.fromiter(post_red_count_abs_err.values(), dtype=float)
    pre_red_count_rel_err_arr = np.fromiter(pre_red_count_rel_err.values(), dtype=float)
    post_red_count_rel_err_arr = np.fromiter(post_red_count_rel_err.values(), dtype=float)
    agg_species_counts = {
        'pre_red_btn':{
            'avg_abs_err': round(np.mean(pre_red_count_abs_err_arr[pre_red_count_abs_err_arr >= 0]), 3),
            'avg_rel_err': round(np.mean(pre_red_count_rel_err_arr[pre_red_count_rel_err_arr >= 0]), 3)
        },
        'post_red_btn':{
            'avg_abs_err': round(np.mean(post_red_count_abs_err_arr[post_red_count_abs_err_arr >= 0]), 3),
            'avg_rel_err': round(np.mean(post_red_count_rel_err_arr[post_red_count_rel_err_arr >= 0]), 3)
        }
    }
    all_performances['Aggregate_species_counts'][test_id] = agg_species_counts


    # Per species presence
    species_presence = {
        'pre_red_btn':{
            'auc': pre_red_per_spe_auc,
            'precision': pre_red_per_spe_precision,
            'recall': pre_red_per_spe_recall,
            'f1_score': pre_red_per_spe_f1
        },
        'post_red_btn':{
            'auc': post_red_per_spe_auc,
            'precision': post_red_per_spe_precision,
            'recall': post_red_per_spe_recall,
            'f1_score': post_red_per_spe_f1
        }
    }
    all_performances['Per_species_presence'][test_id] = species_presence

    # Species presence confusion matrix
    '''
    species_cm = {
        'pre_red_btn': {
            'cm': pre_red_species_cm,
            'species_ids': np.sort(pre_red_unique_spe)
        },
        'post_red_btn': {
            'cm': post_red_species_cm,
            'species_ids': np.sort(post_red_unique_spe)
        }
    }
    all_performances['Species_confusion_matrices'][test_id] = species_cm
    '''

    # Aggregate species presence
    pre_red_per_spe_auc_arr = np.fromiter(pre_red_per_spe_auc.values(), dtype=float)
    pre_red_per_spe_precision_arr = np.fromiter(pre_red_per_spe_precision.values(), dtype=float)
    pre_red_per_spe_recall_arr = np.fromiter(pre_red_per_spe_recall.values(), dtype=float)
    pre_red_per_spe_f1_arr = np.fromiter(pre_red_per_spe_f1.values(), dtype=float)

    post_red_per_spe_auc_arr = np.fromiter(post_red_per_spe_auc.values(), dtype=float)
    post_red_per_spe_precision_arr = np.fromiter(post_red_per_spe_precision.values(), dtype=float)
    post_red_per_spe_recall_arr = np.fromiter(post_red_per_spe_recall.values(), dtype=float)
    post_red_per_spe_f1_arr = np.fromiter(post_red_per_spe_f1.values(), dtype=float)

    agg_species_presence = {
        'pre_red_btn':{
        'avg_auc': round(np.mean(pre_red_per_spe_auc_arr[pre_red_per_spe_auc_arr >= 0]), 2),
        'avg_precision': round(np.mean(pre_red_per_spe_precision_arr[pre_red_per_spe_precision_arr >= 0]), 2),
        'avg_recall': round(np.mean(pre_red_per_spe_recall_arr[pre_red_per_spe_recall_arr >= 0]), 2),
        'avg_f1_score': round(np.mean(pre_red_per_spe_f1_arr[pre_red_per_spe_f1_arr >= 0]), 2)
        },
        'post_red_btn':{
            'avg_auc': round(np.mean(post_red_per_spe_auc_arr[post_red_per_spe_auc_arr >= 0]), 2),
            'avg_precision': round(np.mean(post_red_per_spe_precision_arr[post_red_per_spe_precision_arr >= 0]), 2),
            'avg_recall': round(np.mean(post_red_per_spe_recall_arr[post_red_per_spe_recall_arr >= 0]), 2),
            'avg_f1_score': round(np.mean(post_red_per_spe_f1_arr[post_red_per_spe_f1_arr >= 0]), 2)
        }
    }
    all_performances['Aggregate_species_presence'][test_id] = agg_species_presence

    # Per activity presence
    activity_presence = {
        'pre_red_btn':{
            'auc': pre_red_per_act_auc,
            'precision': pre_red_per_act_precision,
            'recall': pre_red_per_act_recall,
            'f1_score': pre_red_per_act_f1
        },
        'post_red_btn':{
            'auc': post_red_per_act_auc,
            'precision': post_red_per_act_precision,
            'recall': post_red_per_act_recall,
            'f1_score': post_red_per_act_f1
        }
    }
    all_performances['Per_activity_presence'][test_id] = activity_presence

    # Aggregate activity presence
    pre_red_per_act_auc_arr = np.fromiter(pre_red_per_act_auc.values(), dtype=float)
    pre_red_per_act_precision_arr = np.fromiter(pre_red_per_act_precision.values(), dtype=float)
    pre_red_per_act_recall_arr = np.fromiter(pre_red_per_act_recall.values(), dtype=float)
    pre_red_per_act_f1_arr = np.fromiter(pre_red_per_act_f1.values(), dtype=float)

    post_red_per_act_auc_arr = np.fromiter(post_red_per_act_auc.values(), dtype=float)
    post_red_per_act_precision_arr = np.fromiter(post_red_per_act_precision.values(), dtype=float)
    post_red_per_act_recall_arr = np.fromiter(post_red_per_act_recall.values(), dtype=float)
    post_red_per_act_f1_arr = np.fromiter(post_red_per_act_f1.values(), dtype=float)
    agg_activity_presence = {
        'pre_red_btn':{
        'avg_auc': round(np.mean(pre_red_per_act_auc_arr[pre_red_per_act_auc_arr >= 0]), 2),
        'avg_precision': round(np.mean(pre_red_per_act_precision_arr[pre_red_per_act_precision_arr >= 0]), 2),
        'avg_recall': round(np.mean(pre_red_per_act_recall_arr[pre_red_per_act_recall_arr >= 0]), 2),
        'avg_f1_score': round(np.mean(pre_red_per_act_f1_arr[pre_red_per_act_f1_arr >= 0]), 2)
        },
        'post_red_btn':{
            'avg_auc': round(np.mean(post_red_per_act_auc_arr[post_red_per_act_auc_arr >= 0]), 2),
            'avg_precision': round(np.mean(post_red_per_act_precision_arr[post_red_per_act_precision_arr >= 0]), 2),
            'avg_recall': round(np.mean(post_red_per_act_recall_arr[post_red_per_act_recall_arr >= 0]), 2),
            'avg_f1_score': round(np.mean(post_red_per_act_f1_arr[post_red_per_act_f1_arr >= 0]), 2)
        }
    }
    all_performances['Aggregate_activity_presence'][test_id] = agg_activity_presence

    # Activity presence confusion matrix
    '''
    activity_cm = {
        'pre_red_btn': {
            'cm': pre_red_activity_cm,
            'activity_ids': np.sort(pre_red_unique_act)
        },
        'post_red_btn': {
            'cm': post_red_activity_cm,
            'activity_ids': np.sort(post_red_unique_act)
        }
    }
    all_performances['Activity_confusion_matrices'][test_id] = activity_cm
    '''



def score_test_from_boxes(
    test_id, metadata, test_df, detect_lines, class_lines, boxes_pred_dict, class_file_reader, 
    log, all_performances, detection_threshold, presence_threshold=0.5, count_metric='AE'
):
    # import ipdb; ipdb.set_trace()

    # The metadata file is not currently used.
    # import ipdb; ipdb.set_trace()
    total_pre_red, total_post_red, total_novel, total = 0, 0, 0, 0
    test_tuples = test_df.itertuples()
    len_test = len(test_df)
    red_button = False
    red_button_pos = -1
    sys_declare_pos = -1

    nbr_known_species = 11

    bboxes_prediction_dict = boxes_pred_dict['per_box_predictions']

    species_cols = [x for x in class_lines[0].split(',') if 'species_' in x and '_presence' in x]
    total_nbr_species = len(species_cols)

    activity_cols = [x for x in class_lines[0].split(',') if 'activity_' in x]
    total_nbr_activities = len(activity_cols)

    # *********  brier score, accuracy and auc  *********
    # species variables initialization
    pre_red_per_spe_running_bs = np.zeros((1, total_nbr_species)).ravel()
    pre_red_per_spe_presence_acc = np.zeros((1, total_nbr_species)).ravel()
    pre_red_per_spe_auc = np.zeros((1, total_nbr_species)).ravel()

    post_red_per_spe_running_bs = np.zeros((1, total_nbr_species)).ravel()
    post_red_per_spe_presence_acc = np.zeros((1, total_nbr_species)).ravel()
    post_red_per_spe_auc = np.zeros((1, total_nbr_species)).ravel()

    # activities variables initialization
    pre_red_per_act_running_bs = np.zeros((1, total_nbr_activities)).ravel()
    pre_red_per_act_acc = np.zeros((1, total_nbr_activities)).ravel()
    pre_red_per_act_auc = np.zeros((1, total_nbr_activities)).ravel()

    post_red_per_act_running_bs = np.zeros((1, total_nbr_activities)).ravel()
    post_red_per_act_acc = np.zeros((1, total_nbr_activities)).ravel()
    post_red_per_act_auc = np.zeros((1, total_nbr_activities)).ravel()

    # aninal counts variables initialization (absolute error and relative error)
    pre_red_per_spe_count_abs_err = np.zeros((1, total_nbr_species)).ravel()
    pre_red_per_spe_count_rel_err = np.zeros((1, total_nbr_species)).ravel()
    
    post_red_per_spe_count_abs_err = np.zeros((1, total_nbr_species)).ravel()
    post_red_per_spe_count_rel_err = np.zeros((1, total_nbr_species)).ravel()

    pre_red_btn_per_spe_animal_count = np.zeros((1, total_nbr_species)).ravel()
    post_red_btn_per_spe_animal_count = np.zeros((1, total_nbr_species)).ravel()

    total_pre_red_btn, total_post_red_btn = 0, 0

    all_grd_truth_spe_in_box = []  # contains the ground truth species in all image with single species
    all_pred_spe_in_box = []  # contains the predicted species in all image with single species

    all_grd_truth_act_in_box = []  # contains the ground truth species in all image with single species
    all_pred_act_in_box = []  # contains the predicted species in all image with single species

    pre_red_act_present_yn = np.zeros((1, total_nbr_activities)).ravel()
    post_red_act_present_yn = np.zeros((1, total_nbr_activities)).ravel()

    species_id2name_mapping = {}
    activity_id2name_mapping = {}

    nbr_empty_imgs, nbr_imgs_w_single_spe, nbr_imgs_w_single_act = 0, 0, 0


    for pos, (test_tuple, detect_line, class_line) in enumerate(zip(test_tuples, detect_lines[1:], class_lines[1:])):
        if test_tuple.novel:
            red_button = True
            if red_button_pos == -1:
                red_button_pos = pos

            total_novel += 1
                
        ans_nov = float(detect_line.split(',')[1]) > detection_threshold
        if sys_declare_pos == -1 and ans_nov:
            sys_declare_pos = pos      

        # ground truth vector of species counts
        ground_truth_spe_counts = np.zeros((1, total_nbr_species)).ravel()
        ground_truth_spe_presence = np.zeros((1, total_nbr_species)).ravel()
        if not pd.isnull(test_tuple.agent1_id):
            ground_truth_spe_counts[int(test_tuple.agent1_id)] = int(test_tuple.agent1_count)
            ground_truth_spe_presence[int(test_tuple.agent1_id)] = 1

            if int(test_tuple.agent1_id) not in species_id2name_mapping:
                species_id2name_mapping[int(test_tuple.agent1_id)] = test_tuple.agent1_name
        else:
            # This is an empty image

            nbr_empty_imgs += 1
            if 0 not in species_id2name_mapping:
                species_id2name_mapping[0] = test_tuple.agent1_name
            continue

        if not pd.isnull(test_tuple.agent2_id):
            ground_truth_spe_counts[int(test_tuple.agent2_id)] = int(test_tuple.agent2_count)
            ground_truth_spe_presence[int(test_tuple.agent2_id)] = 1

            if int(test_tuple.agent2_id) not in species_id2name_mapping:
                species_id2name_mapping[int(test_tuple.agent2_id)] = test_tuple.agent2_name

        if not pd.isnull(test_tuple.agent3_id):
            ground_truth_spe_counts[int(test_tuple.agent3_id)] = int(test_tuple.agent3_count)
            ground_truth_spe_presence[int(test_tuple.agent3_id)] = 1

            if int(test_tuple.agent3_id) not in species_id2name_mapping:
                species_id2name_mapping[int(test_tuple.agent3_id)] = test_tuple.agent3_name

        # ground truth boolean vector of activity presence
        ground_truth_act_presence = np.zeros((1, total_nbr_activities)).ravel()
        # list_activities = ast.literal_eval(test_tuple.activities)
        list_activities_id = ast.literal_eval(test_tuple.activities_id)
        list_activities = ast.literal_eval(test_tuple.activities)

        for idx, act_id in enumerate(list_activities_id):
            ground_truth_act_presence[act_id] = 1

            if act_id not in activity_id2name_mapping:
                activity_id2name_mapping[act_id] = list_activities[idx]

        # vector of predicted species counts
        pred_spe_counts, pred_spe_presence, pred_act_presence = class_file_reader.get_answers(
            class_line, total_nbr_species, total_nbr_activities
        )

        # ground truth specie counts with all novel species count summed
        grd_trth_sp_cts_known_vs_novel = ground_truth_spe_counts[:nbr_known_species].tolist()
        grd_trth_sp_cts_known_vs_novel.append(
            sum(ground_truth_spe_counts[nbr_known_species:])
        )
        # Specie prediction counts with all novel species count summed
        pred_sp_cts_comb_novel = pred_spe_counts[:nbr_known_species]
        pred_sp_cts_comb_novel.append(
            sum(pred_spe_counts[nbr_known_species:])
        )

        # count_abs_error = species_count_error(grd_trth_sp_cts_known_vs_novel, pred_sp_cts_comb_novel, metric='AE')
        count_abs_error = species_count_error(ground_truth_spe_counts, pred_spe_counts, metric='AE')

        total += 1
        if not red_button:
            total_pre_red_btn += 1

            # ---------->>>  Brier Score and accuracy element for current image (presence/absence error)  <<<----------
            # Species presence error for current image
            for i in range(total_nbr_species):
                pre_red_per_spe_running_bs[i] += (ground_truth_spe_presence[i] - pred_spe_presence[i])**2
                # pre_red_per_spe_presence_acc[i] += 1 if ground_truth_spe_presence[i] == pred_spe_presence_yn[i] else 0

                # ------------------->>>  Accuracy counts element for current image  <<<------------------
                pre_red_per_spe_count_abs_err[i] += count_abs_error[i]
                pre_red_btn_per_spe_animal_count[i] += ground_truth_spe_counts[i]

            # Activity presence error for current images
            for i in range(total_nbr_activities):
                pre_red_per_act_running_bs[i] += (ground_truth_act_presence[i] - pred_act_presence[i])**2
                # pre_red_per_activity_acc[i] += 1 if ground_truth_activity_presence[i] == pred_activity_presence_yn[i] else 0

                if ground_truth_act_presence[i] == 1:
                    pre_red_act_present_yn[i] = 1

        else:
            total_post_red_btn += 1

            # ---------->>>  Brier Score and accuracy element for current image (presence/absence error)  <<<----------
            # Species presence error for current image

            for i in range(total_nbr_species):
                post_red_per_spe_running_bs[i] += (ground_truth_spe_presence[i] - pred_spe_presence[i])**2
                # post_red_per_spe_presence_acc[i] += 1 if ground_truth_spe_presence[i] == pred_spe_presence_yn[i] else 0

                # ------------------->>>  Accuracy counts element for current image  <<<------------------
                post_red_per_spe_count_abs_err[i] += count_abs_error[i]
                post_red_btn_per_spe_animal_count[i] += ground_truth_spe_counts[i]

            # Activity presence error for current images
            for i in range(total_nbr_activities):
                post_red_per_act_running_bs[i] += (ground_truth_act_presence[i] - pred_act_presence[i])**2
                # post_red_per_activity_acc[i] += 1 if ground_truth_activity_presence[i] == pred_activity_presence_yn[i] else 0

                if ground_truth_act_presence[i] == 1:
                    post_red_act_present_yn[act_id] = 1


        # >>>>>>>>>>>>>>>>>>>>>> SPECIES AND ACTIVITY PRESENCE (FOR EACH BOUNDING BOX) <<<<<<<<<<<<<<<<<<<<

        if (not pd.isnull(test_tuple.agent1_id)) & (pd.isnull(test_tuple.agent2_id)) :
            # This image contains a single species therefore the image level label can be pushed to the bounding
            # boxes to get a confusion matrix

            nbr_imgs_w_single_spe += 1

            pred_spe_in_boxes = np.argmax(bboxes_prediction_dict[test_tuple.image_path]['species_probs'], axis=1)
            for id_spe_pred in pred_spe_in_boxes:
                onehot_pred_spe = [0] * total_nbr_species
                onehot_pred_spe[id_spe_pred] = 1
                all_pred_spe_in_box.append(onehot_pred_spe)

                onehot_grd_trth = [0] * total_nbr_species
                onehot_grd_trth[int(test_tuple.agent1_id)] = 1
                all_grd_truth_spe_in_box.append(onehot_grd_trth)

        if len(list_activities_id) == 1:
            # This image contains a single activity therefore the image level label can be pushed to the bounding
            # boxes to get a confusion matrix

            nbr_imgs_w_single_act += 1

            pred_act_in_boxes = np.argmax(bboxes_prediction_dict[test_tuple.image_path]['activity_probs'], axis=1)
            for id_act_pred in pred_act_in_boxes:
                onehot_pred_act = [0] * total_nbr_activities
                onehot_pred_act[id_act_pred] = 1
                all_pred_act_in_box.append(onehot_pred_act)

                onehot_grd_trth_act = [0] * total_nbr_activities
                onehot_grd_trth_act[int(list_activities_id[0])] = 1
                all_grd_truth_act_in_box.append(onehot_grd_trth_act)


    all_grd_truth_spe_in_box = np.array(all_grd_truth_spe_in_box)
    all_pred_spe_in_box = np.array(all_pred_spe_in_box)

    all_grd_truth_act_in_box = np.array(all_grd_truth_act_in_box)
    all_pred_act_in_box = np.array(all_pred_act_in_box)

    # *****************************  ACTIVITY PERFORMANCE  ********************************
    
    # ---------->>  Activity presence Brier score  <<----------
    pre_red_per_act_bs = np.divide(pre_red_per_act_running_bs, total_pre_red_btn)
    # post_red_per_act_running_bs = np.divide(post_red_per_act_running_bs, total_post_red_btn)
    post_red_per_act_bs = np.array(
        [post_red_per_act_running_bs[i] / total_post_red_btn if total_post_red_btn > 0 else -1 \
            for i in range(len(post_red_per_act_running_bs))]
    ).ravel()

    # pre_red_act_mean_brier_score = np.mean(pre_red_per_act_running_bs)
    # post_red_act_mean_brier_score = np.mean(post_red_per_act_running_bs)

    # flag brier score of activities not present in the trial
    pre_red_per_act_bs = np.array(
        [pre_red_per_act_bs[i] if pre_red_act_present_yn[i] > 0 else -1 \
        for i in range(len(pre_red_per_act_bs))]
    )
    post_red_per_act_bs = np.array(
        [post_red_per_act_bs[i] if post_red_act_present_yn[i] > 0 else -1 \
        for i in range(len(post_red_per_act_bs))]
    )

    # --------->>  Activity presence Accuracty  <<----------
    per_act_acc = np.mean(all_grd_truth_act_in_box==all_pred_act_in_box, axis=0)
    mean_act_acc = np.mean(per_act_acc)

    # -------->>  Activity presence confusion matrix  <<---------
    act_grd_trth = np.argmax(all_grd_truth_act_in_box, axis=1)
    act_pred = np.argmax(all_pred_act_in_box, axis=1)
    unique_act = np.unique([act_grd_trth, act_pred])
    activity_cm = metrics.confusion_matrix(
        act_grd_trth, 
        act_pred,
        labels=np.sort(unique_act)
    )

    # *****************************  SPECIES PERFORMANCE  ********************************

    # ---------->>  Species presence Brier score  <<----------
    pre_red_per_spe_bs = np.divide(pre_red_per_spe_running_bs, total_pre_red_btn)
    # post_red_per_spe_running_bs= np.divide(post_red_per_spe_running_bs, total_post_red_btn)
    post_red_per_spe_bs = np.array(
        [post_red_per_spe_running_bs[i] / total_post_red_btn if total_post_red_btn > 0 else -1 \
            for i in range(len(post_red_per_spe_running_bs))]
    ).ravel()

    # flag brier score of species not present in the trial
    pre_red_per_spe_bs = np.array(
        [pre_red_per_spe_bs[i] if pre_red_btn_per_spe_animal_count[i] > 0 else -1 \
        for i in range(len(pre_red_per_spe_bs))]
    )
    post_red_per_spe_bs = np.array(
        [
        post_red_per_spe_bs[i] if post_red_btn_per_spe_animal_count[i] > 0 else -1 \
        for i in range(len(post_red_per_spe_bs))]
    )


    # pre_red_spe_mean_brier_score = np.mean(pre_red_per_spe_running_bs)
    # post_red_spe_mean_brier_score = np.mean(post_red_per_spe_running_bs)

    # --------->>  Species presence Accuracty  <<----------
    per_spe_acc = np.mean(all_grd_truth_spe_in_box==all_pred_spe_in_box, axis=0)
    mean_spe_acc = np.mean(per_spe_acc)

    # -------->>  Species presence confusion matrix  <<---------
    spe_grd_trth = np.argmax(all_grd_truth_spe_in_box, axis=1)
    spe_pred = np.argmax(all_pred_spe_in_box, axis=1)
    unique_spe = np.unique([spe_grd_trth, spe_pred])
    species_cm = metrics.confusion_matrix(
        spe_grd_trth, 
        spe_pred,
        labels=np.sort(unique_spe)
    )

    # -------->>  Species count error metric  <<--------

    # pre_red_per_spe_count_rel_err = np.divide(pre_red_per_spe_count_abs_err, pre_red_btn_per_spe_animal_count + epsilon)
    # post_red_per_spe_count_rel_err = np.divide(post_red_per_spe_count_abs_err, post_red_btn_per_spe_animal_count + epsilon)
    pre_red_per_spe_count_rel_err = np.array(
        [pre_red_per_spe_count_abs_err[i]/pre_red_btn_per_spe_animal_count[i] if pre_red_btn_per_spe_animal_count[i] > 0 else -1 \
            for i in range(len(pre_red_per_spe_count_rel_err))]
        ).ravel()
    post_red_per_spe_count_rel_err = np.array(
        [post_red_per_spe_count_abs_err[i]/post_red_btn_per_spe_animal_count[i] if post_red_btn_per_spe_animal_count[i] > 0 else -1 \
            for i in range(len(post_red_per_spe_count_rel_err))]
        ).ravel()
    
    # pre_red_per_spe_count_abs_err = pre_red_per_spe_count_abs_err / total_pre_red_btn
    # post_red_per_spe_count_abs_err = post_red_per_spe_count_abs_err / total_post_red_btn

    pre_red_per_spe_count_abs_err = np.array(
        [pre_red_per_spe_count_abs_err[i]/total_pre_red_btn if pre_red_btn_per_spe_animal_count[i] > 0 else -1 \
            for i in range(len(pre_red_per_spe_count_abs_err))]
    ).ravel()

    post_red_per_spe_count_abs_err = np.array(
        [post_red_per_spe_count_abs_err[i] / total_post_red_btn if post_red_btn_per_spe_animal_count[i] > 0 else -1 \
            for i in range(len(post_red_per_spe_count_abs_err))]
    ).ravel()

    '''
    # **** flag absolute and relative errors of species not present in the trial ****
    pre_red_per_spe_count_abs_err = [
        pre_red_per_spe_count_abs_err[i] if pre_red_btn_per_spe_animal_count[i] > 0 else -1 \
        for i in range(len(pre_red_per_spe_count_abs_err))
    ]
    post_red_per_spe_count_abs_err = [
        post_red_per_spe_count_abs_err[i] if post_red_btn_per_spe_animal_count[i] > 0 else -1 \
        for i in range(len(post_red_per_spe_count_abs_err))
    ]

    pre_red_per_spe_count_rel_err = [
        pre_red_per_spe_count_rel_err[i] if pre_red_btn_per_spe_animal_count[i] > 0 else -1 \
        for i in range(len(pre_red_per_spe_count_rel_err))
    ]
    post_red_per_spe_count_rel_err = [
        post_red_per_spe_count_rel_err[i] if post_red_btn_per_spe_animal_count[i] > 0 else -1 \
        for i in range(len(post_red_per_spe_count_rel_err))
    ]
    '''

    # Aggregate species presence
    pre_red_spe_mean_brier_score = np.mean([x for x in pre_red_per_spe_bs if x > 0])
    post_red_spe_mean_brier_score = np.mean([x for x in post_red_per_spe_bs if x> 0])

    # Aggregate activities presence
    pre_red_act_mean_brier_score = np.mean([x for x in pre_red_per_act_bs if x > 0])
    post_red_act_mean_brier_score = np.mean([x for x in post_red_per_act_bs if x > 0])

    # Aggregate species counts
    pre_red_mean_spe_count_abs_err = np.mean([x for x in pre_red_per_spe_count_abs_err if x > 0])
    post_red_mean_spe_count_abs_err = np.mean([x for x in post_red_per_spe_count_abs_err if x > 0])

    pre_red_mean_spe_count_rel_err = np.mean([x for x in pre_red_per_spe_count_rel_err if x > 0]) 
    post_red_mean_spe_count_rel_err = np.mean([x for x in post_red_per_spe_count_rel_err if x > 0])

    # log.write(f'{" "*86} {total_top_1_score}  {total_top_3_score}\n')
    if sys_declare_pos == -1:
        detect = 'Miss'
        delay = ' NA'
    elif sys_declare_pos < red_button_pos:
        detect = ' FA '
        delay = ' NA'
    else:
        detect = 'HIT '
        delay = f'{sys_declare_pos - red_button_pos:3}'


    # ************ Write all performance for current test_id *****************
    all_performances['Red_button'][test_id] = red_button_pos
    all_performances['Red_button_declared'][test_id] = sys_declare_pos
    all_performances['Red_button_result'][test_id] = detect
    all_performances['Delay'][test_id] = delay
    all_performances['Total'][test_id] = total
    all_performances['Novel'][test_id] = total_novel

    all_performances['Nber_empty_images'][test_id] = nbr_empty_imgs
    all_performances['Nber_images_single_species'][test_id] = nbr_imgs_w_single_spe
    all_performances['Nber_images_single_activity'][test_id] = nbr_imgs_w_single_act

    all_performances['Species_id2name'][test_id] = species_id2name_mapping
    all_performances['Activity_id2name'][test_id] = activity_id2name_mapping

    # species counts
    counts = {
        'pre_red_btn':{
            'absolute_error': pre_red_per_spe_count_abs_err,
            'relative_error': pre_red_per_spe_count_rel_err
        },
        'post_red_btn':{
            'absolute_error': post_red_per_spe_count_abs_err,
            'relative_error': post_red_per_spe_count_rel_err
        }
    }
    all_performances['Species_counts'][test_id] = counts

    # Per species presence
    species_presence = {
        'pre_red_btn':{
            'bs': pre_red_per_spe_bs,
            'acc': pre_red_per_spe_presence_acc,
            'auc': pre_red_per_spe_auc
        },
        'post_red_btn':{
            'bs': post_red_per_spe_bs,
            'acc': post_red_per_spe_presence_acc,
            'auc': post_red_per_spe_auc
        }
    }
    all_performances['Per_species_presence'][test_id] = species_presence

    # Species presence confusion matrix
    conf_mat = {
        'species': {
            'cm': species_cm,
            'species_ids': np.sort(unique_spe)
        },
        'activity': {
            'cm': activity_cm,
            'activity_ids': np.sort(unique_act)
        }
    }
    all_performances['Confusion_matrices'][test_id] = conf_mat


    # Aggregate species presence
    agg_species_presence = {
        'pre_red_btn':{
            'avg_bs': round(pre_red_spe_mean_brier_score, 3)
        },
        'post_red_btn':{
            'avg_bs': round(post_red_spe_mean_brier_score, 3)
        }
    }
    all_performances['Aggregate_species_presence'][test_id] = agg_species_presence

    # Aggregate species counts
    agg_species_counts = {
        'pre_red_btn':{
            'avg_abs_err': round(pre_red_mean_spe_count_abs_err, 3),
            'avg_rel_err': round(pre_red_mean_spe_count_rel_err, 3)
        },
        'post_red_btn':{
            'avg_abs_err': round(post_red_mean_spe_count_abs_err, 3),
            'avg_rel_err': round(post_red_mean_spe_count_rel_err, 3)
        }
    }
    all_performances['Aggregate_species_counts'][test_id] = agg_species_counts

    # Per activity presence
    activity_presence = {
        'pre_red_btn':{
            'bs': pre_red_per_act_bs
        },
        'post_red_btn':{
            'bs': post_red_per_act_bs
        }
    }
    all_performances['Per_activity_presence'][test_id] = activity_presence

    # Aggregate activity presence
    agg_activity_presence = {
        'pre_red_btn':{
            'avg_bs': round(pre_red_act_mean_brier_score, 3)
        },
        'post_red_btn':{
            'avg_bs': round(post_red_act_mean_brier_score, 3)
        }
    }
    all_performances['Aggregate_activity_presence'][test_id] = agg_activity_presence


def score_tests(
    test_dir, sys_output_dir, bboxes_dir, session_id, class_file_reader, log_dir,
    save_symlinks, dataset_root, detection_threshold, spe_presence_threshold,
        act_presence_threshold
):   
    
    nbr_rounds = 100
    
    # test_ids = open(test_dir/'test_ids.csv', 'r').read().splitlines()
    import pathlib 
    test_ids = []
    for p in pathlib.Path(test_dir).glob('*'):
        if p.suffix != '.csv':
            continue
        test_name = p.name.split('_')[0]
        test_ids.append(test_name)
    
    # test_ids = ['OND.102.000']

    all_performances = {
        'Red_button':{},
        'Red_button_declared': {},
        'Red_button_result': {},
        'Delay': {},
        'Total': {},
        'Novel': {},
        'Nber_empty_images': {},
        'Nber_images_single_species': {},
        'Nber_images_single_activity': {},
        'Per_species_presence': {},
        'Aggregate_species_presence': {},
        'Aggregate_species_counts': {},
        'Per_activity_presence': {},
        'Aggregate_activity_presence': {},
        'Species_counts': {},
        'Confusion_matrices': {},
        'Species_id2name': {},
        'Activity_id2name': {}
    }
    for test_id in test_ids:
        # if 'OND' in test_id and '100.000' not in test_id:
        if 'OND' in test_id:# and '1061.000' not in test_id and '1062.000' not in test_id:
            metadata = json.load(open(test_dir / f'{test_id}_metadata.json', 'r'))
            test_df = pd.read_csv(test_dir / f'{test_id}_single_df.csv')

            pkl_fname = os.path.join(bboxes_dir, test_id+'/'+test_id+'.pkl')
            print(pkl_fname)
            assert os.path.exists(pkl_fname)
            with (open(pkl_fname, "rb")) as pkl_file:
                boxes_pred_dict = pickle.load(pkl_file)

            # test_id = test_id[4:]

            detect_lines = []
            class_lines = []
            # for round_ in range(nbr_rounds):
            #     if (sys_output_dir / f'{session_id}.{test_id}_{round_}_detection.csv').exists():
            #         detect_lines.append(open(sys_output_dir / f'{session_id}.{test_id}_{round_}_detection.csv').read().splitlines())
            #         class_lines.append(open(sys_output_dir / f'{session_id}.{test_id}_{round_}_classification.csv').read().splitlines())

            #     else:
            #         print(f'No results found for Test {session_id}.{test_id}_{round_}.')
            # detect_lines = np.concatenate(detect_lines)

            # class_lines = np.concatenate(class_lines)
            
            if (sys_output_dir / f'{session_id}.{test_id}_detection.csv').exists():
                detect_lines = open(sys_output_dir / f'{session_id}.{test_id}_detection.csv').read().splitlines()
                class_lines = open(sys_output_dir / f'{session_id}.{test_id}_classification.csv').read().splitlines()
                
            else:
                print(f'No results found for Test {session_id}.{test_id}_.')
           
            with open(log_dir / f'{test_id}.log', 'w') as log:
                score_test(
                    test_id, metadata, test_df, detect_lines, class_lines, class_file_reader,
                    log, all_performances, detection_threshold, spe_presence_threshold, act_presence_threshold
                )
                # score_test_from_boxes(
                #     test_id, metadata, test_df, detect_lines, class_lines, boxes_pred_dict, class_file_reader,
                #     log, all_performances, detection_threshold, presence_threshold
                # )
        

    write_results_to_log(all_performances, output_path=log_dir)
    # print_confusion_matrices(all_performances, log_dir / "confusion.pdf")



def main():
    p = ArgumentParser()
    p.add_argument("test_root", help="test specification directory, "
                   "named 'api_tests' for tests generated by umd_test_generate")
    p.add_argument("sys_output_root", help="system output directory, parent of 'OND'")
    p.add_argument("log_dir", help="where to write scoring output directory")
    p.add_argument("--save_symlinks", action="store_true",
                   help="flag to request saving a classified tree of image symlinks")
    p.add_argument("--dataset_root", help="path to UMD image dataset, "
                   "required for save_symlinks")
    p.add_argument('--detection_threshold', type=float, default=0.5)
    p.add_argument('--activity_presence_threshold', type=float, default=0.4)
    p.add_argument('--species_presence_threshold', type=float, default=0.4)
    p.add_argument('--box_pred_dir', default='./session/temp/logsDete10',
        help="dir containing the system\'s prediction of each box (predicted species and activity)")
    args = p.parse_args()
    if args.save_symlinks and not args.dataset_root:
        raise Exception("dataset_root must be specified if you want to save image symlinks")
    dataset_root = Path(args.dataset_root) if args.dataset_root else None
    test_dir = Path(args.test_root) #/"OND"/'svo_classification'
    sys_output_dir = Path(args.sys_output_root) #/'OND'/'svo_classification'
    session_ids = set()
    for file in sys_output_dir.iterdir():
        session_id = file.name.split('.')[0]
        session_ids.add(session_id)
    
    if len(session_ids) > 1:
        raise Exception('More than one session id in results dir')
    session_id = list(session_ids)[0]
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)
    for file in log_dir.iterdir():
        if file.is_dir():
            shutil.rmtree(file)
        else:
            file.unlink()

    detection_threshold = args.detection_threshold
    species_presence_threshold = args.species_presence_threshold
    activity_presence_threshold = args.activity_presence_threshold
    bboxes_pred_dir = Path(args.box_pred_dir)
    
    score_tests(
        test_dir, sys_output_dir, bboxes_pred_dir, session_id, ClassFileReader(), log_dir,
        args.save_symlinks, dataset_root, detection_threshold, species_presence_threshold,
        activity_presence_threshold
    )

if __name__ == '__main__':
    main()