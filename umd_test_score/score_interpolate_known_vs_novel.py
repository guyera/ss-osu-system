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
import shutil
import numpy as np
import sklearn.metrics as metrics
import ast

from write_results_known_vs_novel import write_results_to_csv
from helpers import species_count_error, percent_string
from boostrap_conf_int import boostrap_conf_interval


epsilon = 1e-6


def interpolated_prec_rec_f1(y_true, y_pred_pre_retrain, y_pred_post_retrain):
    assert len(y_true) == len(y_pred_pre_retrain)
    assert len(y_true) == len(y_pred_post_retrain)

    if isinstance(y_true, pd.Series) or isinstance(y_true, pd.DataFrame):
        y_true_arr = np.array(y_true)
        y_pred_pre_retrain_arr = np.array(y_pred_pre_retrain)
        y_pred_post_retrain_arr = np.array(y_pred_post_retrain)
    else:
        assert isinstance(y_true, np.array)
        y_true_arr = y_true.copy()
        y_pred_pre_retrain_arr = y_pred_pre_retrain.copy()
        y_pred_post_retrain_arr = y_pred_post_retrain.copy()

    lambda_vec = np.array([(i - 1) / len(y_true) for i in range(1, len(y_true) + 1)])
    lambda_x_pre_retrain = (1 - lambda_vec) * y_pred_pre_retrain_arr
    lambda_x_post_retrain = lambda_vec * y_pred_post_retrain_arr
    interpolated_pred = lambda_x_pre_retrain + lambda_x_post_retrain
    sum_inter_pred_x_true = np.sum(np.multiply(interpolated_pred, y_true_arr))

    if sum_inter_pred_x_true == 0 or np.sum(y_true_arr) == 0:
        return 0.0, 0.0, 0.0
    precision = sum_inter_pred_x_true / np.sum(interpolated_pred)
    recall = sum_inter_pred_x_true / np.sum(y_true_arr)
    f1_score = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1_score


def interpolated_species_count_error(grd_truth_counts, pred_counts_pre_retrain, pred_counts_post_retrain):
    assert len(grd_truth_counts) == len(pred_counts_pre_retrain)
    assert len(grd_truth_counts) == len(pred_counts_post_retrain)

    if isinstance(grd_truth_counts, pd.Series) or isinstance(grd_truth_counts, pd.DataFrame):
        grd_truth_counts_arr = np.array(grd_truth_counts)
        pred_counts_pre_retrain_arr = np.array(pred_counts_pre_retrain)
        pred_counts_post_retrain_arr = np.array(pred_counts_post_retrain)
    else:
        assert isinstance(grd_truth_counts, np.array)
        grd_truth_counts_arr = grd_truth_counts.copy()
        pred_counts_pre_retrain_arr = pred_counts_pre_retrain.copy()
        pred_counts_post_retrain_arr = pred_counts_post_retrain.copy()
    
    abs_err_pre_retrain = np.abs(grd_truth_counts_arr - pred_counts_pre_retrain_arr)
    abs_err_post_retrain = np.abs(grd_truth_counts_arr - pred_counts_post_retrain_arr)

    lambda_vec = np.array([(i - 1) / len(grd_truth_counts) for i in range(1, len(grd_truth_counts) + 1)])
    lambda_x_pre_retrain = np.multiply((1 - lambda_vec).reshape(-1, 1), abs_err_pre_retrain)
    lambda_x_post_retrain = np.multiply(lambda_vec.reshape(-1, 1), abs_err_post_retrain)

    interpolated_abs_err = lambda_x_pre_retrain + lambda_x_post_retrain
    if np.sum(interpolated_abs_err) == 0:
        return -1
    return np.sum(interpolated_abs_err) / len(grd_truth_counts)

def score_test(
    test_id, metadata, test_df, detect_lines, class_lines_pre_retrain,
    class_lines_post_retrain, class_file_reader, log, all_performances, 
    detection_threshold, spe_presence_threshold=0.5, act_presence_threshold=0.5, 
    estimate_ci=False, nbr_samples_conf_int=300, len_test_phase=1000, output_dir=None
):
    """
    Computes the score of the system for detection and classification tasks
    Arguments:
        test_id: id of the novelty type
        test_df: ground truth test (or validation) dataframe
        detect_lines: system detection results
        class_lines_pre_retrain: system classification results for novelty unaware model
        class_lines_post_retrain: system classification results for post retraining model
        all_performances: output file (dictionary) where the performance should be saved
        detection_threshold: threshold used for declaring an entry to be novel
        spe_presence_threshold: threshold used for declaring a species present based on predicted probability
        act_presence_threshold: threshold used for declaring an activity present based on predicted probability
        estimate_ci: boolean value for deciding whether to compute the confidence intervals with prediction
        nbr_samples_conf_int: number of samples used in estimating the confidence intervals via boostraping
        len_test_phase: number of post-novelty images used for testing (after accommodation strategy applied)
    """
    # The metadata file is not currently used.
    # import ipdb; ipdb.set_trace()
    total_novel, total = 0, 0
    test_tuples = test_df.itertuples()
    len_test = len(test_df)
    red_button = False
    red_button_pos = -1
    sys_declare_pos = -1

    # index where test phase starts in the trial
    start_test_phase = len_test - len_test_phase  

    # indices novel images
    idx_novel = test_df.loc[test_df['novel'] == 1].index
    idx_novel_test_post_red = [x for x in idx_novel if x >= start_test_phase]
    idx_novel_post_red = [x for x in idx_novel if x not in idx_novel_test_post_red]

    # indices novel images
    idx_known = test_df.loc[test_df['novel'] == 0].index
    idx_known_test_post_red = [x for x in idx_known if x >= start_test_phase]
    idx_known_post_red = [x for x in idx_known if x not in idx_known_test_post_red]

    spe_counts_cols = [x for x in class_lines_post_retrain[0].split(',') if 'species_' in x and '_count' in x]

    spe_presence_cols = [x for x in class_lines_post_retrain[0].split(',') if 'species_' in x and '_presence' in x]
    total_nbr_species = len(spe_presence_cols)

    act_presence_cols = [x for x in class_lines_post_retrain[0].split(',') if 'activity_' in x]
    total_nbr_activities = len(act_presence_cols)

    # model before accommodation (before retraining)
    class_lines_post_retrain = pd.DataFrame(
        [class_line_post_retrain.split(',') for class_line_post_retrain in class_lines_post_retrain[1:]], 
        columns=class_lines_post_retrain[0].split(',')
    )
    all_pred_spe_counts_post_retrain = class_lines_post_retrain[spe_counts_cols].astype(float).copy(deep=True)
    all_pred_spe_presence_post_retrain = class_lines_post_retrain[spe_presence_cols].astype(float).copy(deep=True)
    all_pred_act_presence_post_retrain = class_lines_post_retrain[act_presence_cols].astype(float).copy(deep=True)

    # model after accommodation (before retraining)
    class_lines_pre_retrain = pd.DataFrame(
        [class_line_pre_retrain.split(',') for class_line_pre_retrain in class_lines_pre_retrain[1:]], 
        columns=class_lines_pre_retrain[0].split(',')
    )
    all_pred_spe_counts_pre_retrain = class_lines_pre_retrain[spe_counts_cols].astype(float).copy(deep=True)
    all_pred_spe_presence_pre_retrain = class_lines_pre_retrain[spe_presence_cols].astype(float).copy(deep=True)
    all_pred_act_presence_pre_retrain = class_lines_pre_retrain[act_presence_cols].astype(float).copy(deep=True)

    all_grd_truth_spe_counts = np.zeros((len_test, total_nbr_species))
    all_grd_truth_spe_presence = np.zeros((len_test, total_nbr_species))
    all_grd_truth_act_presence = np.zeros((len_test, total_nbr_activities))

    pre_red_per_spe_precision, pre_red_per_spe_recall, pre_red_per_spe_f1, pre_red_per_spe_auc = {}, {}, {}, {}
    post_red_per_spe_precision, post_red_per_spe_recall, post_red_per_spe_f1, post_red_per_spe_auc = {}, {}, {}, {}
    test_post_red_per_spe_precision, test_post_red_per_spe_recall, test_post_red_per_spe_f1, test_post_red_per_spe_auc = {}, {}, {}, {}

    pre_red_per_act_precision, pre_red_per_act_recall, pre_red_per_act_f1, pre_red_per_act_auc = {}, {}, {}, {}
    post_red_per_act_precision, post_red_per_act_recall, post_red_per_act_f1, post_red_per_act_auc = {}, {}, {}, {}
    test_post_red_per_act_precision, test_post_red_per_act_recall, test_post_red_per_act_f1, test_post_red_per_act_auc = {}, {}, {}, {}

    total_pre_red_btn, total_post_red_btn = 0, 0
    nbr_imgs_w_single_spe, nbr_imgs_w_single_act = 0, 0

    species_id2name_mapping = {}
    activity_id2name_mapping = {}

    pre_red_abs_err_count, pre_red_rel_err_count = {}, {}
    post_red_abs_err_count, post_red_rel_err_count = {}, {}
    test_post_red_abs_err_count, test_post_red_rel_err_count = {}, {}
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
    all_pred_spe_counts_post_retrain.rename(columns=new_spe_count_col_names, inplace=True)
    all_pred_spe_presence_post_retrain.rename(columns=new_spe_presence_col_names, inplace=True)
    all_pred_act_presence_post_retrain.rename(columns=new_act_col_names, inplace=True)

    all_pred_spe_counts_pre_retrain.rename(columns=new_spe_count_col_names, inplace=True)
    all_pred_spe_presence_pre_retrain.rename(columns=new_spe_presence_col_names, inplace=True)
    all_pred_act_presence_pre_retrain.rename(columns=new_act_col_names, inplace=True)

    all_grd_truth_spe_counts = pd.DataFrame(all_grd_truth_spe_counts, columns=all_pred_spe_counts_post_retrain.columns)
    all_grd_truth_spe_presence = pd.DataFrame(all_grd_truth_spe_presence, columns=all_pred_spe_presence_post_retrain.columns)
    all_grd_truth_act_presence = pd.DataFrame(all_grd_truth_act_presence, columns=all_pred_act_presence_post_retrain.columns)

    # ** Drop empty species from ground truth and predicted data
    for df in [
        all_pred_spe_counts_pre_retrain, all_pred_spe_presence_pre_retrain, all_pred_act_presence_pre_retrain, 
        all_pred_spe_counts_post_retrain, all_pred_spe_presence_post_retrain, all_pred_act_presence_post_retrain,
        all_grd_truth_spe_counts, all_grd_truth_spe_presence, all_grd_truth_act_presence
        ]:
        if 'blank' in df.columns:
            df.drop(columns='blank', inplace=True)


    # *****************************  ACTIVITY PRESENCE PERFORMANCE  ********************************

    for act in all_grd_truth_act_presence.columns:
        if act not in activity_id2name_mapping.values():
            pre_red_per_act_auc[act] = {
                'known': {'value': -1},
                'novel': {'value': -1}
            }
            pre_red_per_act_precision[act] = {
                'known': {'value': -1},
                'novel': {'value': -1}
            }
            pre_red_per_act_recall[act] = {
                'known': {'value': -1},
                'novel': {'value': -1}
            }
            pre_red_per_act_f1[act] = {
                'known': {'value': -1},
                'novel': {'value': -1}
            }

            post_red_per_act_auc[act] = {
                'known': {'value': -1},
                'novel': {'value': -1}
            }
            post_red_per_act_precision[act] = {
                'known': {'value': -1},
                'novel': {'value': -1}
            }
            post_red_per_act_recall[act] = {
                'known': {'value': -1},
                'novel': {'value': -1}
            }
            post_red_per_act_f1[act] = {
                'known': {'value': -1},
                'novel': {'value': -1}
            }
            continue

        # ---------->>  Activity presence Precision, Recall, F1  <<----------
        # ** pre retraining
        all_pred_act_presence_yn_pre_retrain = all_pred_act_presence_pre_retrain.copy(deep=True)
        all_pred_act_presence_yn_pre_retrain = (all_pred_act_presence_yn_pre_retrain >= act_presence_threshold).astype(int)

        # ** post retraining
        all_pred_act_presence_yn_post_retrain = all_pred_act_presence_post_retrain.copy(deep=True)
        all_pred_act_presence_yn_post_retrain = (all_pred_act_presence_yn_post_retrain >= act_presence_threshold).astype(int)

        # ++++ Known
        try:
            pre_red_precision, pre_red_rec, pre_red_f1, _ = metrics.precision_recall_fscore_support(
                all_grd_truth_act_presence[act].iloc[:total_pre_red_btn], 
                all_pred_act_presence_yn_pre_retrain[act].iloc[:total_pre_red_btn],
                average='binary',
                zero_division=0.0
            )
            
            post_red_precision_known, post_red_rec_known, post_red_f1_known = interpolated_prec_rec_f1(
                y_true=all_grd_truth_act_presence[act].iloc[idx_known_post_red], 
                y_pred_pre_retrain=all_pred_act_presence_yn_pre_retrain[act].iloc[idx_known_post_red], 
                y_pred_post_retrain=all_pred_act_presence_yn_post_retrain[act].iloc[idx_known_post_red]
            )

            test_post_red_precision_known, test_post_red_rec_known, test_post_red_f1_known, _ = metrics.precision_recall_fscore_support(
                all_grd_truth_act_presence[act].iloc[idx_known_test_post_red], 
                all_pred_act_presence_yn_post_retrain[act].iloc[idx_known_test_post_red],
                average='binary',
                zero_division=0.0
            )
                

            # Precision
            pre_red_per_act_precision[act] = {
                'known':{
                    'value': pre_red_precision if pre_red_precision != 0.0 else -1,
                    'ci': -1
                }
            }
            post_red_per_act_precision[act] = {
                'known':{
                    'value': post_red_precision_known if post_red_precision_known != 0.0 else -1,
                    'ci': -1
                }
            }
            test_post_red_per_act_precision[act] = {
                'known':{
                    'value': test_post_red_precision_known if test_post_red_precision_known != 0.0 else -1,
                    'ci': -1
                }
            }

            # Recall
            pre_red_per_act_recall[act] = {
                'known':{
                    'value': pre_red_rec if pre_red_rec != 0.0 else -1,
                    'ci': -1
                }
            }
            post_red_per_act_recall[act] = {
                'known':{
                    'value': post_red_rec_known if post_red_rec_known != 0.0 else -1,
                    'ci': -1
                }
            }
            test_post_red_per_act_recall[act] = {
                'known':{
                    'value': test_post_red_rec_known if test_post_red_rec_known != 0.0 else -1,
                    'ci': -1
                }
            }

            # F1 score
            pre_red_per_act_f1[act] = {
                'known':{
                    'value': pre_red_f1 if pre_red_f1 != 0.0 else -1,
                    'ci': -1
                }
            }
            post_red_per_act_f1[act] = {
                'known':{
                    'value': post_red_f1_known if post_red_f1_known != 0.0 else -1,
                    'ci': -1
                }
            }
            test_post_red_per_act_f1[act] = {
                'known':{
                    'value': test_post_red_f1_known if test_post_red_f1_known != 0.0 else -1,
                    'ci': -1
                }
            }
        except Exception as ex:
            print('+++ The following exception has occured:', ex)
            pre_red_per_act_precision[act] = {'known':{'value': -1}}
            pre_red_per_act_recall[act] = {'known':{'value': -1}}
            pre_red_per_act_f1[act] = {'known':{'value': -1}}
            post_red_per_act_precision[act] = {'known':{'value': -1}}
            post_red_per_act_recall[act] = {'known':{'value': -1}}
            post_red_per_act_f1[act] = {'known':{'value': -1}}
            test_post_red_per_act_precision[act] = {'known':{'value': -1}}
            test_post_red_per_act_recall[act] = {'known':{'value': -1}}
            test_post_red_per_act_f1[act] = {'known':{'value': -1}}
            continue

        # ++++ Novel
        try:
            pre_red_precision, pre_red_rec, pre_red_f1, _ = metrics.precision_recall_fscore_support(
                all_grd_truth_act_presence[act].iloc[:total_pre_red_btn], 
                all_pred_act_presence_yn_pre_retrain[act].iloc[:total_pre_red_btn],
                average='binary',
                zero_division=0.0
            )
            
            post_red_precision_novel, post_red_rec_novel, post_red_f1_novel = interpolated_prec_rec_f1(
                y_true=all_grd_truth_act_presence[act].iloc[idx_novel_post_red], 
                y_pred_pre_retrain=all_pred_act_presence_yn_pre_retrain[act].iloc[idx_novel_post_red], 
                y_pred_post_retrain=all_pred_act_presence_yn_post_retrain[act].iloc[idx_novel_post_red]
            )

            test_post_red_precision_novel, test_post_red_rec_novel, test_post_red_f1_novel, _ = metrics.precision_recall_fscore_support(
                all_grd_truth_act_presence[act].iloc[idx_novel_test_post_red], 
                all_pred_act_presence_yn_post_retrain[act].iloc[idx_novel_test_post_red],
                average='binary',
                zero_division=0.0
            )
                

            # Precision
            pre_red_per_act_precision[act]['novel'] = {
                'value': pre_red_precision if pre_red_precision != 0.0 else -1,
                'ci': -1
            }
            post_red_per_act_precision[act]['novel'] = {
                'value': post_red_precision_novel if post_red_precision_novel != 0.0 else -1,
                'ci': -1
            }
            test_post_red_per_act_precision[act]['novel'] = {
                'value': test_post_red_precision_novel if test_post_red_precision_novel != 0.0 else -1,
                'ci': -1
            }

            # Recall
            pre_red_per_act_recall[act]['novel'] = {
                'value': pre_red_rec if pre_red_rec != 0.0 else -1,
                'ci': -1
            }
            post_red_per_act_recall[act]['novel'] = {
                'value': post_red_rec_novel if post_red_rec_novel != 0.0 else -1,
                'ci': -1
            }
            test_post_red_per_act_recall[act]['novel'] = {
                'value': test_post_red_rec_novel if test_post_red_rec_novel != 0.0 else -1,
                'ci': -1
            }

            # F1 score
            pre_red_per_act_f1[act]['novel'] = {
                'value': pre_red_f1 if pre_red_f1 != 0.0 else -1,
                'ci': -1
            }
            post_red_per_act_f1[act]['novel'] = {
                'value': post_red_f1_novel if post_red_f1_novel != 0.0 else -1,
                'ci': -1
            }
            test_post_red_per_act_f1[act]['novel'] = {
                'value': test_post_red_f1_novel if test_post_red_f1_novel != 0.0 else -1,
                'ci': -1
            }
        except Exception as ex:
            print('+++ The following exception has occured:', ex)
            pre_red_per_act_precision[act]['novel'] = {'value': -1}
            pre_red_per_act_recall[act]['novel'] = {'value': -1}
            pre_red_per_act_f1[act]['novel'] = {'value': -1}
            post_red_per_act_precision[act]['novel'] = {'value': -1}
            post_red_per_act_recall[act]['novel'] = {'value': -1}
            post_red_per_act_f1[act]['novel'] = {'value': -1}
            test_post_red_per_act_precision[act]['novel'] = {'value': -1}
            test_post_red_per_act_recall[act]['novel'] = {'value': -1}
            test_post_red_per_act_f1[act]['novel'] = {'value': -1}
            continue


    # *****************************  SPECIES PRESENCE PERFORMANCE  ********************************

    for spe in all_grd_truth_spe_presence.columns:

        if spe not in species_id2name_mapping.values():
            pre_red_per_spe_auc[spe] = {
                'known': {'value': -1},
                'novel': {'value': -1}
            }
            pre_red_per_spe_precision[spe] = {
                'known': {'value': -1},
                'novel': {'value': -1}
            }
            pre_red_per_spe_recall[spe] = {
                'known': {'value': -1},
                'novel': {'value': -1}
            }
            pre_red_per_spe_f1[spe] = {
                'known': {'value': -1},
                'novel': {'value': -1}
            }

            post_red_per_spe_auc[spe] = {
                'known': {'value': -1},
                'novel': {'value': -1}
            }
            post_red_per_spe_precision[spe] = {
                'known': {'value': -1},
                'novel': {'value': -1}
            }
            post_red_per_spe_recall[spe] = {
                'known': {'value': -1},
                'novel': {'value': -1}
            }
            post_red_per_spe_f1[spe] = {
                'known': {'value': -1},
                'novel': {'value': -1}
            }

            test_post_red_per_spe_auc[spe] = {
                'known': {'value': -1},
                'novel': {'value': -1}
            }
            test_post_red_per_spe_precision[spe] = {
                'known': {'value': -1},
                'novel': {'value': -1}
            }
            test_post_red_per_spe_recall[spe] = {
                'known': {'value': -1},
                'novel': {'value': -1}
            }
            test_post_red_per_spe_f1[spe] = {
                'known': {'value': -1},
                'novel': {'value': -1}
            }
            continue
        

        # ---------->>  Species presence Precision, Recall, F1  <<----------
        all_pred_spe_presence_yn_pre_retrain = all_pred_spe_presence_pre_retrain.copy(deep=True)
        all_pred_spe_presence_yn_pre_retrain = (all_pred_spe_presence_yn_pre_retrain >= spe_presence_threshold).astype(int)

        all_pred_spe_presence_yn_post_retrain = all_pred_spe_presence_post_retrain.copy(deep=True)
        all_pred_spe_presence_yn_post_retrain = (all_pred_spe_presence_yn_post_retrain >= spe_presence_threshold).astype(int)

        # ** Known
        try:
            pre_red_precision, pre_red_rec, pre_red_f1, _ = metrics.precision_recall_fscore_support(
                all_grd_truth_spe_presence[spe].iloc[:total_pre_red_btn], 
                all_pred_spe_presence_yn_pre_retrain[spe].iloc[:total_pre_red_btn],
                average='binary',
                zero_division=0.0
            )
            post_red_precision_known, post_red_rec_known, post_red_f1_known = interpolated_prec_rec_f1(
                y_true=all_grd_truth_spe_presence[spe].iloc[idx_known_post_red], 
                y_pred_pre_retrain=all_pred_spe_presence_yn_pre_retrain[spe].iloc[idx_known_post_red], 
                y_pred_post_retrain=all_pred_spe_presence_yn_post_retrain[spe].iloc[idx_known_post_red]
            )
            test_post_red_precision_known, test_post_red_rec_known, test_post_red_f1_known, _ = metrics.precision_recall_fscore_support(
                all_grd_truth_spe_presence[spe].iloc[idx_known_test_post_red], 
                all_pred_spe_presence_yn_post_retrain[spe].iloc[idx_known_test_post_red],
                average='binary',
                zero_division=0.0
            )

            
            # Precision
            pre_red_per_spe_precision[spe] = {
                'known': {
                    'value': pre_red_precision if pre_red_precision != 0.0 else -1,
                    'ci': -1
                }
            }
            post_red_per_spe_precision[spe] = {
                'known': {
                    'value': post_red_precision_known if post_red_precision_known != 0.0 else -1,
                    'ci': -1
                }
            }
            test_post_red_per_spe_precision[spe] = {
                'known': {
                    'value': test_post_red_precision_known if test_post_red_precision_known != 0.0 else -1,
                    'ci': -1
                }
            }

            # Recall
            pre_red_per_spe_recall[spe] = {
                'known': {
                    'value': pre_red_rec if pre_red_rec != 0.0 else -1,
                    'ci': -1
                }
            }
            post_red_per_spe_recall[spe] = {
                'known': {
                    'value': post_red_rec_known if post_red_rec_known != 0.0 else -1,
                    'ci': -1
                }
            }
            test_post_red_per_spe_recall[spe] = {
                'known': {
                    'value': test_post_red_rec_known if test_post_red_rec_known != 0.0 else -1,
                    'ci': -1
                }
            }

            # F1 score
            pre_red_per_spe_f1[spe] = {
                'known': {
                    'value': pre_red_f1 if pre_red_f1 != 0.0 else -1,
                    'ci': -1
                }
            }
            post_red_per_spe_f1[spe] = {
                'known': {
                    'value': post_red_f1_known if post_red_f1_known != 0.0 else -1,
                    'ci': -1
                }
            }
            test_post_red_per_spe_f1[spe] = {
                'known': {
                    'value': test_post_red_f1_known if test_post_red_f1_known != 0.0 else -1,
                    'ci': -1
                }
            }

        except Exception as ex:
            print('**** Exception when computing species presence metrics:', ex)
            pre_red_per_spe_auc[spe] = {
                'known': {'value': -1},
                'novel': {'value': -1}
            }
            pre_red_per_spe_precision[spe] = {
                'known': {'value': -1},
                'novel': {'value': -1}
            }
            pre_red_per_spe_recall[spe] = {
                'known': {'value': -1},
                'novel': {'value': -1}
            }
            pre_red_per_spe_f1[spe] = {
                'known': {'value': -1},
                'novel': {'value': -1}
            }

            post_red_per_spe_auc[spe] = {'known': {'value': -1}}
            post_red_per_spe_precision[spe] = {'known': {'value': -1}}
            post_red_per_spe_recall[spe] = {'known': {'value': -1}}
            post_red_per_spe_f1[spe] = {'known': {'value': -1}}

            test_post_red_per_spe_auc[spe] = {'known': {'value': -1}}
            test_post_red_per_spe_precision[spe] = {'known': {'value': -1}}
            test_post_red_per_spe_recall[spe] = {'known': {'value': -1}}
            test_post_red_per_spe_f1[spe] = {'known': {'value': -1}}

        # ** Kovel
        try:
            pre_red_precision, pre_red_rec, pre_red_f1, _ = metrics.precision_recall_fscore_support(
                all_grd_truth_spe_presence[spe].iloc[:total_pre_red_btn], 
                all_pred_spe_presence_yn_pre_retrain[spe].iloc[:total_pre_red_btn],
                average='binary',
                zero_division=0.0
            )
            post_red_precision_novel, post_red_rec_novel, post_red_f1_novel = interpolated_prec_rec_f1(
                y_true=all_grd_truth_spe_presence[spe].iloc[idx_novel_post_red], 
                y_pred_pre_retrain=all_pred_spe_presence_yn_pre_retrain[spe].iloc[idx_novel_post_red], 
                y_pred_post_retrain=all_pred_spe_presence_yn_post_retrain[spe].iloc[idx_novel_post_red]
            )
            test_post_red_precision_novel, test_post_red_rec_novel, test_post_red_f1_novel, _ = metrics.precision_recall_fscore_support(
                all_grd_truth_spe_presence[spe].iloc[idx_novel_test_post_red], 
                all_pred_spe_presence_yn_post_retrain[spe].iloc[idx_novel_test_post_red],
                average='binary',
                zero_division=0.0
            )

            
            # Precision
            pre_red_per_spe_precision[spe]['novel'] = {
                'value': pre_red_precision if pre_red_precision != 0.0 else -1,
                'ci': -1
            }
            post_red_per_spe_precision[spe]['novel'] = {
                'value': post_red_precision_novel if post_red_precision_novel != 0.0 else -1,
                'ci': -1
            }
            test_post_red_per_spe_precision[spe]['novel'] = {
                'value': test_post_red_precision_novel if test_post_red_precision_novel != 0.0 else -1,
                'ci': -1
            }

            # Recall
            pre_red_per_spe_recall[spe]['novel'] = {
                'value': pre_red_rec if pre_red_rec != 0.0 else -1,
                'ci': -1
            }
            post_red_per_spe_recall[spe]['novel'] = {
                'value': post_red_rec_novel if post_red_rec_novel != 0.0 else -1,
                'ci': -1
            }
            test_post_red_per_spe_recall[spe]['novel'] = {
                'value': test_post_red_rec_novel if test_post_red_rec_novel != 0.0 else -1,
                'ci': -1
            }

            # F1 score
            pre_red_per_spe_f1[spe]['novel'] = {
                'value': pre_red_f1 if pre_red_f1 != 0.0 else -1,
                'ci': -1
            }
            post_red_per_spe_f1[spe]['novel'] = {
                'value': post_red_f1_novel if post_red_f1_novel != 0.0 else -1,
                'ci': -1
            }
            test_post_red_per_spe_f1[spe]['novel'] = {
                'value': test_post_red_f1_novel if test_post_red_f1_novel != 0.0 else -1,
                'ci': -1
            }

        except Exception as ex:
            print('**** Exception when computing species presence metrics:', ex)
            post_red_per_spe_auc[spe]['novel'] = {'value': -1}
            post_red_per_spe_precision[spe]['novel'] = {'value': -1}
            post_red_per_spe_recall[spe]['novel'] = {'value': -1}
            post_red_per_spe_f1[spe]['novel'] = {'value': -1}

            test_post_red_per_spe_auc[spe]['novel'] = {'value': -1}
            test_post_red_per_spe_precision[spe]['novel'] = {'value': -1}
            test_post_red_per_spe_recall[spe]['novel'] = {'value': -1}
            test_post_red_per_spe_f1[spe]['novel'] = {'value': -1}


    # *****************************  SPECIES COUNT PERFORMANCE  ********************************
    '''
    # Save predicted counts and ground truth for debugging
    if output_dir is not None:
        tmp_dir = os.path.join(output_dir, 'temp_csv_files')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir, exist_ok=True)

        all_pred_spe_counts.to_csv(os.path.join(tmp_dir, f'{test_id}_pred_spe_count.csv'))
        all_grd_truth_spe_counts.to_csv(os.path.join(tmp_dir, f'{test_id}_grd_trth_spe_count.csv'))
    '''

    # *** Known
    num_pre_red_img_w_spe = all_grd_truth_spe_counts.iloc[:total_pre_red_btn].astype(bool).sum(axis=0)
    num_post_red_img_w_spe_known = all_grd_truth_spe_counts.iloc[idx_known_post_red].astype(bool).sum(axis=0)
    num_test_post_red_img_w_spe_known = all_grd_truth_spe_counts.iloc[idx_known_test_post_red].astype(bool).sum(axis=0)

    for spe in all_grd_truth_spe_counts.columns:
        if spe not in species_id2name_mapping.values():
            pre_red_abs_err_count[spe] = {
                'known': {'value': -1},
                'novel': {'value': -1}
            }

            post_red_abs_err_count[spe] = {'known': {'value': -1}}
            test_post_red_abs_err_count[spe] = {'known': {'value': -1}}
            continue
        
        # ---------->>  Absolute error  <<----------
        if num_pre_red_img_w_spe[spe] > 0:
            pre_red_cnt_abs_err = species_count_error(
                all_grd_truth_spe_counts[spe].iloc[:total_pre_red_btn], 
                all_pred_spe_counts_pre_retrain[spe].iloc[:total_pre_red_btn], 
                metric='AE'
            )

            if estimate_ci:
                pre_red_count_ci = boostrap_conf_interval(
                    y_true=all_grd_truth_spe_counts[spe].iloc[:total_pre_red_btn].to_numpy(), 
                    y_pred=all_pred_spe_counts_pre_retrain[spe].iloc[:total_pre_red_btn].to_numpy(), 
                    metric_name='count_err', 
                    n_samples=nbr_samples_conf_int
                )

            pre_red_abs_err_count[spe] = {
                'known': {
                    'value': pre_red_cnt_abs_err,
                    'ci': pre_red_count_ci
                }
            }
        else:
            pre_red_abs_err_count[spe] = {'value': -1}

        # ** post red button absolute count error
        if num_post_red_img_w_spe_known[spe] > 0:
            post_red_cnt_abs_err_known = interpolated_species_count_error(
                all_grd_truth_spe_counts[spe].iloc[idx_known_post_red], 
                all_pred_spe_counts_pre_retrain[spe].iloc[idx_known_post_red], 
                all_pred_spe_counts_post_retrain[spe].iloc[idx_known_post_red]
            )

            post_red_abs_err_count[spe] = {
                'known': {
                    'value': post_red_cnt_abs_err_known,
                    'ci': -1
                }
            }
        else:
            post_red_abs_err_count[spe] = {
                'known': {'value': -1}
            }

        # ** post red button test absolute count error
        if num_test_post_red_img_w_spe_known[spe] > 0:
            test_post_red_cnt_abs_err_known = species_count_error(
                all_grd_truth_spe_counts[spe].iloc[idx_known_test_post_red], 
                all_pred_spe_counts_post_retrain[spe].iloc[idx_known_test_post_red], 
                metric='AE'
            )

            if estimate_ci:
                test_post_red_count_ci_known = boostrap_conf_interval(
                    y_true=all_grd_truth_spe_counts[spe].iloc[idx_known_test_post_red].to_numpy(), 
                    y_pred=all_pred_spe_counts_post_retrain[spe].iloc[idx_known_test_post_red].to_numpy(), 
                    metric_name='count_err', 
                    n_samples=nbr_samples_conf_int
                )

            test_post_red_abs_err_count[spe] = {
                'known': {
                    'value': test_post_red_cnt_abs_err_known,
                    'ci': test_post_red_count_ci_known if estimate_ci else -1
                }
            }
        else:
            test_post_red_abs_err_count[spe] = {
                'known': {'value': -1}
            }
    
    # *** Novel
    # num_pre_red_img_w_spe = all_grd_truth_spe_counts.iloc[:total_pre_red_btn].astype(bool).sum(axis=0)
    num_post_red_img_w_spe_novel = all_grd_truth_spe_counts.iloc[idx_novel_post_red].astype(bool).sum(axis=0)
    num_test_post_red_img_w_spe_novel = all_grd_truth_spe_counts.iloc[idx_novel_test_post_red].astype(bool).sum(axis=0)

    for spe in all_grd_truth_spe_counts.columns:
        if spe not in species_id2name_mapping.values():
            post_red_abs_err_count[spe]['novel'] = {'value': -1}
            test_post_red_abs_err_count[spe]['novel'] = {'value': -1}
            continue
        
        # ---------->>  Absolute error  <<----------
        pre_red_abs_err_count[spe]['novel'] = {'value': -1}

        # ** post red button absolute count error
        if num_post_red_img_w_spe_novel[spe] > 0:
            post_red_cnt_abs_err_novel = interpolated_species_count_error(
                all_grd_truth_spe_counts[spe].iloc[idx_novel_post_red], 
                all_pred_spe_counts_pre_retrain[spe].iloc[idx_novel_post_red], 
                all_pred_spe_counts_post_retrain[spe].iloc[idx_novel_post_red]
            )

            post_red_abs_err_count[spe]['novel'] = {
                'value': post_red_cnt_abs_err_novel,
                'ci': -1
            }
        else:
            post_red_abs_err_count[spe]['novel'] = {'value': -1}

        # ** post red button test absolute count error
        if num_test_post_red_img_w_spe_novel[spe] > 0:
            test_post_red_cnt_abs_err_novel = species_count_error(
                all_grd_truth_spe_counts[spe].iloc[idx_novel_test_post_red], 
                all_pred_spe_counts_post_retrain[spe].iloc[idx_novel_test_post_red], 
                metric='AE'
            )

            if estimate_ci:
                test_post_red_count_ci_novel = boostrap_conf_interval(
                    y_true=all_grd_truth_spe_counts[spe].iloc[idx_novel_test_post_red].to_numpy(), 
                    y_pred=all_pred_spe_counts_post_retrain[spe].iloc[idx_novel_test_post_red].to_numpy(), 
                    metric_name='count_err', 
                    n_samples=nbr_samples_conf_int
                )

            test_post_red_abs_err_count[spe]['novel'] = {
                'value': test_post_red_cnt_abs_err_novel,
                'ci': test_post_red_count_ci_novel if estimate_ci else -1
            }
        else:
            test_post_red_abs_err_count[spe]['novel'] = {'value': -1}


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
        'pre_red_btn': pre_red_abs_err_count,
        'post_red_btn': post_red_abs_err_count,
        'test_post_red_btn': test_post_red_abs_err_count
    }
    all_performances['Species_counts'][test_id] = counts

    # **********************************
    # **** Aggregate species counts ****
    # ----------------------------------
    # ** Mean Absolute Error **
    # -----------------------------------
    # ** Known
    pre_red_avg_abs_err = species_count_error(
        all_grd_truth_spe_counts.iloc[:total_pre_red_btn], 
        all_pred_spe_counts_pre_retrain.iloc[:total_pre_red_btn], 
        metric='MAE'
    )
    post_red_avg_abs_err_known = interpolated_species_count_error(
        all_grd_truth_spe_counts.iloc[idx_known_post_red], 
        all_pred_spe_counts_pre_retrain.iloc[idx_known_post_red], 
        all_pred_spe_counts_post_retrain.iloc[idx_known_post_red]
    )

    test_post_red_avg_abs_err_known = species_count_error(
        all_grd_truth_spe_counts.iloc[idx_known_test_post_red], 
        all_pred_spe_counts_post_retrain.iloc[idx_known_test_post_red], 
        metric='MAE'
    )

    pre_red_avg_abs_err_count_ci, _ = boostrap_conf_interval(
        y_true=all_grd_truth_spe_counts.iloc[:total_pre_red_btn], 
        y_pred=all_pred_spe_counts_pre_retrain.iloc[:total_pre_red_btn], 
        metric_name='avg_count_err', 
        n_samples=nbr_samples_conf_int
    )
    post_red_avg_abs_err_count_ci_known = -1
    test_post_red_avg_abs_err_count_ci_known, _ = boostrap_conf_interval(
        y_true=all_grd_truth_spe_counts.iloc[idx_known_test_post_red], 
        y_pred=all_pred_spe_counts_post_retrain.iloc[idx_known_test_post_red], 
        metric_name='avg_count_err', 
        n_samples=nbr_samples_conf_int
    )

    # ** Novel
    post_red_avg_abs_err_novel = interpolated_species_count_error(
        all_grd_truth_spe_counts.iloc[idx_novel_post_red], 
        all_pred_spe_counts_pre_retrain.iloc[idx_novel_post_red], 
        all_pred_spe_counts_post_retrain.iloc[idx_novel_post_red]
    )

    test_post_red_avg_abs_err_novel = species_count_error(
        all_grd_truth_spe_counts.iloc[idx_novel_test_post_red], 
        all_pred_spe_counts_post_retrain.iloc[idx_novel_test_post_red], 
        metric='MAE'
    )

    post_red_avg_abs_err_count_ci_novel = -1
    test_post_red_avg_abs_err_count_ci_novel, _ = boostrap_conf_interval(
        y_true=all_grd_truth_spe_counts.iloc[idx_novel_test_post_red], 
        y_pred=all_pred_spe_counts_post_retrain.iloc[idx_novel_test_post_red], 
        metric_name='avg_count_err', 
        n_samples=nbr_samples_conf_int
    )

    agg_species_counts = {
        'pre_red_btn':{
            'known': {
                'value': round(pre_red_avg_abs_err,2),
                'ci': pre_red_avg_abs_err_count_ci
            },
            'novel': {
                'value': -1
            }
        },
        'post_red_btn':{
            'known': {
                'value': round(post_red_avg_abs_err_known,2),
                'ci': post_red_avg_abs_err_count_ci_known
            },
            'novel': {
                'value': round(post_red_avg_abs_err_novel,2),
                'ci': post_red_avg_abs_err_count_ci_novel
            }
        },
        'test_post_red_btn':{
            'known': {
                'value': round(test_post_red_avg_abs_err_known,2),
                'ci': test_post_red_avg_abs_err_count_ci_known
            },
            'novel': {
                'value': round(test_post_red_avg_abs_err_novel,2),
                'ci': test_post_red_avg_abs_err_count_ci_novel
            }
        }
    }
    all_performances['Aggregate_species_counts'][test_id] = agg_species_counts


    # Per species presence
    species_presence = {
        'pre_red_btn':{
            'auc': {k: {'value': -1} for k in pre_red_per_spe_precision.keys()},
            'precision': pre_red_per_spe_precision,
            'recall': pre_red_per_spe_recall,
            'f1_score': pre_red_per_spe_f1
        },
        'post_red_btn':{
            'auc': {k: {'value': -1} for k in post_red_per_spe_precision.keys()},
            'precision': post_red_per_spe_precision,
            'recall': post_red_per_spe_recall,
            'f1_score': post_red_per_spe_f1
        },
        'test_post_red_btn':{
            'auc': {k: {'value': -1} for k in test_post_red_per_spe_precision.keys()},
            'precision': test_post_red_per_spe_precision,
            'recall': test_post_red_per_spe_recall,
            'f1_score': test_post_red_per_spe_f1
        }
    }
    all_performances['Per_species_presence'][test_id] = species_presence    

    # Aggregate species presence
    pre_red_per_spe_auc_arr = -1
    pre_red_per_spe_precision_arr = np.array([pre_red_per_spe_precision[spe]['known']['value'] for spe in pre_red_per_spe_precision])
    pre_red_per_spe_recall_arr = np.array([pre_red_per_spe_recall[spe]['known']['value'] for spe in pre_red_per_spe_recall])
    pre_red_per_spe_f1_arr = np.array([pre_red_per_spe_f1[spe]['known']['value'] for spe in pre_red_per_spe_f1])

    pre_red_spe_avg_auc = -1
    pre_red_spe_avg_pre = round(np.mean(pre_red_per_spe_precision_arr[pre_red_per_spe_precision_arr >= 0]), 3)
    pre_red_spe_avg_rec = round(np.mean(pre_red_per_spe_recall_arr[pre_red_per_spe_recall_arr >= 0]), 3)
    pre_red_spe_avg_f1 = round(np.mean(pre_red_per_spe_f1_arr[pre_red_per_spe_f1_arr >= 0]), 3)

    pre_red_spe_avg_auc_ci, _ = boostrap_conf_interval(
        y_true=all_grd_truth_spe_presence.iloc[:total_pre_red_btn], 
        y_pred=all_pred_spe_presence_pre_retrain.iloc[:total_pre_red_btn], 
        metric_name='avg_auc', 
        n_samples=nbr_samples_conf_int
    )
    pre_red_spe_avg_pre_rec_f1_ci, _ = boostrap_conf_interval(
        y_true=all_grd_truth_spe_presence.iloc[:total_pre_red_btn], 
        y_pred=all_pred_spe_presence_yn_pre_retrain.iloc[:total_pre_red_btn], 
        metric_name='avg_pre/rec/f1', 
        n_samples=nbr_samples_conf_int
    )

    if total_pre_red_btn < start_test_phase:
        # *** Known ***
        post_red_per_spe_auc_arr_known = -1
        post_red_per_spe_precision_arr_known = np.array([post_red_per_spe_precision[spe]['known']['value'] for spe in post_red_per_spe_precision])
        post_red_per_spe_recall_arr_known = np.array([post_red_per_spe_recall[spe]['known']['value'] for spe in post_red_per_spe_recall])
        post_red_per_spe_f1_arr_known = np.array([post_red_per_spe_f1[spe]['known']['value'] for spe in post_red_per_spe_f1])
        
        post_red_spe_avg_auc_known = -1

        post_red_spe_avg_pre_known = -1
        if any(post_red_per_spe_precision_arr_known >= 0):
            post_red_spe_avg_pre_known = round(np.mean(post_red_per_spe_precision_arr_known[post_red_per_spe_precision_arr_known >= 0]), 3)

        post_red_spe_avg_rec_known = -1
        if any(post_red_per_spe_recall_arr_known >= 0):
            post_red_spe_avg_rec_known = round(np.mean(post_red_per_spe_recall_arr_known[post_red_per_spe_recall_arr_known >= 0]), 3)

        post_red_spe_avg_f1_known = -1
        if any(post_red_per_spe_f1_arr_known >= 0):
            post_red_spe_avg_f1_known = round(np.mean(post_red_per_spe_f1_arr_known[post_red_per_spe_f1_arr_known >= 0]), 3)

        post_red_spe_avg_auc_ci_known = -1
        post_red_spe_avg_pre_rec_f1_ci_known = -1

        # test phase
        test_post_red_per_spe_auc_arr_known = -1
        test_post_red_per_spe_precision_arr_known = np.array([test_post_red_per_spe_precision[spe]['known']['value'] for spe in test_post_red_per_spe_precision])
        test_post_red_per_spe_recall_arr_known = np.array([test_post_red_per_spe_recall[spe]['known']['value'] for spe in test_post_red_per_spe_recall])
        test_post_red_per_spe_f1_arr_known = np.array([test_post_red_per_spe_f1[spe]['known']['value'] for spe in test_post_red_per_spe_f1])

        test_post_red_spe_avg_auc_known = -1

        test_post_red_spe_avg_pre_known = -1
        if any(test_post_red_per_spe_precision_arr_known >= 0):
            test_post_red_spe_avg_pre_known = round(np.mean(test_post_red_per_spe_precision_arr_known[test_post_red_per_spe_precision_arr_known >= 0]), 3)

        test_post_red_spe_avg_rec_known = -1
        if any(test_post_red_per_spe_recall_arr_known >= 0):
            test_post_red_spe_avg_rec_known = round(np.mean(test_post_red_per_spe_recall_arr_known[test_post_red_per_spe_recall_arr_known >= 0]), 3)

        test_post_red_spe_avg_f1_known = -1
        if any(test_post_red_per_spe_f1_arr_known >= 0):
            test_post_red_spe_avg_f1_known = round(np.mean(test_post_red_per_spe_f1_arr_known[test_post_red_per_spe_f1_arr_known >= 0]), 3)

        test_post_red_spe_avg_auc_ci_known = -1
        test_post_red_spe_avg_pre_rec_f1_ci_known, _ = boostrap_conf_interval(
            y_true=all_grd_truth_spe_presence.iloc[idx_known_test_post_red], 
            y_pred=all_pred_spe_presence_yn_post_retrain.iloc[idx_known_test_post_red], 
            metric_name='avg_pre/rec/f1', 
            n_samples=nbr_samples_conf_int
        )

        # *** Novel ***
        post_red_per_spe_auc_arr_novel = -1
        post_red_per_spe_precision_arr_novel = np.array([post_red_per_spe_precision[spe]['novel']['value'] for spe in post_red_per_spe_precision])
        post_red_per_spe_recall_arr_novel = np.array([post_red_per_spe_recall[spe]['novel']['value'] for spe in post_red_per_spe_recall])
        post_red_per_spe_f1_arr_novel = np.array([post_red_per_spe_f1[spe]['novel']['value'] for spe in post_red_per_spe_f1])
        
        post_red_spe_avg_auc_novel = -1

        post_red_spe_avg_pre_novel = -1
        if any(post_red_per_spe_precision_arr_novel >= 0):
            post_red_spe_avg_pre_novel = round(np.mean(post_red_per_spe_precision_arr_novel[post_red_per_spe_precision_arr_novel >= 0]), 3)

        post_red_spe_avg_rec_novel = -1
        if any(post_red_per_spe_recall_arr_novel >= 0):
            post_red_spe_avg_rec_novel = round(np.mean(post_red_per_spe_recall_arr_novel[post_red_per_spe_recall_arr_novel >= 0]), 3)

        post_red_spe_avg_f1_novel = -1
        if any(post_red_per_spe_f1_arr_novel >= 0):
            post_red_spe_avg_f1_novel = round(np.mean(post_red_per_spe_f1_arr_novel[post_red_per_spe_f1_arr_novel >= 0]), 3)

        post_red_spe_avg_auc_ci_novel = -1
        post_red_spe_avg_pre_rec_f1_ci_novel = -1

        # test phase
        test_post_red_per_spe_auc_arr_novel = -1
        test_post_red_per_spe_precision_arr_novel = np.array([test_post_red_per_spe_precision[spe]['novel']['value'] for spe in test_post_red_per_spe_precision])
        test_post_red_per_spe_recall_arr_novel = np.array([test_post_red_per_spe_recall[spe]['novel']['value'] for spe in test_post_red_per_spe_recall])
        test_post_red_per_spe_f1_arr_novel = np.array([test_post_red_per_spe_f1[spe]['novel']['value'] for spe in test_post_red_per_spe_f1])

        test_post_red_spe_avg_auc_novel = -1

        test_post_red_spe_avg_pre_novel = -1
        if any(test_post_red_per_spe_precision_arr_novel >= 0):
            test_post_red_spe_avg_pre_novel = round(np.mean(test_post_red_per_spe_precision_arr_novel[test_post_red_per_spe_precision_arr_novel >= 0]), 3)

        test_post_red_spe_avg_rec_novel = -1
        if any(test_post_red_per_spe_recall_arr_novel >= 0):
            test_post_red_spe_avg_rec_novel = round(np.mean(test_post_red_per_spe_recall_arr_novel[test_post_red_per_spe_recall_arr_novel >= 0]), 3)

        test_post_red_spe_avg_f1_novel = -1
        if any(test_post_red_per_spe_f1_arr_novel >= 0):
            test_post_red_spe_avg_f1_novel = round(np.mean(test_post_red_per_spe_f1_arr_novel[test_post_red_per_spe_f1_arr_novel >= 0]), 3)

        test_post_red_spe_avg_auc_ci_novel = -1
        test_post_red_spe_avg_pre_rec_f1_ci_novel, _ = boostrap_conf_interval(
            y_true=all_grd_truth_spe_presence.iloc[idx_novel_test_post_red], 
            y_pred=all_pred_spe_presence_yn_post_retrain.iloc[idx_novel_test_post_red], 
            metric_name='avg_pre/rec/f1', 
            n_samples=nbr_samples_conf_int
        )

    else:
        # the trial does not have any novelty
        post_red_spe_avg_auc_known, post_red_spe_avg_auc_novel = -1, -1
        post_red_spe_avg_pre_known, post_red_spe_avg_pre_novel = -1, -1
        post_red_spe_avg_rec_known, post_red_spe_avg_rec_novel = -1, -1
        post_red_spe_avg_f1_known, post_red_spe_avg_f1_novel = -1, -1

        post_red_spe_avg_auc_ci_known, post_red_spe_avg_auc_ci_novel = -1, -1
        post_red_spe_avg_pre_rec_f1_ci_known = {
            'avg_precision': -1,
            'avg_recall': -1,
            'avg_f1_score': -1
        }
        post_red_spe_avg_pre_rec_f1_ci_novel = {
            'avg_precision': -1,
            'avg_recall': -1,
            'avg_f1_score': -1
        }

        test_post_red_spe_avg_auc_known, test_post_red_spe_avg_auc_novel = -1, -1
        test_post_red_spe_avg_pre_known, test_post_red_spe_avg_pre_novel = -1, -1
        test_post_red_spe_avg_rec_known, test_post_red_spe_avg_rec_novel = -1, -1
        test_post_red_spe_avg_f1_known, test_post_red_spe_avg_f1_novel = -1, -1
        test_post_red_spe_avg_auc_ci_known, test_post_red_spe_avg_auc_ci_novel = -1, -1
        test_post_red_spe_avg_pre_rec_f1_ci_known = {
            'avg_precision': -1,
            'avg_recall': -1,
            'avg_f1_score': -1
        }
        test_post_red_spe_avg_pre_rec_f1_ci_novel = {
            'avg_precision': -1,
            'avg_recall': -1,
            'avg_f1_score': -1
        }

    agg_species_presence = {
        'pre_red_btn':{
            'avg_auc': {
                'known': {
                    'value': pre_red_spe_avg_auc,
                    'ci': pre_red_spe_avg_auc_ci
                },
                'novel': {
                    'value': -1
                }
            },
            'avg_precision': {
                'known': {
                    'value': pre_red_spe_avg_pre,
                    'ci': pre_red_spe_avg_pre_rec_f1_ci['avg_precision']
                },
                'novel': {
                    'value': -1
                }
            },
            'avg_recall': {
                'known': {
                    'value': pre_red_spe_avg_rec,
                    'ci': pre_red_spe_avg_pre_rec_f1_ci['avg_recall']
                },
                'novel': {
                    'value': -1
                }
            },
            'avg_f1_score': {
                'known': {
                    'value': pre_red_spe_avg_f1,
                    'ci': pre_red_spe_avg_pre_rec_f1_ci['avg_f1_score']
                },
                'novel': {
                    'value': -1
                }
            }
        },
        'post_red_btn':{
            'avg_auc': {
                'known': {
                    'value': post_red_spe_avg_auc_known,
                    'ci': -1
                },
                'novel': {
                    'value': post_red_spe_avg_auc_novel,
                    'ci': -1
                }
            },
            'avg_precision': {
                'known': {
                    'value': post_red_spe_avg_pre_known,
                    'ci': -1
                },
                'novel': {
                    'value': post_red_spe_avg_pre_novel,
                    'ci': -1
                }
            },
            'avg_recall': {
                'known': {
                    'value': post_red_spe_avg_rec_known,
                    'ci': -1
                },
                'novel': {
                    'value': post_red_spe_avg_rec_novel,
                    'ci': -1
                }
            },
            'avg_f1_score': {
                'known': {
                    'value': post_red_spe_avg_f1_known,
                    'ci': -1
                },
                'novel': {
                    'value': post_red_spe_avg_f1_novel,
                    'ci': -1
                }
            }
        },
        'test_post_red_btn':{
            'avg_auc': {
                'known': {
                    'value': test_post_red_spe_avg_auc_known,
                    'ci': -1
                },
                'novel': {
                    'value': test_post_red_spe_avg_auc_novel,
                    'ci': -1
                }
            },
            'avg_precision': {
                'known': {
                    'value': test_post_red_spe_avg_pre_known,
                    'ci': -1
                },
                'novel': {
                    'value': test_post_red_spe_avg_pre_novel,
                    'ci': -1
                }
            },
            'avg_recall': {
                'known': {
                    'value': test_post_red_spe_avg_rec_known,
                    'ci': -1
                },
                'novel': {
                    'value': test_post_red_spe_avg_rec_novel,
                    'ci': -1
                }
            },
            'avg_f1_score': {
                'known': {
                    'value': test_post_red_spe_avg_f1_known,
                    'ci': -1
                },
                'novel': {
                    'value': test_post_red_spe_avg_f1_novel,
                    'ci': -1
                }
            }
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
        },
        'test_post_red_btn':{
            'auc': test_post_red_per_act_auc,
            'precision': test_post_red_per_act_precision,
            'recall': test_post_red_per_act_recall,
            'f1_score': test_post_red_per_act_f1
        }
    }
    all_performances['Per_activity_presence'][test_id] = activity_presence

    # Aggregate activity presence
    # >> Pre-novely phase <<
    pre_red_per_act_auc_arr = -1
    pre_red_per_act_precision_arr = np.array([pre_red_per_act_precision[act]['known']['value'] for act in pre_red_per_act_precision])
    pre_red_per_act_recall_arr = np.array([pre_red_per_act_recall[act]['known']['value'] for act in pre_red_per_act_recall])
    pre_red_per_act_f1_arr = np.array([pre_red_per_act_f1[act]['known']['value'] for act in pre_red_per_act_f1])

    pre_red_act_avg_auc = -1
    pre_red_act_avg_pre = round(np.mean(pre_red_per_act_precision_arr[pre_red_per_act_precision_arr >= 0]), 3)
    pre_red_act_avg_rec = round(np.mean(pre_red_per_act_recall_arr[pre_red_per_act_recall_arr >= 0]), 3)
    pre_red_act_avg_f1 = round(np.mean(pre_red_per_act_f1_arr[pre_red_per_act_f1_arr >= 0]), 3)

    pre_red_act_avg_auc_ci = -1
    pre_red_act_avg_pre_rec_f1_ci, _ = boostrap_conf_interval(
        y_true=all_grd_truth_act_presence.iloc[:total_pre_red_btn], 
        y_pred=all_pred_act_presence_yn_pre_retrain.iloc[:total_pre_red_btn], 
        metric_name='avg_pre/rec/f1', 
        n_samples=nbr_samples_conf_int
    )

    # >> Post-novely phase <<
    if total_pre_red_btn < start_test_phase:
        # -----------------
        try:
            # ****  Known  ****
            post_red_per_act_auc_arr_known = np.array([post_red_per_act_auc[act]['known']['value'] for act in post_red_per_act_auc])
            post_red_per_act_precision_arr_known = np.array([post_red_per_act_precision[act]['known']['value'] for act in post_red_per_act_precision])
            post_red_per_act_recall_arr_known = np.array([post_red_per_act_recall[act]['known']['value'] for act in post_red_per_act_recall])
            post_red_per_act_f1_arr_known = np.array([post_red_per_act_f1[act]['known']['value'] for act in post_red_per_act_f1])

            post_red_act_avg_auc_known = -1 
            if any(post_red_per_act_auc_arr_known >= 0): 
                post_red_act_avg_auc_known = round(np.mean(post_red_per_act_auc_arr_known[post_red_per_act_auc_arr_known >= 0]), 2)

            post_red_act_avg_pre_known = -1 
            if any(post_red_per_act_precision_arr_known >= 0): 
                post_red_act_avg_pre_known = round(np.mean(post_red_per_act_precision_arr_known[post_red_per_act_precision_arr_known >= 0]), 2)

            post_red_act_avg_rec_known = -1 
            if any(post_red_per_act_recall_arr_known >= 0): 
                post_red_act_avg_rec_known = round(np.mean(post_red_per_act_recall_arr_known[post_red_per_act_recall_arr_known >= 0]), 2)

            post_red_act_avg_f1_known = -1 
            if any(post_red_per_act_f1_arr_known >= 0): 
                post_red_act_avg_f1_known = round(np.mean(post_red_per_act_f1_arr_known[post_red_per_act_f1_arr_known >= 0]), 2)

            # post_red_act_avg_auc_known = round(np.mean(post_red_per_act_auc_arr_known[post_red_per_act_auc_arr_known >= 0]), 2)
            # post_red_act_avg_pre_known = round(np.mean(post_red_per_act_precision_arr_known[post_red_per_act_precision_arr_known >= 0]), 2)
            # post_red_act_avg_rec_known = round(np.mean(post_red_per_act_recall_arr_known[post_red_per_act_recall_arr_known >= 0]), 2)
            # post_red_act_avg_f1_known = round(np.mean(post_red_per_act_f1_arr_known[post_red_per_act_f1_arr_known >= 0]), 2)


            post_red_act_avg_auc_ci_known = -1
            post_red_act_avg_pre_rec_f1_ci_known = -1

            test_post_red_per_act_auc_arr_known = np.array([test_post_red_per_act_auc[act]['known']['value'] for act in test_post_red_per_act_auc])
            test_post_red_per_act_precision_arr_known = np.array([test_post_red_per_act_precision[act]['known']['value'] for act in test_post_red_per_act_precision])
            test_post_red_per_act_recall_arr_known = np.array([test_post_red_per_act_recall[act]['known']['value'] for act in test_post_red_per_act_recall])
            test_post_red_per_act_f1_arr_known = np.array([test_post_red_per_act_f1[act]['known']['value'] for act in test_post_red_per_act_f1])

            test_post_red_act_avg_auc_known = -1 
            if any(test_post_red_per_act_auc_arr_known >= 0): 
                test_post_red_act_avg_auc_known = round(np.mean(test_post_red_per_act_auc_arr_known[test_post_red_per_act_auc_arr_known >= 0]), 2)

            test_post_red_act_avg_pre_known = -1 
            if any(test_post_red_per_act_precision_arr_known >= 0): 
                test_post_red_act_avg_pre_known = round(np.mean(test_post_red_per_act_precision_arr_known[test_post_red_per_act_precision_arr_known >= 0]), 2)

            test_post_red_act_avg_rec_known = -1 
            if any(test_post_red_per_act_recall_arr_known >= 0): 
                test_post_red_act_avg_rec_known = round(np.mean(test_post_red_per_act_recall_arr_known[test_post_red_per_act_recall_arr_known >= 0]), 2)

            test_post_red_act_avg_f1_known = -1 
            if any(test_post_red_per_act_f1_arr_known >= 0): 
                test_post_red_act_avg_f1_known = round(np.mean(test_post_red_per_act_f1_arr_known[test_post_red_per_act_f1_arr_known >= 0]), 2)

            test_post_red_act_avg_auc_ci_known, _ = boostrap_conf_interval(
                y_true=all_grd_truth_act_presence.iloc[idx_known_test_post_red], 
                y_pred=all_pred_act_presence_post_retrain.iloc[idx_known_test_post_red], 
                metric_name='avg_auc', 
                n_samples=nbr_samples_conf_int
            )
            test_post_red_act_avg_pre_rec_f1_ci_known, _ = boostrap_conf_interval(
                y_true=all_grd_truth_act_presence.iloc[idx_known_test_post_red], 
                y_pred=all_pred_act_presence_yn_post_retrain.iloc[idx_known_test_post_red], 
                metric_name='avg_pre/rec/f1', 
                n_samples=nbr_samples_conf_int
            )

            # -----------------
            # ****  novel  ****
            post_red_per_act_auc_arr_novel = np.array([post_red_per_act_auc[act]['novel']['value'] for act in post_red_per_act_auc])
            post_red_per_act_precision_arr_novel = np.array([post_red_per_act_precision[act]['novel']['value'] for act in post_red_per_act_precision])
            post_red_per_act_recall_arr_novel = np.array([post_red_per_act_recall[act]['novel']['value'] for act in post_red_per_act_recall])
            post_red_per_act_f1_arr_novel = np.array([post_red_per_act_f1[act]['novel']['value'] for act in post_red_per_act_f1])

            post_red_act_avg_auc_novel = -1 
            if any(post_red_per_act_auc_arr_novel >= 0): 
                post_red_act_avg_auc_novel = round(np.mean(post_red_per_act_auc_arr_novel[post_red_per_act_auc_arr_novel >= 0]), 2)

            post_red_act_avg_pre_novel = -1 
            if any(post_red_per_act_precision_arr_novel >= 0): 
                post_red_act_avg_pre_novel = round(np.mean(post_red_per_act_precision_arr_novel[post_red_per_act_precision_arr_novel >= 0]), 2)

            post_red_act_avg_rec_novel = -1 
            if any(post_red_per_act_recall_arr_novel >= 0): 
                post_red_act_avg_rec_novel = round(np.mean(post_red_per_act_recall_arr_novel[post_red_per_act_recall_arr_novel >= 0]), 2)

            post_red_act_avg_f1_novel = -1 
            if any(post_red_per_act_f1_arr_novel >= 0): 
                post_red_act_avg_f1_novel = round(np.mean(post_red_per_act_f1_arr_novel[post_red_per_act_f1_arr_novel >= 0]), 2)
            
            # post_red_act_avg_auc_novel = round(np.mean(post_red_per_act_auc_arr_novel[post_red_per_act_auc_arr_novel >= 0]), 2)
            # post_red_act_avg_pre_novel = round(np.mean(post_red_per_act_precision_arr_novel[post_red_per_act_precision_arr_novel >= 0]), 2)
            # post_red_act_avg_rec_novel = round(np.mean(post_red_per_act_recall_arr_novel[post_red_per_act_recall_arr_novel >= 0]), 2)
            # post_red_act_avg_f1_novel = round(np.mean(post_red_per_act_f1_arr_novel[post_red_per_act_f1_arr_novel >= 0]), 2)

            post_red_act_avg_auc_ci_novel = -1
            post_red_act_avg_pre_rec_f1_ci_novel = -1

            test_post_red_per_act_auc_arr_novel = np.array([test_post_red_per_act_auc[act]['novel']['value'] for act in test_post_red_per_act_auc])
            test_post_red_per_act_precision_arr_novel = np.array([test_post_red_per_act_precision[act]['novel']['value'] for act in test_post_red_per_act_precision])
            test_post_red_per_act_recall_arr_novel = np.array([test_post_red_per_act_recall[act]['novel']['value'] for act in test_post_red_per_act_recall])
            test_post_red_per_act_f1_arr_novel = np.array([test_post_red_per_act_f1[act]['novel']['value'] for act in test_post_red_per_act_f1])

            test_post_red_act_avg_auc_novel = -1 
            if any(test_post_red_per_act_auc_arr_novel >= 0): 
                test_post_red_act_avg_auc_novel = round(np.mean(test_post_red_per_act_auc_arr_novel[test_post_red_per_act_auc_arr_novel >= 0]), 2)

            test_post_red_act_avg_pre_novel = -1 
            if any(test_post_red_per_act_precision_arr_novel >= 0): 
                test_post_red_act_avg_pre_novel = round(np.mean(test_post_red_per_act_precision_arr_novel[test_post_red_per_act_precision_arr_novel >= 0]), 2)

            test_post_red_act_avg_rec_novel = -1 
            if any(test_post_red_per_act_recall_arr_novel >= 0): 
                test_post_red_act_avg_rec_novel = round(np.mean(test_post_red_per_act_recall_arr_novel[test_post_red_per_act_recall_arr_novel >= 0]), 2)

            test_post_red_act_avg_f1_novel = -1 
            if any(test_post_red_per_act_f1_arr_novel >= 0): 
                test_post_red_act_avg_f1_novel = round(np.mean(test_post_red_per_act_f1_arr_novel[test_post_red_per_act_f1_arr_novel >= 0]), 2)

            test_post_red_act_avg_auc_ci_novel, _ = boostrap_conf_interval(
                y_true=all_grd_truth_act_presence.iloc[idx_novel_test_post_red], 
                y_pred=all_pred_act_presence_post_retrain.iloc[idx_novel_test_post_red], 
                metric_name='avg_auc', 
                n_samples=nbr_samples_conf_int
            )
            test_post_red_act_avg_pre_rec_f1_ci_novel, _ = boostrap_conf_interval(
                y_true=all_grd_truth_act_presence.iloc[idx_novel_test_post_red], 
                y_pred=all_pred_act_presence_yn_post_retrain.iloc[idx_novel_test_post_red], 
                metric_name='avg_pre/rec/f1', 
                n_samples=nbr_samples_conf_int
            )
        except Exception as ex:
            # the trial does not have any novelty
            post_red_act_avg_auc_known, post_red_act_avg_auc_novel = -1, -1
            post_red_act_avg_pre_known, post_red_act_avg_pre_novel = -1, -1
            post_red_act_avg_rec_known, post_red_act_avg_rec_novel = -1, -1
            post_red_act_avg_f1_known, post_red_act_avg_f1_novel = -1, -1

            post_red_act_avg_auc_ci_known, post_red_act_avg_auc_ci_novel = -1, -1
            post_red_act_avg_pre_rec_f1_ci_known = {
                'avg_precision': -1,
                'avg_recall': -1,
                'avg_f1_score': -1
            }
            post_red_act_avg_pre_rec_f1_ci_novel = {
                'avg_precision': -1,
                'avg_recall': -1,
                'avg_f1_score': -1
            }

            test_post_red_act_avg_auc_known, test_post_red_act_avg_auc_novel = -1, -1
            test_post_red_act_avg_pre_known, test_post_red_act_avg_pre_novel = -1, -1
            test_post_red_act_avg_rec_known, test_post_red_act_avg_rec_novel = -1, -1
            test_post_red_act_avg_f1_known, test_post_red_act_avg_f1_novel = -1, -1

            test_post_red_act_avg_auc_ci_known, test_post_red_act_avg_auc_ci_novel = -1, -1
            test_post_red_act_avg_pre_rec_f1_ci_known = {
                'avg_precision': -1,
                'avg_recall': -1,
                'avg_f1_score': -1
            }
            test_post_red_act_avg_pre_rec_f1_ci_novel = {
                'avg_precision': -1,
                'avg_recall': -1,
                'avg_f1_score': -1
            }
    else:
        # the trial does not have any novelty
        post_red_act_avg_auc_known, post_red_act_avg_auc_novel = -1, -1
        post_red_act_avg_pre_known, post_red_act_avg_pre_novel = -1, -1
        post_red_act_avg_rec_known, post_red_act_avg_rec_novel = -1, -1
        post_red_act_avg_f1_known, post_red_act_avg_f1_novel = -1, -1

        post_red_act_avg_auc_ci_known, post_red_act_avg_auc_ci_novel = -1, -1
        post_red_act_avg_pre_rec_f1_ci_known = {
            'avg_precision': -1,
            'avg_recall': -1,
            'avg_f1_score': -1
        }
        post_red_act_avg_pre_rec_f1_ci_novel = {
            'avg_precision': -1,
            'avg_recall': -1,
            'avg_f1_score': -1
        }

        test_post_red_act_avg_auc_known, test_post_red_act_avg_auc_novel = -1, -1
        test_post_red_act_avg_pre_known, test_post_red_act_avg_pre_novel = -1, -1
        test_post_red_act_avg_rec_known, test_post_red_act_avg_rec_novel = -1, -1
        test_post_red_act_avg_f1_known, test_post_red_act_avg_f1_novel = -1, -1

        test_post_red_act_avg_auc_ci_known, test_post_red_act_avg_auc_ci_novel = -1, -1
        test_post_red_act_avg_pre_rec_f1_ci_known = {
            'avg_precision': -1,
            'avg_recall': -1,
            'avg_f1_score': -1
        }
        test_post_red_act_avg_pre_rec_f1_ci_novel = {
            'avg_precision': -1,
            'avg_recall': -1,
            'avg_f1_score': -1
        }


    agg_activity_presence = {
        'pre_red_btn':{
            'avg_auc': {
                'known': {
                    'value': pre_red_act_avg_auc,
                    'ci': pre_red_act_avg_auc_ci
                },
                'novel': {
                    'value': -1
                }
            },
            'avg_precision': {
                'known': {
                    'value': pre_red_act_avg_pre,
                    'ci': pre_red_act_avg_pre_rec_f1_ci['avg_precision']
                },
                'novel': {
                    'value': -1
                }
            },
            'avg_recall': {
                'known': {
                    'value': pre_red_act_avg_rec,
                    'ci': pre_red_act_avg_pre_rec_f1_ci['avg_recall']
                },
                'novel': {
                    'value': -1
                }
            },
            'avg_f1_score': {
                'known': {
                    'value': pre_red_act_avg_f1,
                    'ci': pre_red_act_avg_pre_rec_f1_ci['avg_f1_score']
                },
                'novel': {
                    'value': -1
                }
            }
        },
        'post_red_btn':{
            'avg_auc': {
                'known': {
                    'value': post_red_act_avg_auc_known,
                    'ci': post_red_act_avg_auc_ci_known
                },
                'novel': {
                    'value': post_red_act_avg_auc_novel,
                    'ci': post_red_act_avg_auc_ci_novel
                }
            },
            'avg_precision': {
                'known': {
                    'value': post_red_act_avg_pre_known,
                    'ci': -1
                },
                'novel': {
                    'value': post_red_act_avg_pre_novel,
                    'ci': -1
                }
            },
            'avg_recall': {
                'known': {
                    'value': post_red_act_avg_rec_known,
                    'ci': -1
                },
                'novel': {
                    'value': post_red_act_avg_rec_novel,
                    'ci': -1
                }
            },
            'avg_f1_score': {
                'known': {
                    'value': post_red_act_avg_f1_known,
                    'ci': -1
                },
                'novel': {
                    'value': post_red_act_avg_f1_novel,
                    'ci': -1
                }
            }
        },
        'test_post_red_btn':{
            'avg_auc': {
                'known': {
                    'value': test_post_red_act_avg_auc_known,
                    'ci': -1
                },
                'novel': {
                    'value': test_post_red_act_avg_auc_novel,
                    'ci': -1
                }
            },
            'avg_precision': {
                'known': {
                    'value': test_post_red_act_avg_pre_known,
                    'ci': -1
                },
                'novel': {
                    'value': test_post_red_act_avg_pre_novel,
                    'ci': -1
                }
            },
            'avg_recall': {
                'known': {
                    'value': test_post_red_act_avg_rec_known,
                    'ci': -1
                },
                'novel': {
                    'value': test_post_red_act_avg_rec_novel,
                    'ci': -1
                }
            },
            'avg_f1_score': {
                'known': {
                    'value': test_post_red_act_avg_f1_known,
                    'ci': -1
                },
                'novel': {
                    'value': test_post_red_act_avg_f1_novel,
                    'ci': -1
                }
            }
        }
    }
    all_performances['Aggregate_activity_presence'][test_id] = agg_activity_presence


def score_tests(
    test_dir, sys_output_dir, bboxes_dir, session_id, class_file_reader, log_dir,
    save_symlinks, dataset_root, detection_threshold, spe_presence_threshold,
        act_presence_threshold
):   
    
    nbr_rounds = 300
    size_test_phase = 1000  # number of images at the end of the trials used for testing
    
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
        'Prediction_confidence': {},
        'Species_id2name': {},
        'Activity_id2name': {}
    }
    for test_id in test_ids:
        # if 'OND' in test_id and '100.000' not in test_id:
        if 'OND' in test_id and '100.000' not in test_id and '105.000' not in test_id:
        # if 'OND' in test_id:
            metadata = json.load(open(test_dir / f'{test_id}_metadata.json', 'r'))
            test_df = pd.read_csv(test_dir / f'{test_id}_single_df.csv')

            pkl_fname = os.path.join(bboxes_dir, test_id+'/'+test_id+'.pkl')
            print(pkl_fname)
            assert os.path.exists(pkl_fname)
            with (open(pkl_fname, "rb")) as pkl_file:
                boxes_pred_dict = pickle.load(pkl_file)

            
            # ***
            test_id = test_id[4:]

            detect_lines = []
            class_lines_pre_retrain, class_lines_post_retrain = [], []
            for round_ in range(nbr_rounds):
                if (sys_output_dir / f'{session_id}.{test_id}_{round_}_detection.csv').exists():
                    detect_lines.append(open(sys_output_dir / f'{session_id}.{test_id}_{round_}_detection.csv').read().splitlines())
                    class_lines_pre_retrain.append(open(sys_output_dir / f'{session_id}.{test_id}_{round_}_classification.csv').read().splitlines())
                    class_lines_post_retrain.append(open(sys_output_dir / f'{session_id}.{test_id}_{round_}_classification.csv').read().splitlines())

                else:
                    print(f'No results found for Test {session_id}.{test_id}_{round_}.')
            detect_lines = np.concatenate(detect_lines)

            class_lines_pre_retrain = np.concatenate(class_lines_pre_retrain)
            class_lines_post_retrain = np.concatenate(class_lines_post_retrain)
         
            '''
            if (sys_output_dir / f'{session_id}.{test_id}_detection.csv').exists():
                detect_lines = open(sys_output_dir / f'{session_id}.{test_id}_detection.csv').read().splitlines()
                class_lines_pre_retrain = open(sys_output_dir / f'{session_id}.{test_id}_classification.csv').read().splitlines()
                class_lines_post_retrain = open(sys_output_dir / f'{session_id}.{test_id}_classification.csv').read().splitlines()
            else:
                print(f'No results found for Test {session_id}.{test_id}_.')
            '''
           
            print('/n', detection_threshold, spe_presence_threshold, act_presence_threshold, '/n')
        
            with open(log_dir / f'{test_id}.log', 'w') as log:
                score_test(
                    test_id, metadata, test_df, detect_lines, class_lines_pre_retrain, class_lines_post_retrain, 
                    class_file_reader, log, all_performances, detection_threshold, spe_presence_threshold, 
                    act_presence_threshold, estimate_ci=True, nbr_samples_conf_int=300, len_test_phase=size_test_phase, output_dir=log_dir
                )       


    write_results_to_csv(all_performances, output_path=log_dir)


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
    print(session_ids)
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