"""
Takes a tests directory and a results directory.
Writes log files for each test and a summary scores file to an output logs directory.
"""
#########################################################################
# Copyright 2021-2022 by Raytheon BBN Technologies.  All Rights Reserved
#########################################################################


from pathlib import Path
from argparse import ArgumentParser
import json
import pandas as pd
from class_file_reader import ClassFileReader
# from stats import Stats, Instance
import shutil
import numpy as np
import sklearn.metrics as metrics
import ast

from write_results import write_results_to_log, print_confusion_matrices


def percent_string(num, denom=None):
    if denom is None:
        return f'{100 * num:6.2f}%'
    return f'{100 * num / denom:6.2f}%'


def species_count_error(grd_trth_vec, pred_vec, metric='MSE'):
    '''
    Computes the squared error, mean squared error, mean absolute error or 
    root mean squared error between the ground truth and the predicion
    '''
    assert metric in ('SE', 'RMSE', 'MSE', 'MAE')
    assert len(grd_trth_vec) == len(pred_vec)

    grd_trth_vec = np.array(grd_trth_vec).ravel()
    pred_vec = np.array(pred_vec).ravel()
    diff = grd_trth_vec - pred_vec
    sqrt_diff = np.power(diff, 2)
    if metric == 'SE':
        return sum(diff)
    if metric == 'MSE':
        return np.mean(sqrt_diff)
    elif metric == 'MAE':
        return np.mean(np.abs(diff))
    return np.sqrt(np.mean(sqrt_diff))

def score_test(
    test_id, metadata, test_df, detect_lines, class_lines, class_file_reader, 
    log, all_performances, stats, detection_threshold, presence_threshold=0.5, count_metric='MAE'
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

    species_cols = [x for x in class_lines[0].split(',') if 'species_' in x and '_presence' in x]
    total_nbr_species = len(species_cols)

    activity_cols = [x for x in class_lines[0].split(',') if 'activity_' in x]
    total_nbr_activities = len(activity_cols)

    pre_red_per_species_brier_score = np.zeros((1, total_nbr_species)).ravel()
    pre_red_per_species_presence_acc = np.zeros((1, total_nbr_species)).ravel()
    pre_red_per_species_auc = np.zeros((1, total_nbr_species)).ravel()

    pre_red_per_activity_brier_score = np.zeros((1, total_nbr_activities)).ravel()
    pre_red_per_activity_acc = np.zeros((1, total_nbr_activities)).ravel()
    pre_red_per_activity_auc = np.zeros((1, total_nbr_activities)).ravel()

    post_red_per_species_brier_score = np.zeros((1, total_nbr_species)).ravel()
    post_red_per_species_presence_acc = np.zeros((1, total_nbr_species)).ravel()
    post_red_per_species_auc = np.zeros((1, total_nbr_species)).ravel()

    post_red_per_activity_brier_score = np.zeros((1, total_nbr_activities)).ravel()
    post_red_per_activity_acc = np.zeros((1, total_nbr_activities)).ravel()
    post_red_per_activity_auc = np.zeros((1, total_nbr_activities)).ravel()

    pre_red_species_count_err1, pre_red_species_count_err2 = 0, 0
    post_red_species_count_err1, post_red_species_count_err2 = 0, 0

    total_pre_red_btn, total_post_red_btn = 0, 0

    all_grd_truth_spe_presence = np.zeros((len_test, total_nbr_species))
    all_pred_spe_presence = np.zeros((len_test, total_nbr_species))

    all_grd_truth_activity_presence = np.zeros((len_test, total_nbr_activities))
    all_pred_activity_presence = np.zeros((len_test, total_nbr_activities))

    species_id2name_mapping = {}
    activity_id2name_mapping = {}


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
        ground_truth_species_counts = np.zeros((1, total_nbr_species)).ravel()
        ground_truth_species_presence = np.zeros((1, total_nbr_species)).ravel()
        if not pd.isnull(test_tuple.agent1_id):
            ground_truth_species_counts[int(test_tuple.agent1_id)] = int(test_tuple.agent1_count)
            ground_truth_species_presence[int(test_tuple.agent1_id)] = 1

            if int(test_tuple.agent1_id) not in species_id2name_mapping:
                species_id2name_mapping[int(test_tuple.agent1_id)] = test_tuple.agent1_name
        else:
            # # this is an empty image
            # if '--' not in species_id2name_mapping:
            #     species_id2name_mapping['--'] = test_tuple.agent1_name
            if 0 not in species_id2name_mapping:
                species_id2name_mapping[0] = test_tuple.agent1_name

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

        all_grd_truth_activity_presence[pos, :] = ground_truth_activity_presence

        # vector of predicted species counts
        pred_species_counts, pred_species_presence, pred_activity_presence = class_file_reader.get_answers(
            class_line, total_nbr_species, total_nbr_activities
        )

        # Apply presence threshold to predicted species presence and predicted activity presence
        pred_species_presence_yn = [
            1 if pred_species_presence[i] >= presence_threshold else 0 for i in range(len(pred_species_presence)) 
        ]

        # Index 0 is used for empty images and should not predict presence
        assert pred_species_presence_yn[0] == 0

        pred_activity_presence_yn = [
            1 if pred_activity_presence[i] >= presence_threshold else 0 for i in range(len(pred_activity_presence))
        ]

        # If no species was predicted, consider the image empty and assign prediction to index 0
        if  sum(pred_species_presence_yn) == 0:
            # assert sum(pred_activity_presence_yn) == 0
            if sum(pred_activity_presence_yn) != 0:
                print(f'\n\ntuple: {test_tuple}')
            pred_species_presence_yn[0] = 1

        all_pred_spe_presence[pos, :] = pred_species_presence_yn
        all_pred_activity_presence[pos, :] = pred_activity_presence_yn

        # ground truth specie counts with all novel species count summed
        grd_trth_sp_cts_known_vs_novel = ground_truth_species_counts[:nbr_known_species].tolist()
        grd_trth_sp_cts_known_vs_novel.append(
            sum(ground_truth_species_counts[nbr_known_species:])
        )
        # Specie prediction counts with all novel species count summed
        pred_sp_cts_comb_novel = pred_species_counts[:nbr_known_species]
        pred_sp_cts_comb_novel.append(
            sum(pred_species_counts[nbr_known_species:])
        )

        total += 1
        if not red_button:
            total_pre_red_btn += 1

            # ---------->>>  Brier Score and accuracy element for current image (presence/absence error)  <<<----------
            # Species presence error for current image
            for i in range(total_nbr_species):
                pre_red_per_species_brier_score[i] += (ground_truth_species_presence[i] - pred_species_presence[i])**2
                pre_red_per_species_presence_acc[i] += 1 if ground_truth_species_presence[i] == pred_species_presence_yn[i] else 0

            # Activity presence error for current images
            for i in range(total_nbr_activities):
                pre_red_per_activity_brier_score[i] += (ground_truth_activity_presence[i] - pred_activity_presence[i])**2
                pre_red_per_activity_acc[i] += 1 if ground_truth_activity_presence[i] == pred_activity_presence_yn[i] else 0

            # ------------------->>>  Accuracy counts element for current image  <<<------------------
            pre_red_species_count_err1 += species_count_error(grd_trth_sp_cts_known_vs_novel, pred_sp_cts_comb_novel, metric=count_metric)
            pre_red_species_count_err2 += species_count_error(ground_truth_species_counts, pred_species_counts, metric=count_metric)

        else:
            total_post_red_btn += 1

            # ---------->>>  Brier Score and accuracy element for current image (presence/absence error)  <<<----------
            # Species presence error for current image
            for i in range(total_nbr_species):
                post_red_per_species_brier_score[i] += (ground_truth_species_presence[i] - pred_species_presence[i])**2
                post_red_per_species_presence_acc[i] += 1 if ground_truth_species_presence[i] == pred_species_presence_yn[i] else 0

            # Activity presence error for current images
            for i in range(total_nbr_activities):
                post_red_per_activity_brier_score[i] += (ground_truth_activity_presence[i] - pred_activity_presence[i])**2
                post_red_per_activity_acc[i] += 1 if ground_truth_activity_presence[i] == pred_activity_presence_yn[i] else 0

            # ------------------->>>  Accuracy counts element for current image  <<<------------------
            post_red_species_count_err1 += species_count_error(grd_trth_sp_cts_known_vs_novel, pred_sp_cts_comb_novel, metric=count_metric)
            post_red_species_count_err2 += species_count_error(ground_truth_species_counts, pred_species_counts, metric=count_metric)


    # *****************************  ACTIVITY PERFORMANCE  ********************************
    
    # ---------->>  Activity presence Brier score  <<----------
    pre_red_per_activity_brier_score = np.divide(pre_red_per_activity_brier_score, total_pre_red_btn)
    post_red_per_activity_brier_score = np.divide(post_red_per_activity_brier_score, total_post_red_btn)

    pre_red_activity_mean_brier_score = np.mean(pre_red_per_activity_brier_score)
    post_red_activity_mean_brier_score = np.mean(post_red_per_activity_brier_score)

    # --------->>  Activity presence Accuracty  <<----------
    pre_red_per_activity_acc = np.divide(pre_red_per_activity_acc, total_pre_red_btn)
    post_red_per_activity_acc = np.divide(post_red_per_activity_acc, total_post_red_btn)

    pre_red_mean_activity_acc = np.mean(pre_red_per_activity_acc)
    post_red_mean_activity_acc = np.mean(post_red_per_activity_acc)

    # -------->>  Activity presence AUC  <<---------
    for i in range(all_grd_truth_activity_presence.shape[1]):
        try:
            pre_red_per_activity_auc[i] = metrics.roc_auc_score(
                all_grd_truth_activity_presence[:total_pre_red_btn, i], 
                all_pred_activity_presence[:total_pre_red_btn, i]
            )
            post_red_per_activity_auc[i] = metrics.roc_auc_score(
                all_grd_truth_activity_presence[total_pre_red_btn:, i], 
                all_pred_activity_presence[total_pre_red_btn:, i]
            )
        except:
            pre_red_per_activity_auc[i] = -1
            post_red_per_activity_auc[i] = -1

    pre_red_mean_activity_auc = np.mean(pre_red_per_activity_auc[pre_red_per_activity_auc > 0])
    post_red_mean_activity_auc = np.mean(post_red_per_activity_auc[post_red_per_activity_auc > 0])

    # -------->>  Activity presence confusion matrix  <<---------
    pre_red_act_grd_trth = np.argmax(all_grd_truth_activity_presence[:total_pre_red_btn, :], axis=1)
    pre_red_act_pred = np.argmax(all_pred_activity_presence[:total_pre_red_btn, :], axis=1)
    pre_red_unique_act = np.unique([pre_red_act_grd_trth, pre_red_act_pred])
    pre_red_activity_cm = metrics.confusion_matrix(
        pre_red_act_grd_trth, 
        pre_red_act_pred,
        labels=np.sort(pre_red_unique_act)
    )

    post_red_act_grd_trth = np.argmax(all_grd_truth_activity_presence[total_pre_red_btn:, :], axis=1)
    post_red_act_pred = np.argmax(all_pred_activity_presence[total_pre_red_btn:, :], axis=1)
    post_red_unique_act = np.unique([post_red_act_grd_trth, post_red_act_pred])
    post_red_activity_cm = metrics.confusion_matrix(
        post_red_act_grd_trth, 
        post_red_act_pred,
        labels=np.sort(post_red_unique_act)
    )


    # *****************************  SPECIES PERFORMANCE  ********************************

    # ---------->>  Species presence Brier score  <<----------
    pre_red_per_species_brier_score = np.divide(pre_red_per_species_brier_score, total_pre_red_btn)
    post_red_per_species_brier_score = np.divide(post_red_per_species_brier_score, total_post_red_btn)

    pre_red_species_mean_brier_score = np.mean(pre_red_per_species_brier_score)
    post_red_species_mean_brier_score = np.mean(post_red_per_species_brier_score)

    # --------->>  Species presence Accuracty  <<----------
    pre_red_per_species_presence_acc = np.divide(pre_red_per_species_presence_acc, total_pre_red_btn)
    post_red_per_species_presence_acc = np.divide(post_red_per_species_presence_acc, total_post_red_btn)

    pre_red_mean_spe_presence_acc = np.mean(pre_red_per_species_presence_acc)
    post_red_mean_spe_presence_acc = np.mean(post_red_per_species_presence_acc)

    # -------->>  Species presence AUC  <<---------
    for i in range(all_grd_truth_spe_presence.shape[1]):
        try:
            pre_red_per_species_auc[i] = metrics.roc_auc_score(
                all_grd_truth_spe_presence[:total_pre_red_btn, i], 
                all_pred_spe_presence[:total_pre_red_btn, i]
            )
            post_red_per_species_auc[i] = metrics.roc_auc_score(
                all_grd_truth_spe_presence[total_pre_red_btn:, i], 
                all_pred_spe_presence[total_pre_red_btn:, i]
            )
        except:
            pre_red_per_species_auc[i] = -1
            post_red_per_species_auc[i] = -1

    pre_red_mean_spe_presence_auc = np.mean(pre_red_per_species_auc[pre_red_per_species_auc > 0])
    post_red_mean_spe_presence_auc = np.mean(post_red_per_species_auc[post_red_per_species_auc > 0])

    # -------->>  Species presence confusion matrix  <<---------
    pre_red_spe_grd_trth = np.argmax(all_grd_truth_spe_presence[:total_pre_red_btn, :], axis=1)
    pre_red_spe_pred = np.argmax(all_pred_spe_presence[:total_pre_red_btn, :], axis=1)
    pre_red_unique_spe = np.unique([pre_red_spe_grd_trth, pre_red_spe_pred])
    pre_red_species_cm = metrics.confusion_matrix(
        pre_red_spe_grd_trth, 
        pre_red_spe_pred,
        labels=np.sort(pre_red_unique_spe)
    )

    post_red_spe_grd_trth = np.argmax(all_grd_truth_spe_presence[total_pre_red_btn:, :], axis=1)
    post_red_spe_pred = np.argmax(all_pred_spe_presence[total_pre_red_btn:, :], axis=1)
    post_red_unique_spe = np.unique([post_red_spe_grd_trth, post_red_spe_pred])
    post_red_species_cm = metrics.confusion_matrix(
        post_red_spe_grd_trth, 
        post_red_spe_pred,
        labels=np.sort(post_red_unique_spe)
    )

    # -------->>  Species count error metric  <<--------
    pre_red_species_count_err1 = pre_red_species_count_err1 / total_pre_red_btn
    pre_red_species_count_err2 = pre_red_species_count_err2 / total_pre_red_btn
    post_red_species_count_err1 = post_red_species_count_err1 / total_post_red_btn
    post_red_species_count_err2 = post_red_species_count_err2 / total_post_red_btn
    

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
            'all_species': round(pre_red_species_count_err2, 3),
            'combined_novel_species': round(pre_red_species_count_err1, 3)
        },
        'post_red_btn':{
            'all_species': round(post_red_species_count_err2, 3),
            'combined_novel_species': round(post_red_species_count_err1, 3)
        }
    }
    all_performances['Species_counts'][test_id] = counts

    # Per species presence
    species_presence = {
        'pre_red_btn':{
            'bs': pre_red_per_species_brier_score,
            'acc': pre_red_per_species_presence_acc,
            'auc': pre_red_per_species_auc
        },
        'post_red_btn':{
            'bs': post_red_per_species_brier_score,
            'acc': post_red_per_species_presence_acc,
            'auc': post_red_per_species_auc
        }
    }
    all_performances['Per_species_presence'][test_id] = species_presence

    # Species presence confusion matrix
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


    # Aggregate species presence
    agg_species_presence = {
        'pre_red_btn':{
            'mean_bs': round(pre_red_species_mean_brier_score, 2),
            'mean_acc': percent_string(pre_red_mean_spe_presence_acc),
            'mean_auc': percent_string(pre_red_mean_spe_presence_auc)
        },
        'post_red_btn':{
            'mean_bs': round(post_red_species_mean_brier_score, 2),
            'mean_acc': percent_string(post_red_mean_spe_presence_acc),
            'mean_auc': percent_string(post_red_mean_spe_presence_auc)
        }
    }
    all_performances['Aggregate_species_presence'][test_id] = agg_species_presence

    # Per activity presence
    activity_presence = {
        'pre_red_btn':{
            'bs': pre_red_per_activity_brier_score,
            'acc': pre_red_per_activity_acc,
            'auc': pre_red_per_activity_auc
        },
        'post_red_btn':{
            'bs': post_red_per_activity_brier_score,
            'acc': post_red_per_activity_acc,
            'auc': post_red_per_activity_auc
        }
    }
    all_performances['Per_activity_presence'][test_id] = activity_presence

    # Aggregate activity presence
    agg_activity_presence = {
        'pre_red_btn':{
            'mean_bs': round(pre_red_activity_mean_brier_score, 2),
            'mean_acc': percent_string(pre_red_mean_activity_acc),
            'mean_auc': percent_string(pre_red_mean_activity_auc)
        },
        'post_red_btn':{
            'mean_bs': round(post_red_activity_mean_brier_score, 2),
            'mean_acc': percent_string(post_red_mean_activity_acc),
            'mean_auc': percent_string(post_red_mean_activity_auc)
        }
    }
    all_performances['Aggregate_activity_presence'][test_id] = agg_activity_presence

    # Activity presence confusion matrix
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


def score_tests(test_dir, sys_output_dir, session_id, class_file_reader, log_dir,
                save_symlinks, dataset_root, detection_threshold, presence_threshold):   
    
    stats = Stats()
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
        'Per_species_presence': {},
        'Aggregate_species_presence': {},
        'Per_activity_presence': {},
        'Aggregate_activity_presence': {},
        'Species_counts': {},
        'Species_confusion_matrices': {},
        'Activity_confusion_matrices': {},
        'Species_id2name': {},
        'Activity_id2name': {}
    }
    for test_id in test_ids:
        if 'OND' in test_id and '100.000' not in test_id:

            metadata = json.load(open(test_dir / f'{test_id}_metadata.json', 'r'))
            test_df = pd.read_csv(test_dir / f'{test_id}_single_df.csv')
            test_id = test_id[4:]

            detect_lines = []
            class_lines = []
            for round_ in range(nbr_rounds):
                if (sys_output_dir / f'{session_id}.{test_id}_{round_}_detection.csv').exists():
                    detect_lines.append(open(sys_output_dir / f'{session_id}.{test_id}_{round_}_detection.csv').read().splitlines())
                    class_lines.append(open(sys_output_dir / f'{session_id}.{test_id}_{round_}_classification.csv').read().splitlines())

                else:
                    print(f'No results found for Test {session_id}.{test_id}_{round_}.')
            detect_lines = np.concatenate(detect_lines)

            class_lines = np.concatenate(class_lines)
            
            # if (sys_output_dir / f'{session_id}.{test_id}_detection.csv').exists():
            #     detect_lines = open(sys_output_dir / f'{session_id}.{test_id}_detection.csv').read().splitlines()
            #     class_lines = open(sys_output_dir / f'{session_id}.{test_id}_classification.csv').read().splitlines()
                
            # else:
            #     print(f'No results found for Test {session_id}.{test_id}_.')
            

            with open(log_dir / f'{test_id}.log', 'w') as log:
                        score_test(test_id, metadata, test_df, detect_lines, class_lines, class_file_reader,
                                log, all_performances, stats, detection_threshold, presence_threshold)
        
        # for round_ in range(19):
        #     if (sys_output_dir / f'{session_id}.{test_id}_{round_}_detection.csv').exists():
        #         detect_lines = open(sys_output_dir / f'{session_id}.{test_id}_{round_}_detection.csv').read().splitlines()
        #         class_lines = open(sys_output_dir / f'{session_id}.{test_id}_{round_}_classification.csv').read().splitlines()
        #         with open(log_dir / f'{test_id}.log', 'w') as log:
        #             score_test(test_id, metadata, test_df, detect_lines, class_lines, class_file_reader,
        #                     log, summary, stats)
        #     else:
        #         print(f'No results found for Test {session_id}.{test_id}_{round_}.')

    write_results_to_log(all_performances, output_path=log_dir)
    print_confusion_matrices(all_performances, log_dir / "confusion.pdf")

    # stats.print_confusion_matrices_1(log_dir / "confusion.pdf")
    # if save_symlinks:
    #     stats.save_image_paths(log_dir / "images", dataset_root)

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
    p.add_argument('--presence_threshold', type=float, default=0.3)
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
    presence_threshold = args.presence_threshold
    
    score_tests(test_dir, sys_output_dir, session_id, ClassFileReader(), log_dir,
                args.save_symlinks, dataset_root, detection_threshold, presence_threshold)

if __name__ == '__main__':
    main()