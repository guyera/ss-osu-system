import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict
from pathlib import Path
import seaborn as sns

from sklearn.calibration import calibration_curve


def percent_string(num, denom=None):
    if num < 0:
        return '  --  '
    elif denom is None:
        return f'{100 * num:6.2f}%'
    return f'{100 * num / denom:6.2f}%'


def ci_string(val):
    if isinstance(val, tuple) or isinstance(val, list) or isinstance(val, np.ndarray):
        # return tuple([round(x, 2) for x in val])
        return str(tuple([round(x, 2) for x in val]))
    return ('--')


def write_results_to_csv(results_dict, output_path):

    test_ids = sorted(list(results_dict['Red_button'].keys()))
    output = [['Novelty type', 'Red button', 'Declared', 'Result', 'Delay', 'Total', 'Novel', '', '']]
    for test_id in test_ids:
        red_button_pos = results_dict['Red_button'][test_id]
        sys_declare_pos = results_dict['Red_button_declared'][test_id]
        detect = results_dict['Red_button_result'][test_id]
        delay = results_dict['Delay'][test_id]
        total = results_dict['Total'][test_id]
        novel = results_dict['Novel'][test_id]

        output.append(
            [test_id, red_button_pos, sys_declare_pos, detect, delay, total, novel, '', '']
        )

    output += [
        ['', '', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', '', '']
    ]


    # Aggregate species presence
    output += [
        ['Average Species Counts', '', '', '', '', '', '', '', ''],
        ['', 'Pre-nov', 'Pre-nov', '', 'Post-nov', 'Post-nov', '', 'Test', 'Test'],
        ['', 'MAE', 'CI MAE', '', 'MAE', 'CI MAE', '', 'MAE', 'CI MAE']
    ]
    for test_id in test_ids:
        # pre red button
        pre_red_avg_spe_abs_err = results_dict['Aggregate_species_counts'][test_id]['pre_red_btn']['value']
        if 'ci' in results_dict['Aggregate_species_counts'][test_id]['pre_red_btn']:
            pre_red_avg_spe_abs_err_ci = ci_string(results_dict['Aggregate_species_counts'][test_id]['pre_red_btn']['ci'])

        # post red button
        post_red_avg_spe_abs_err = results_dict['Aggregate_species_counts'][test_id]['post_red_btn']['value']
        if 'ci' in results_dict['Aggregate_species_counts'][test_id]['post_red_btn']:
            post_red_avg_spe_abs_err_ci = ci_string(results_dict['Aggregate_species_counts'][test_id]['post_red_btn']['ci'])

        # post red button test phase
        test_post_red_avg_spe_abs_err = results_dict['Aggregate_species_counts'][test_id]['test_post_red_btn']['value']
        if 'ci' in results_dict['Aggregate_species_counts'][test_id]['test_post_red_btn']:
            test_post_red_avg_spe_abs_err_ci = ci_string(results_dict['Aggregate_species_counts'][test_id]['test_post_red_btn']['ci'])

        output.append(
            [
                test_id, 
                f'{pre_red_avg_spe_abs_err}', 
                f'{pre_red_avg_spe_abs_err_ci}', 
                '', 
                f'{post_red_avg_spe_abs_err}', 
                f'{post_red_avg_spe_abs_err_ci}', 
                '', 
                f'{test_post_red_avg_spe_abs_err}', 
                f'{test_post_red_avg_spe_abs_err_ci}'
            ]
        )

    output += [
        ['', '', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', '', '']
    ]

        
    # Aggregate species presence
    output += [
        ['Average Species Presence', '', '', '', '', '', '', '', ''],
        ['', 'Pre-nov', 'Pre-nov', '', 'Post-nov', 'Post-nov', '', 'Test', 'Test'],
        ['', 'Avg AUC', 'Avg F1', '', 'Avg AUC', 'Avg F1', '', 'Avg AUC', 'Avg F1']
    ]

    for test_id in test_ids:
        # pre red button
        pre_red_species_avg_auc_ci, pre_red_species_avg_f1_ci, pre_red_avg_spe_abs_err_ci = '--', '--', '--'

        pre_red_species_avg_auc = results_dict['Aggregate_species_presence'][test_id]['pre_red_btn']['avg_auc']['value']
        if 'ci' in results_dict['Aggregate_species_presence'][test_id]['pre_red_btn']['avg_auc']:
            pre_red_species_avg_auc_ci = ci_string(results_dict['Aggregate_species_presence'][test_id]['pre_red_btn']['avg_auc']['ci'])

        pre_red_species_avg_f1 = results_dict['Aggregate_species_presence'][test_id]['pre_red_btn']['avg_f1_score']['value']
        if 'ci' in results_dict['Aggregate_species_presence'][test_id]['pre_red_btn']['avg_f1_score']:
            pre_red_species_avg_f1_ci = ci_string(results_dict['Aggregate_species_presence'][test_id]['pre_red_btn']['avg_f1_score']['ci'])


        # post red button
        post_red_species_avg_auc_ci, post_red_species_avg_f1_ci, post_red_avg_spe_abs_err_ci = '--', '--', '--'

        post_red_species_avg_auc = results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['avg_auc']['value']
        if 'ci' in results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['avg_auc']:
            post_red_species_avg_auc_ci = ci_string(results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['avg_auc']['ci'])

        post_red_species_avg_f1 = results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['avg_f1_score']['value']
        if 'ci' in results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['avg_f1_score']:
            post_red_species_avg_f1_ci = ci_string(results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['avg_f1_score']['ci'])

        # post red button test phase
        test_post_red_species_avg_auc_ci, test_post_red_species_avg_f1_ci, test_post_red_avg_spe_abs_err_ci = '--', '--', '--'

        test_post_red_species_avg_auc = results_dict['Aggregate_species_presence'][test_id]['test_post_red_btn']['avg_auc']['value']
        if 'ci' in results_dict['Aggregate_species_presence'][test_id]['test_post_red_btn']['avg_auc']:
            test_post_red_species_avg_auc_ci = ci_string(results_dict['Aggregate_species_presence'][test_id]['test_post_red_btn']['avg_auc']['ci'])

        test_post_red_species_avg_f1 = results_dict['Aggregate_species_presence'][test_id]['test_post_red_btn']['avg_f1_score']['value']
        if 'ci' in results_dict['Aggregate_species_presence'][test_id]['test_post_red_btn']['avg_f1_score']:
            test_post_red_species_avg_f1_ci = ci_string(results_dict['Aggregate_species_presence'][test_id]['test_post_red_btn']['avg_f1_score']['ci'])
        
        output.append(
            [
                test_id, 
                f'{pre_red_species_avg_auc} {pre_red_species_avg_auc_ci}', 
                f'{pre_red_species_avg_f1} {pre_red_species_avg_f1_ci}', 
                '', 
                f'{post_red_species_avg_auc} {post_red_species_avg_auc_ci}', 
                f'{post_red_species_avg_f1} {post_red_species_avg_f1_ci}', 
                '', 
                f'{test_post_red_species_avg_auc} {test_post_red_species_avg_auc_ci}', 
                f'{test_post_red_species_avg_f1} {test_post_red_species_avg_f1_ci}'
            ]
        )

    output += [
        ['', '', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', '', '']
    ]


    # Aggregate activity presence
    output += [
        ['Average Activity Presence', '', '', '', '', '', '', '', ''],
        ['', 'Pre-nov', 'Pre-nov', '', 'Post-nov', 'Post-nov', '', 'Test', 'Test'],
        ['', 'Avg AUC', 'Avg F1', '', 'Avg AUC', 'Avg F1', '', 'Avg AUC', 'Avg F1']
    ]

    # ** print average activity presence metrics
    for test_id in test_ids:
        # -- pre red button
        pre_red_activity_avg_auc_ci, pre_red_activity_avg_f1_ci = '--', '--'

        pre_red_activity_avg_auc = results_dict['Aggregate_activity_presence'][test_id]['pre_red_btn']['avg_auc'] ['value']
        if 'ci' in results_dict['Aggregate_activity_presence'][test_id]['pre_red_btn']['avg_auc']:
            pre_red_activity_avg_auc_ci = ci_string(results_dict['Aggregate_activity_presence'][test_id]['pre_red_btn']['avg_auc']['ci'])

        pre_red_activity_avg_precision = results_dict['Aggregate_activity_presence'][test_id]['pre_red_btn']['avg_precision']
        pre_red_activity_avg_recall = ci_string(results_dict['Aggregate_activity_presence'][test_id]['pre_red_btn']['avg_recall'])

        pre_red_activity_avg_f1 = results_dict['Aggregate_activity_presence'][test_id]['pre_red_btn']['avg_f1_score']['value']
        if 'ci' in results_dict['Aggregate_activity_presence'][test_id]['pre_red_btn']['avg_f1_score']:
            pre_red_activity_avg_f1_ci = ci_string(results_dict['Aggregate_activity_presence'][test_id]['pre_red_btn']['avg_f1_score']['ci'])

        # -- post red button
        post_red_activity_avg_auc_ci, post_red_activity_avg_f1_ci = '--', '--'

        post_red_activity_avg_auc = results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_auc']['value']
        if 'ci' in results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_auc']:
            post_red_activity_avg_auc_ci = ci_string(results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_auc']['ci'])

        post_red_activity_avg_precision = results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_precision']
        post_red_activity_avg_recall = results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_recall']

        post_red_activity_avg_f1 = results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_f1_score']['value']
        if 'ci' in results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_f1_score']:
            post_red_activity_avg_f1_ci = ci_string(results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_f1_score']['ci'])

        # -- post red button test phase
        test_post_red_activity_avg_auc_ci, test_post_red_activity_avg_f1_ci = '--', '--'

        test_post_red_activity_avg_auc = results_dict['Aggregate_activity_presence'][test_id]['test_post_red_btn']['avg_auc']['value']
        if 'ci' in results_dict['Aggregate_activity_presence'][test_id]['test_post_red_btn']['avg_auc']:
            test_post_red_activity_avg_auc_ci = ci_string(results_dict['Aggregate_activity_presence'][test_id]['test_post_red_btn']['avg_auc']['ci'])

        test_post_red_activity_avg_precision = results_dict['Aggregate_activity_presence'][test_id]['test_post_red_btn']['avg_precision']
        test_post_red_activity_avg_recall = results_dict['Aggregate_activity_presence'][test_id]['test_post_red_btn']['avg_recall']

        test_post_red_activity_avg_f1 = results_dict['Aggregate_activity_presence'][test_id]['test_post_red_btn']['avg_f1_score']['value']
        if 'ci' in results_dict['Aggregate_activity_presence'][test_id]['test_post_red_btn']['avg_f1_score']:
            test_post_red_activity_avg_f1_ci = ci_string(results_dict['Aggregate_activity_presence'][test_id]['test_post_red_btn']['avg_f1_score']['ci'])

        output.append(
            [
                test_id, 
                f'{pre_red_activity_avg_auc} {pre_red_activity_avg_auc_ci}', 
                f'{pre_red_activity_avg_f1} {pre_red_activity_avg_f1_ci}', 
                '', 
                f'{post_red_activity_avg_auc} {post_red_activity_avg_auc_ci}', 
                f'{post_red_activity_avg_f1} {post_red_activity_avg_f1_ci}', 
                '', 
                f'{test_post_red_activity_avg_auc} {test_post_red_activity_avg_auc_ci}', 
                f'{test_post_red_activity_avg_f1} {test_post_red_activity_avg_f1_ci}'
            ]
        )
  
    output += [
        ['', '', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', '', '']
    ]

    # Per species count errors
    output.append(['Per Species Counts', '', '', '', '', '', '', '', ''])
    for test_id in test_ids:
        # species counts
        pre_red_per_species_count_abs_err = results_dict['Species_counts'][test_id]['pre_red_btn']
        
        post_red_per_species_count_abs_err = results_dict['Species_counts'][test_id]['post_red_btn']
        
        test_post_red_per_species_count_abs_err = results_dict['Species_counts'][test_id]['test_post_red_btn']
        

        species_id2name_mapping = results_dict['Species_id2name'][test_id]
        species_name2id_mapping = dict((v,k) for k,v in species_id2name_mapping.items())

        output += [
            [f'{test_id}', '', '', '', '', '', '', '', ''],
            ['', 'Pre-nov', 'Pre-nov', '', 'Post-nov', 'Post-nov', '', 'Test', 'Test'],
            ['', 'MAE', 'MAE-CI', '', 'MAE', 'MAE-CI', '', 'MAE', 'MAE-CI']
        ]

        species_present = list(set(list(pre_red_per_species_count_abs_err.keys()) + list(pre_red_per_species_count_abs_err.keys())))
        species_present_ids = [int(x.split('_')[-1]) if 'unknown' in x else species_name2id_mapping[x] for x in species_present]
        idx_sort_spe_present = np.argsort(species_present_ids)

        for i in idx_sort_spe_present:
            spe_name = species_present[i]
            if 'unknown' in spe_name:
                species_name = 'unknown'
                spe_id = int(spe_name.split('_')[-1])
            else:
                # this is an empty images
                species_name = spe_name
                spe_id = species_name2id_mapping[spe_name]
            
            # pre red button
            pre_red_per_spe_cnt_abs_err_ci = '--'
            pre_red_per_spe_cnt_abs_err = round(pre_red_per_species_count_abs_err[spe_name]['value'], 2)
            if 'ci' in pre_red_per_species_count_abs_err[spe_name]:
                pre_red_per_spe_cnt_abs_err_ci = ci_string(pre_red_per_species_count_abs_err[spe_name]['ci'])

            # post red button
            post_red_per_spe_cnt_abs_err_ci = '--'

            post_red_per_spe_cnt_abs_err = round(post_red_per_species_count_abs_err[spe_name]['value'], 2)
            if 'ci' in post_red_per_species_count_abs_err[spe_name]:
                post_red_per_spe_cnt_abs_err_ci = ci_string(post_red_per_species_count_abs_err[spe_name]['ci'])

            # post red button test phase
            test_post_red_per_spe_cnt_abs_err_ci = '--'

            test_post_red_per_spe_cnt_abs_err = round(test_post_red_per_species_count_abs_err[spe_name]['value'], 2)
            if 'ci' in test_post_red_per_species_count_abs_err[spe_name]:
                test_post_red_per_spe_cnt_abs_err_ci = ci_string(test_post_red_per_species_count_abs_err[spe_name]['ci'])

            output.append(
                [
                    f'{species_name} ({spe_id}):', 
                    f'{pre_red_per_spe_cnt_abs_err}', 
                    f'{pre_red_per_spe_cnt_abs_err_ci}', 
                    '', 
                    f'{post_red_per_spe_cnt_abs_err}', 
                    f'{post_red_per_spe_cnt_abs_err_ci}', 
                    '', 
                    f'{test_post_red_per_spe_cnt_abs_err}', 
                    f'{test_post_red_per_spe_cnt_abs_err_ci}'
                ]
            )

        output.append(['', '', '', '', '', '', '', '', ''])


    output += [
        ['', '', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', '', '']
    ]

    # ****
    # Per species presence
    output.append(['Per Species Presence', '', '', '', '', '', '', '', ''])
    for test_id in test_ids:
        # species present
        pre_red_per_species_auc = results_dict['Per_species_presence'][test_id]['pre_red_btn']['auc']
        pre_red_per_species_f1_score = results_dict['Per_species_presence'][test_id]['pre_red_btn']['f1_score']

        post_red_per_species_auc = results_dict['Per_species_presence'][test_id]['post_red_btn']['auc']
        post_red_per_species_f1_score = results_dict['Per_species_presence'][test_id]['post_red_btn']['f1_score']

        test_post_red_per_species_auc = results_dict['Per_species_presence'][test_id]['test_post_red_btn']['auc']
        test_post_red_per_species_f1_score = results_dict['Per_species_presence'][test_id]['test_post_red_btn']['f1_score']

        species_id2name_mapping = results_dict['Species_id2name'][test_id]
        species_name2id_mapping = dict((v,k) for k,v in species_id2name_mapping.items())

        output += [
            [test_id, '', '', '', '', '', '', '', ''],
            ['', 'Pre-nov', 'Pre-nov', '', 'Post-nov', 'Post-nov', '', 'Test', 'Test'],
            ['', 'AUC', 'F1', '', 'AUC', 'F1', '', 'AUC', 'F1']
        ]

        species_present = list(set(
            list(pre_red_per_species_auc.keys()) + 
            list(post_red_per_species_auc.keys()) + 
            list(test_post_red_per_species_auc.keys())
            ))
        species_present_ids = [int(x.split('_')[-1]) if 'unknown' in x else species_name2id_mapping[x] for x in species_present]
        idx_sort_spe_present = np.argsort(species_present_ids)

        for i in idx_sort_spe_present:
            spe_name = species_present[i]
            if 'unknown' in spe_name:
                species_name = 'unknown'
                spe_id = int(spe_name.split('_')[-1])
            else:
                # this is an empty images
                species_name = spe_name
                spe_id = species_name2id_mapping[spe_name]
            
            # pre red button
            pre_red_per_spe_auc, pre_red_per_spe_f1 = -1, -1
            pre_red_per_spe_auc_ci, pre_red_per_spe_f1_ci = '--', '--'
            
            if spe_name in pre_red_per_species_auc.keys():
                pre_red_per_spe_auc = round(pre_red_per_species_auc[spe_name]['value'], 2)
                if 'ci' in pre_red_per_species_auc[spe_name]:
                    pre_red_per_spe_auc_ci = ci_string(pre_red_per_species_auc[spe_name]['ci'])

                pre_red_per_spe_f1 = round(pre_red_per_species_f1_score[spe_name]['value'], 2)
                if 'ci' in pre_red_per_species_f1_score[spe_name]:
                    pre_red_per_spe_f1_ci = ci_string(pre_red_per_species_f1_score[spe_name]['ci'])

            # post red button
            post_red_per_spe_auc, post_red_per_spe_f1 = -1, -1
            post_red_per_spe_auc_ci, post_red_per_spe_f1_ci = '--', '--'
            
            if spe_name in post_red_per_species_auc.keys():
                post_red_per_spe_auc = round(post_red_per_species_auc[spe_name]['value'], 2)
                if 'ci' in post_red_per_species_auc[spe_name]:
                    post_red_per_spe_auc_ci = ci_string(post_red_per_species_auc[spe_name]['ci'])

                post_red_per_spe_f1 = round(post_red_per_species_f1_score[spe_name]['value'], 2)
                if 'ci' in post_red_per_species_f1_score[spe_name]:
                    post_red_per_spe_f1_ci = ci_string(post_red_per_species_f1_score[spe_name]['ci'])

            # post red button test phase
            test_post_red_per_spe_auc, test_post_red_per_spe_f1 = -1, -1
            test_post_red_per_spe_auc_ci, test_post_red_per_spe_f1_ci = '--', '--'
            
            if spe_name in test_post_red_per_species_auc.keys():
                test_post_red_per_spe_auc = round(test_post_red_per_species_auc[spe_name]['value'], 2)
                if 'ci' in test_post_red_per_species_auc[spe_name]:
                    test_post_red_per_spe_auc_ci = ci_string(test_post_red_per_species_auc[spe_name]['ci'])

                test_post_red_per_spe_f1 = round(test_post_red_per_species_f1_score[spe_name]['value'], 2)
                if 'ci' in test_post_red_per_species_f1_score[spe_name]:
                    test_post_red_per_spe_f1_ci = ci_string(test_post_red_per_species_f1_score[spe_name]['ci'])

            output.append(
                [
                    f'{species_name} ({spe_id}):', 
                    f'{pre_red_per_spe_auc} {pre_red_per_spe_auc_ci}', 
                    f'{pre_red_per_spe_f1} {pre_red_per_spe_f1_ci}', 
                    '', 
                    f'{post_red_per_spe_auc} {post_red_per_spe_auc_ci}', 
                    f'{post_red_per_spe_f1} {post_red_per_spe_f1_ci}', 
                    '', 
                    f'{test_post_red_per_spe_auc} {test_post_red_per_spe_auc_ci}', 
                    f'{test_post_red_per_spe_f1} {test_post_red_per_spe_f1_ci}'
                ]
            )

        output.append(['', '', '', '', '', '', '', '', ''])


    output += [
        ['', '', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', '', '']
    ]

    # Per activity presence
    output.append(['Per Activity Presence', '', '', '', '', '', '', '', ''])
    for test_id in test_ids:
        pre_red_per_activity_auc = results_dict['Per_activity_presence'][test_id]['pre_red_btn']['auc']
        pre_red_per_activity_f1_score = results_dict['Per_activity_presence'][test_id]['pre_red_btn']['f1_score']

        post_red_per_activity_auc = results_dict['Per_activity_presence'][test_id]['post_red_btn']['auc']
        post_red_per_activity_f1_score = results_dict['Per_activity_presence'][test_id]['post_red_btn']['f1_score']

        test_post_red_per_activity_auc = results_dict['Per_activity_presence'][test_id]['test_post_red_btn']['auc']
        test_post_red_per_activity_f1_score = results_dict['Per_activity_presence'][test_id]['test_post_red_btn']['f1_score']

        activity_id2name_mapping = results_dict['Activity_id2name'][test_id]
        activity_name2id_mapping = dict((v,k) for k,v in activity_id2name_mapping.items())

        output += [
            [test_id, '', '', '', '', '', '', '', ''],
            ['', 'Pre-nov', 'Pre-nov', '', 'Post-nov', 'Post-nov', '', 'Test', 'Test'],
            ['', 'AUC', 'F1', '', 'AUC', 'F1', '', 'AUC', 'F1']
        ]

        present_activities = list(set(
            list(pre_red_per_activity_auc.keys()) + 
            list(post_red_per_activity_auc.keys()) + 
            list(test_post_red_per_activity_auc.keys())
            ))
        activities_present_ids = [int(x.split('_')[-1]) if 'unknown' in x else activity_name2id_mapping[x] for x in present_activities]
        idx_sort_act_present = np.argsort(activities_present_ids)

        for i in idx_sort_act_present:
            act_name = present_activities[i]
            if 'unknown' in act_name:
                activity_name = 'unknown'
                act_id = int(act_name.split('_')[-1])
            else:
                activity_name = act_name
                act_id = activity_name2id_mapping[act_name]

            # pre red button
            pre_red_per_act_auc, pre_red_per_act_f1 = -1, -1
            pre_red_per_act_auc_ci, pre_red_per_act_f1_ci = '--', '--'

            if act_name in pre_red_per_activity_auc.keys():
                pre_red_per_act_auc = round(pre_red_per_activity_auc[act_name]['value'], 2)
                if 'ci' in pre_red_per_activity_auc[act_name]:
                    pre_red_per_act_auc_ci = ci_string(pre_red_per_activity_auc[act_name]['ci'])

                pre_red_per_act_f1 = round(pre_red_per_activity_f1_score[act_name]['value'], 2)
                if 'ci' in pre_red_per_activity_f1_score[act_name]:
                    pre_red_per_act_f1_ci = ci_string(pre_red_per_activity_f1_score[act_name]['ci'])

            # post red button
            post_red_per_act_auc, post_red_per_act_f1 = -1, -1
            post_red_per_act_auc_ci, post_red_per_act_f1_ci = '--', '--'

            if act_name in post_red_per_activity_auc.keys():
                post_red_per_act_auc = round(post_red_per_activity_auc[act_name]['value'], 2)
                if 'ci' in post_red_per_activity_auc[act_name]:
                    post_red_per_act_auc_ci = ci_string(post_red_per_activity_auc[act_name]['ci'])

                post_red_per_act_f1 = round(post_red_per_activity_f1_score[act_name]['value'], 2)
                if 'ci' in post_red_per_activity_f1_score[act_name]:
                    post_red_per_act_f1_ci = ci_string(post_red_per_activity_f1_score[act_name]['ci'])

            # post red button test phase
            test_post_red_per_act_auc, test_post_red_per_act_f1 = -1, -1
            test_post_red_per_act_auc_ci, test_post_red_per_act_f1_ci = '--', '--'

            if act_name in test_post_red_per_activity_auc.keys():
                test_post_red_per_act_auc = round(test_post_red_per_activity_auc[act_name]['value'], 2)
                if 'ci' in test_post_red_per_activity_auc[act_name]:
                    test_post_red_per_act_auc_ci = ci_string(test_post_red_per_activity_auc[act_name]['ci'])

                test_post_red_per_act_f1 = round(test_post_red_per_activity_f1_score[act_name]['value'], 2)
                if 'ci' in test_post_red_per_activity_f1_score[act_name]:
                    test_post_red_per_act_f1_ci = ci_string(test_post_red_per_activity_f1_score[act_name]['ci'])

            output.append(
                [
                    f'{activity_name} ({act_id}):', 
                    f'{pre_red_per_act_auc} {pre_red_per_act_auc_ci}', 
                    f'{pre_red_per_act_f1} {pre_red_per_act_f1_ci}', 
                    '', 
                    f'{post_red_per_act_auc} {post_red_per_act_auc_ci}', 
                    f'{post_red_per_act_f1} {post_red_per_act_f1_ci}', 
                    '', 
                    f'{test_post_red_per_act_auc} {test_post_red_per_act_auc_ci}', 
                    f'{test_post_red_per_act_f1} {test_post_red_per_act_f1_ci}'
                ]
            )
        output.append(['', '', '', '', '', '', '', '', ''])
    
    with open(os.path.join(output_path, f'summary.csv'), 'w') as myfile:
     
        # using csv.writer method from CSV package
        write = csv.writer(myfile)
        write.writerows(output)


def write_results_to_log(results_dict, output_path):

    test_ids = sorted(list(results_dict['Red_button'].keys()))
    with open(os.path.join(output_path, f'summary.log'), 'w') as summary:
        '''
        summary.write(f'              Red Decl Res Delay                  ***********************  Species count  ************************   \n')
        summary.write(f'                                  Total    Novel  ------------  Pre  ----------- | ------------  Post  ----------   \n')
        summary.write(f'                                                  separate_novel  combined_novel | separate_novel  combined_novel    \n')
        '''
        summary.write(f'                                                                 \n')
        summary.write(f'              Red   Decl   Res  Delay   Total    Novel     \n')
        summary.write(f'                                                                                                     \n')
        
        for test_id in test_ids:
            red_button_pos = results_dict['Red_button'][test_id]
            sys_declare_pos = results_dict['Red_button_declared'][test_id]
            detect = results_dict['Red_button_result'][test_id]
            delay = results_dict['Delay'][test_id]
            total = results_dict['Total'][test_id]
            novel = results_dict['Novel'][test_id]

            summary.write(f'{test_id}: '
                  f'{red_button_pos:8} {sys_declare_pos:6}   {detect:5} {delay:5}    ' 
                  f'{total:3}    {novel:3} \n')


        summary.write('\n\n')


        # # Aggregate species presence
        # summary.write(f'               ****************************************************  Aggreate species presence and counts  ***************************************************************   \n')
        # summary.write(f'               ---------------------------  Pre  ---------------------------------         |       ---------------------------------  Post  ------------------------------   \n')
        # summary.write(f'               Avg_AUC            Avg_F1            Avg_abs_err            Avg_rel         |          Avg_AUC            Avg_F1            Avg_abs_err            Avg_rel    \n')

        # Aggregate species presence
        summary.write(f'              **********************************************************  Average species counts  ***************************************************************      \n')
        summary.write(f'               ------------------  Pre  ---------------    |     -------------------  Post  -----------------    |    -------------------  Test  ----------------   \n')
        summary.write(f'                     Avg_abs_err            Avg_rel        |           Avg_abs_err            Avg_rel            |          Avg_abs_err            Avg_rel          \n')

        for test_id in test_ids:
            # pre red button
            pre_red_avg_spe_abs_err = results_dict['Aggregate_species_counts'][test_id]['pre_red_btn']['avg_abs_err']['value']
            if 'ci' in results_dict['Aggregate_species_counts'][test_id]['pre_red_btn']['avg_abs_err']:
                pre_red_avg_spe_abs_err_ci = ci_string(results_dict['Aggregate_species_counts'][test_id]['pre_red_btn']['avg_abs_err']['ci'])

            pre_red_avg_spe_rel_err = results_dict['Aggregate_species_counts'][test_id]['pre_red_btn']['avg_rel_err']['value']
            if 'ci' in results_dict['Aggregate_species_counts'][test_id]['pre_red_btn']['avg_rel_err']:
                pre_red_avg_spe_rel_err_ci = ci_string(results_dict['Aggregate_species_counts'][test_id]['pre_red_btn']['avg_rel_err']['ci'])

            # post red button
            post_red_avg_spe_abs_err = results_dict['Aggregate_species_counts'][test_id]['post_red_btn']['avg_abs_err']['value']
            if 'ci' in results_dict['Aggregate_species_counts'][test_id]['post_red_btn']['avg_abs_err']:
                post_red_avg_spe_abs_err_ci = ci_string(results_dict['Aggregate_species_counts'][test_id]['post_red_btn']['avg_abs_err']['ci'])

            post_red_avg_spe_rel_err = results_dict['Aggregate_species_counts'][test_id]['post_red_btn']['avg_rel_err']['value']
            if 'ci' in results_dict['Aggregate_species_counts'][test_id]['post_red_btn']['avg_rel_err']:
                post_red_avg_spe_rel_err_ci = ci_string(results_dict['Aggregate_species_counts'][test_id]['post_red_btn']['avg_rel_err']['ci'])

            # post red button test phase
            test_post_red_avg_spe_abs_err = results_dict['Aggregate_species_counts'][test_id]['test_post_red_btn']['avg_abs_err']['value']
            if 'ci' in results_dict['Aggregate_species_counts'][test_id]['test_post_red_btn']['avg_abs_err']:
                test_post_red_avg_spe_abs_err_ci = ci_string(results_dict['Aggregate_species_counts'][test_id]['test_post_red_btn']['avg_abs_err']['ci'])

            test_post_red_avg_spe_rel_err = results_dict['Aggregate_species_counts'][test_id]['test_post_red_btn']['avg_rel_err']['value']
            if 'ci' in results_dict['Aggregate_species_counts'][test_id]['test_post_red_btn']['avg_rel_err']:
                test_post_red_avg_spe_rel_err_ci = ci_string(results_dict['Aggregate_species_counts'][test_id]['test_post_red_btn']['avg_rel_err']['ci'])

            """
            summary.write(f'{test_id}: '
                  f'{pre_red_species_avg_auc:4} ({pre_red_species_avg_auc_ci:8})   {pre_red_species_avg_f1:4} ({pre_red_species_avg_f1_ci:8})   '
                  f'{pre_red_avg_spe_abs_err:4} ({pre_red_avg_spe_abs_err_ci:8})   {pre_red_avg_spe_rel_err:4} ({pre_red_avg_spe_rel_err_ci:8})   ' 
                  f'{post_red_species_avg_auc:4} ({post_red_species_avg_auc_ci:8})   {post_red_species_avg_f1:4} ({post_red_species_avg_f1_ci:8})   '
                  f'{post_red_avg_spe_abs_err:4} ({post_red_avg_spe_abs_err_ci:8})   {post_red_avg_spe_rel_err:4} ({post_red_avg_spe_rel_err_ci:8}) \n')
            """
            
            summary.write(f'{test_id}:     '
                  f'{pre_red_avg_spe_abs_err:4} {pre_red_avg_spe_abs_err_ci:12}   {pre_red_avg_spe_rel_err:4} {pre_red_avg_spe_rel_err_ci:12}         ' 
                  f'{post_red_avg_spe_abs_err:4} {post_red_avg_spe_abs_err_ci:12}   {post_red_avg_spe_rel_err:4} {post_red_avg_spe_rel_err_ci:12}         '
                  f'{test_post_red_avg_spe_abs_err:4} {test_post_red_avg_spe_abs_err_ci:12}   {test_post_red_avg_spe_rel_err:4} {test_post_red_avg_spe_rel_err_ci:12} \n')
        
        summary.write('\n\n')


        # Aggregate activity presence
        summary.write(f'              ***************************************************  Average species presence  ***************************************************   \n')
        summary.write(f'               ---------------  Pre  -------------    |   ----------------  Post  ---------------    |   ----------------  Test  ---------------   \n')
        summary.write(f'                      Avg_AUC            Avg_F1       |           avg_AUC            Avg_F1          |           avg_AUC            Avg_F1         \n')

        for test_id in test_ids:
            # pre red button
            pre_red_species_avg_auc_ci, pre_red_species_avg_f1_ci, pre_red_avg_spe_abs_err_ci, pre_red_avg_spe_rel_err_ci = '--', '--', '--', '--'

            pre_red_species_avg_auc = results_dict['Aggregate_species_presence'][test_id]['pre_red_btn']['avg_auc']['value']
            if 'ci' in results_dict['Aggregate_species_presence'][test_id]['pre_red_btn']['avg_auc']:
                pre_red_species_avg_auc_ci = ci_string(results_dict['Aggregate_species_presence'][test_id]['pre_red_btn']['avg_auc']['ci'])

            pre_red_species_avg_precision = results_dict['Aggregate_species_presence'][test_id]['pre_red_btn']['avg_precision']
            pre_red_species_avg_recall = results_dict['Aggregate_species_presence'][test_id]['pre_red_btn']['avg_recall']

            pre_red_species_avg_f1 = results_dict['Aggregate_species_presence'][test_id]['pre_red_btn']['avg_f1_score']['value']
            if 'ci' in results_dict['Aggregate_species_presence'][test_id]['pre_red_btn']['avg_f1_score']:
                pre_red_species_avg_f1_ci = ci_string(results_dict['Aggregate_species_presence'][test_id]['pre_red_btn']['avg_f1_score']['ci'])


            # post red button
            post_red_species_avg_auc_ci, post_red_species_avg_f1_ci, post_red_avg_spe_abs_err_ci, post_red_avg_spe_rel_err_ci = '--', '--', '--', '--'

            post_red_species_avg_auc = results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['avg_auc']['value']
            if 'ci' in results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['avg_auc']:
                post_red_species_avg_auc_ci = ci_string(results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['avg_auc']['ci'])

            post_red_species_avg_precision = results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['avg_precision']
            post_red_species_avg_recall = results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['avg_recall']

            post_red_species_avg_f1 = results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['avg_f1_score']['value']
            if 'ci' in results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['avg_f1_score']:
                post_red_species_avg_f1_ci = ci_string(results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['avg_f1_score']['ci'])

            # post red button test phase
            test_post_red_species_avg_auc_ci, test_post_red_species_avg_f1_ci, test_post_red_avg_spe_abs_err_ci, test_post_red_avg_spe_rel_err_ci = '--', '--', '--', '--'

            test_post_red_species_avg_auc = results_dict['Aggregate_species_presence'][test_id]['test_post_red_btn']['avg_auc']['value']
            if 'ci' in results_dict['Aggregate_species_presence'][test_id]['test_post_red_btn']['avg_auc']:
                test_post_red_species_avg_auc_ci = ci_string(results_dict['Aggregate_species_presence'][test_id]['test_post_red_btn']['avg_auc']['ci'])

            test_post_red_species_avg_f1 = results_dict['Aggregate_species_presence'][test_id]['test_post_red_btn']['avg_f1_score']['value']
            if 'ci' in results_dict['Aggregate_species_presence'][test_id]['test_post_red_btn']['avg_f1_score']:
                test_post_red_species_avg_f1_ci = ci_string(results_dict['Aggregate_species_presence'][test_id]['test_post_red_btn']['avg_f1_score']['ci'])
            
            summary.write(f'{test_id}:   '
                  f'{pre_red_species_avg_auc:4} {pre_red_species_avg_auc_ci:12}   {pre_red_species_avg_f1:4} {pre_red_species_avg_f1_ci:12}         '
                  f'{post_red_species_avg_auc:4} {post_red_species_avg_auc_ci:12}   {post_red_species_avg_f1:4} {post_red_species_avg_f1_ci:12}         '
                  f'{test_post_red_species_avg_auc:4} {test_post_red_species_avg_auc_ci:12}   {test_post_red_species_avg_f1:4} {test_post_red_species_avg_f1_ci:12}  \n')
        
        summary.write('\n\n')


        # Aggregate activity presence
        summary.write(f'              ***************************************************  Average activity presence  **************************************************   \n')
        summary.write(f'               ---------------  Pre  -------------    |   ----------------  Post  ---------------    |   ----------------  Test  ---------------   \n')
        summary.write(f'                      Avg_AUC            Avg_F1       |           avg_AUC            Avg_F1          |           avg_AUC            Avg_F1         \n')
        

        # ** print average activity presence metrics
        for test_id in test_ids:
            # -- pre red button
            pre_red_activity_avg_auc_ci, pre_red_activity_avg_f1_ci = '--', '--'

            pre_red_activity_avg_auc = results_dict['Aggregate_activity_presence'][test_id]['pre_red_btn']['avg_auc'] ['value']
            if 'ci' in results_dict['Aggregate_activity_presence'][test_id]['pre_red_btn']['avg_auc']:
                pre_red_activity_avg_auc_ci = ci_string(results_dict['Aggregate_activity_presence'][test_id]['pre_red_btn']['avg_auc']['ci'])

            pre_red_activity_avg_precision = results_dict['Aggregate_activity_presence'][test_id]['pre_red_btn']['avg_precision']
            pre_red_activity_avg_recall = ci_string(results_dict['Aggregate_activity_presence'][test_id]['pre_red_btn']['avg_recall'])

            pre_red_activity_avg_f1 = results_dict['Aggregate_activity_presence'][test_id]['pre_red_btn']['avg_f1_score']['value']
            if 'ci' in results_dict['Aggregate_activity_presence'][test_id]['pre_red_btn']['avg_f1_score']:
                pre_red_activity_avg_f1_ci = ci_string(results_dict['Aggregate_activity_presence'][test_id]['pre_red_btn']['avg_f1_score']['ci'])

            # -- post red button
            post_red_activity_avg_auc_ci, post_red_activity_avg_f1_ci = '--', '--'

            post_red_activity_avg_auc = results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_auc']['value']
            if 'ci' in results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_auc']:
                post_red_activity_avg_auc_ci = ci_string(results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_auc']['ci'])

            post_red_activity_avg_precision = results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_precision']
            post_red_activity_avg_recall = results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_recall']

            post_red_activity_avg_f1 = results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_f1_score']['value']
            if 'ci' in results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_f1_score']:
                post_red_activity_avg_f1_ci = ci_string(results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_f1_score']['ci'])

            # -- post red button test phase
            test_post_red_activity_avg_auc_ci, test_post_red_activity_avg_f1_ci = '--', '--'

            test_post_red_activity_avg_auc = results_dict['Aggregate_activity_presence'][test_id]['test_post_red_btn']['avg_auc']['value']
            if 'ci' in results_dict['Aggregate_activity_presence'][test_id]['test_post_red_btn']['avg_auc']:
                test_post_red_activity_avg_auc_ci = ci_string(results_dict['Aggregate_activity_presence'][test_id]['test_post_red_btn']['avg_auc']['ci'])

            test_post_red_activity_avg_precision = results_dict['Aggregate_activity_presence'][test_id]['test_post_red_btn']['avg_precision']
            test_post_red_activity_avg_recall = results_dict['Aggregate_activity_presence'][test_id]['test_post_red_btn']['avg_recall']

            test_post_red_activity_avg_f1 = results_dict['Aggregate_activity_presence'][test_id]['test_post_red_btn']['avg_f1_score']['value']
            if 'ci' in results_dict['Aggregate_activity_presence'][test_id]['test_post_red_btn']['avg_f1_score']:
                test_post_red_activity_avg_f1_ci = ci_string(results_dict['Aggregate_activity_presence'][test_id]['test_post_red_btn']['avg_f1_score']['ci'])

            summary.write(f'{test_id}:      '
                    f'{pre_red_activity_avg_auc:5} {pre_red_activity_avg_auc_ci:12}  {pre_red_activity_avg_f1:5} {pre_red_activity_avg_f1_ci:12}         '
                    f'{post_red_activity_avg_auc:5} {post_red_activity_avg_auc_ci:12}    {post_red_activity_avg_f1:5} {post_red_activity_avg_f1_ci:12}         '
                    f'{test_post_red_activity_avg_auc:5} {test_post_red_activity_avg_auc_ci:12}    {test_post_red_activity_avg_f1:5} {test_post_red_activity_avg_f1_ci:12} \n')
                    

        summary.write(f"\n\n{'_'*150}\n\n")

        # Per species count errors
        summary.write(f'                             *********  Per species count errors  *********  \n')

        for test_id in test_ids:
            # species counts
            pre_red_per_species_count_abs_err = results_dict['Species_counts'][test_id]['pre_red_btn']['abs_err']
            pre_red_per_species_count_rel_err = results_dict['Species_counts'][test_id]['pre_red_btn']['rel_err']

            post_red_per_species_count_abs_err = results_dict['Species_counts'][test_id]['post_red_btn']['abs_err']
            post_red_per_species_count_rel_err = results_dict['Species_counts'][test_id]['post_red_btn']['rel_err']

            test_post_red_per_species_count_abs_err = results_dict['Species_counts'][test_id]['test_post_red_btn']['abs_err']
            test_post_red_per_species_count_rel_err = results_dict['Species_counts'][test_id]['test_post_red_btn']['rel_err']


            species_id2name_mapping = results_dict['Species_id2name'][test_id]
            species_name2id_mapping = dict((v,k) for k,v in species_id2name_mapping.items())

            summary.write(f'\n                              ---------------  Pre  ------------------   |     ---------------  Post  ------------------     |    ----------------  Test  ------------------------------    \n')
            summary.write(f'                                 Abs_err                   Rel_err         |        Abs_err                   Rel_err          |           Abs_err                   Rel_err                \n')
            summary.write(f'{test_id}: \n')

            species_present = list(set(list(pre_red_per_species_count_abs_err.keys()) + list(pre_red_per_species_count_abs_err.keys())))
            species_present_ids = [int(x.split('_')[-1]) if 'unknown' in x else species_name2id_mapping[x] for x in species_present]
            idx_sort_spe_present = np.argsort(species_present_ids)

            for i in idx_sort_spe_present:
                spe_name = species_present[i]
                if 'unknown' in spe_name:
                    species_name = 'unknown'
                    spe_id = int(spe_name.split('_')[-1])
                else:
                    # this is an empty images
                    species_name = spe_name
                    spe_id = species_name2id_mapping[spe_name]
                
                # pre red button
                pre_red_per_spe_cnt_abs_err_ci, pre_red_per_spe_cnt_rel_err_ci = '--', '--'
                pre_red_per_spe_cnt_abs_err = round(pre_red_per_species_count_abs_err[spe_name]['value'], 2)
                if 'ci' in pre_red_per_species_count_abs_err[spe_name]:
                    pre_red_per_spe_cnt_abs_err_ci = ci_string(pre_red_per_species_count_abs_err[spe_name]['ci'])

                pre_red_per_spe_cnt_rel_err = round(pre_red_per_species_count_rel_err[spe_name]['value'], 2)
                if 'ci' in pre_red_per_species_count_rel_err[spe_name]:
                    pre_red_per_spe_cnt_rel_err_ci = ci_string(pre_red_per_species_count_rel_err[spe_name]['ci'])

                # post red button
                post_red_per_spe_cnt_abs_err_ci, post_red_per_spe_cnt_rel_err_ci = '--', '--'

                post_red_per_spe_cnt_abs_err = round(post_red_per_species_count_abs_err[spe_name]['value'], 2)
                if 'ci' in post_red_per_species_count_abs_err[spe_name]:
                    post_red_per_spe_cnt_abs_err_ci = ci_string(post_red_per_species_count_abs_err[spe_name]['ci'])

                post_red_per_spe_cnt_rel_err = round(post_red_per_species_count_rel_err[spe_name]['value'], 2)
                if 'ci' in post_red_per_species_count_rel_err[spe_name]:
                    post_red_per_spe_cnt_rel_err_ci = ci_string(post_red_per_species_count_rel_err[spe_name]['ci'])

                # post red button test phase
                test_post_red_per_spe_cnt_abs_err_ci, test_post_red_per_spe_cnt_rel_err_ci = '--', '--'

                test_post_red_per_spe_cnt_abs_err = round(test_post_red_per_species_count_abs_err[spe_name]['value'], 2)
                if 'ci' in test_post_red_per_species_count_abs_err[spe_name]:
                    test_post_red_per_spe_cnt_abs_err_ci = ci_string(test_post_red_per_species_count_abs_err[spe_name]['ci'])

                test_post_red_per_spe_cnt_rel_err = round(test_post_red_per_species_count_rel_err[spe_name]['value'], 2)
                if 'ci' in test_post_red_per_species_count_rel_err[spe_name]:
                    test_post_red_per_spe_cnt_rel_err_ci = ci_string(test_post_red_per_species_count_rel_err[spe_name]['ci'])

                summary.write(f'      {species_name:15} ({spe_id}):   '
                  f'{pre_red_per_spe_cnt_abs_err:5} {pre_red_per_spe_cnt_abs_err_ci:12}   {pre_red_per_spe_cnt_rel_err:5} {pre_red_per_spe_cnt_rel_err_ci:12}         '
                  f'{post_red_per_spe_cnt_abs_err:5} {post_red_per_spe_cnt_abs_err_ci:12}   {post_red_per_spe_cnt_rel_err:5} {post_red_per_spe_cnt_rel_err_ci:12}         '
                  f'{test_post_red_per_spe_cnt_abs_err:5} {test_post_red_per_spe_cnt_abs_err_ci:12}   {test_post_red_per_spe_cnt_rel_err:5} {test_post_red_per_spe_cnt_rel_err_ci:12} \n')
                
            summary.write(f"{'-'*100}\n")

        # ****
        # Per species presence
        summary.write(f'                             *********  Per species presence  *********  \n')

        for test_id in test_ids:
            # species present
            pre_red_per_species_auc = results_dict['Per_species_presence'][test_id]['pre_red_btn']['auc']
            pre_red_per_species_precision = results_dict['Per_species_presence'][test_id]['pre_red_btn']['precision']
            pre_red_per_species_recall = results_dict['Per_species_presence'][test_id]['pre_red_btn']['recall']
            pre_red_per_species_f1_score = results_dict['Per_species_presence'][test_id]['pre_red_btn']['f1_score']

            post_red_per_species_auc = results_dict['Per_species_presence'][test_id]['post_red_btn']['auc']
            post_red_per_species_precision = results_dict['Per_species_presence'][test_id]['post_red_btn']['precision']
            post_red_per_species_recall = results_dict['Per_species_presence'][test_id]['post_red_btn']['recall']
            post_red_per_species_f1_score = results_dict['Per_species_presence'][test_id]['post_red_btn']['f1_score']

            test_post_red_per_species_auc = results_dict['Per_species_presence'][test_id]['test_post_red_btn']['auc']
            test_post_red_per_species_precision = results_dict['Per_species_presence'][test_id]['test_post_red_btn']['precision']
            test_post_red_per_species_recall = results_dict['Per_species_presence'][test_id]['test_post_red_btn']['recall']
            test_post_red_per_species_f1_score = results_dict['Per_species_presence'][test_id]['test_post_red_btn']['f1_score']

            species_id2name_mapping = results_dict['Species_id2name'][test_id]
            species_name2id_mapping = dict((v,k) for k,v in species_id2name_mapping.items())

            summary.write(f'\n                              ---------------  Pre  ------------------   |     ---------------  Post  ------------------    |    ----------------  Test  ------------------------    \n')
            summary.write(f'                                    AUC                      F1            |         AUC                        F1            |        AUC                        F1                  \n')
            summary.write(f'{test_id}: \n')

            species_present = list(set(
                list(pre_red_per_species_auc.keys()) + 
                list(post_red_per_species_auc.keys()) + 
                list(test_post_red_per_species_auc.keys())
                ))
            species_present_ids = [int(x.split('_')[-1]) if 'unknown' in x else species_name2id_mapping[x] for x in species_present]
            idx_sort_spe_present = np.argsort(species_present_ids)

            for i in idx_sort_spe_present:
                spe_name = species_present[i]
                if 'unknown' in spe_name:
                    species_name = 'unknown'
                    spe_id = int(spe_name.split('_')[-1])
                else:
                    # this is an empty images
                    species_name = spe_name
                    spe_id = species_name2id_mapping[spe_name]
                
                # pre red button
                pre_red_per_spe_auc, pre_red_per_spe_f1 = -1, -1
                pre_red_per_spe_auc_ci, pre_red_per_spe_f1_ci = '--', '--'
                
                if spe_name in pre_red_per_species_auc.keys():
                    pre_red_per_spe_auc = round(pre_red_per_species_auc[spe_name]['value'], 2)
                    if 'ci' in pre_red_per_species_auc[spe_name]:
                        pre_red_per_spe_auc_ci = ci_string(pre_red_per_species_auc[spe_name]['ci'])

                    pre_red_per_spe_f1 = round(pre_red_per_species_f1_score[spe_name]['value'], 2)
                    if 'ci' in pre_red_per_species_f1_score[spe_name]:
                        pre_red_per_spe_f1_ci = ci_string(pre_red_per_species_f1_score[spe_name]['ci'])

                # post red button
                post_red_per_spe_auc, post_red_per_spe_f1 = -1, -1
                post_red_per_spe_auc_ci, post_red_per_spe_f1_ci = '--', '--'
                
                if spe_name in post_red_per_species_auc.keys():
                    post_red_per_spe_auc = round(post_red_per_species_auc[spe_name]['value'], 2)
                    if 'ci' in post_red_per_species_auc[spe_name]:
                        post_red_per_spe_auc_ci = ci_string(post_red_per_species_auc[spe_name]['ci'])

                    post_red_per_spe_f1 = round(post_red_per_species_f1_score[spe_name]['value'], 2)
                    if 'ci' in post_red_per_species_f1_score[spe_name]:
                        post_red_per_spe_f1_ci = ci_string(post_red_per_species_f1_score[spe_name]['ci'])

                # post red button test phase
                test_post_red_per_spe_auc, test_post_red_per_spe_f1 = -1, -1
                test_post_red_per_spe_auc_ci, test_post_red_per_spe_f1_ci = '--', '--'
                
                if spe_name in test_post_red_per_species_auc.keys():
                    test_post_red_per_spe_auc = round(test_post_red_per_species_auc[spe_name]['value'], 2)
                    if 'ci' in test_post_red_per_species_auc[spe_name]:
                        test_post_red_per_spe_auc_ci = ci_string(test_post_red_per_species_auc[spe_name]['ci'])

                    test_post_red_per_spe_f1 = round(test_post_red_per_species_f1_score[spe_name]['value'], 2)
                    if 'ci' in test_post_red_per_species_f1_score[spe_name]:
                        test_post_red_per_spe_f1_ci = ci_string(test_post_red_per_species_f1_score[spe_name]['ci'])


                summary.write(f'      {species_name:15} ({spe_id}):   '
                  f'{pre_red_per_spe_auc:5} {pre_red_per_spe_auc_ci:12}   {pre_red_per_spe_f1:5} {pre_red_per_spe_f1_ci:12}         '
                  f'{post_red_per_spe_auc:5} {post_red_per_spe_auc_ci:12}   {post_red_per_spe_f1:5} {post_red_per_spe_f1_ci:12}         '
                  f'{test_post_red_per_spe_auc:5} {test_post_red_per_spe_auc_ci:12}   {test_post_red_per_spe_f1:5} {test_post_red_per_spe_f1_ci:12}      \n')
                
            summary.write(f"{'-'*100}\n")
        # ****
        
        summary.write(f"{'_'*150}\n\n")

        # Per activity presence
        summary.write(f'                        *********  Per Activity presence  *********  \n')

        for test_id in test_ids:
            pre_red_per_activity_auc = results_dict['Per_activity_presence'][test_id]['pre_red_btn']['auc']
            pre_red_per_activity_precision = results_dict['Per_activity_presence'][test_id]['pre_red_btn']['precision']
            pre_red_per_activity_recall = results_dict['Per_activity_presence'][test_id]['pre_red_btn']['recall']
            pre_red_per_activity_f1_score = results_dict['Per_activity_presence'][test_id]['pre_red_btn']['f1_score']

            post_red_per_activity_auc = results_dict['Per_activity_presence'][test_id]['post_red_btn']['auc']
            post_red_per_activity_precision = results_dict['Per_activity_presence'][test_id]['post_red_btn']['precision']
            post_red_per_activity_recall = results_dict['Per_activity_presence'][test_id]['post_red_btn']['recall']
            post_red_per_activity_f1_score = results_dict['Per_activity_presence'][test_id]['post_red_btn']['f1_score']

            test_post_red_per_activity_auc = results_dict['Per_activity_presence'][test_id]['test_post_red_btn']['auc']
            test_post_red_per_activity_precision = results_dict['Per_activity_presence'][test_id]['test_post_red_btn']['precision']
            test_post_red_per_activity_recall = results_dict['Per_activity_presence'][test_id]['test_post_red_btn']['recall']
            test_post_red_per_activity_f1_score = results_dict['Per_activity_presence'][test_id]['test_post_red_btn']['f1_score']

            activity_id2name_mapping = results_dict['Activity_id2name'][test_id]
            activity_name2id_mapping = dict((v,k) for k,v in activity_id2name_mapping.items())


            summary.write(f'\n                              ---------------  Pre  ------------------  |    ---------------  Post  ----------------    |       -------------  Test  ------------------------    \n')
            summary.write(f'                                    AUC                      F1           |        AUC                        F1          |         AUC                        F1                  \n')
            summary.write(f'{test_id}: \n')

            present_activities = list(set(
                list(pre_red_per_activity_auc.keys()) + 
                list(post_red_per_activity_auc.keys()) + 
                list(test_post_red_per_activity_auc.keys())
                ))
            activities_present_ids = [int(x.split('_')[-1]) if 'unknown' in x else activity_name2id_mapping[x] for x in present_activities]
            idx_sort_act_present = np.argsort(activities_present_ids)

            for i in idx_sort_act_present:
                act_name = present_activities[i]
                if 'unknown' in act_name:
                    activity_name = 'unknown'
                    act_id = int(act_name.split('_')[-1])
                else:
                    activity_name = act_name
                    act_id = activity_name2id_mapping[act_name]

                # pre red button
                pre_red_per_act_auc, pre_red_per_act_f1 = -1, -1
                pre_red_per_act_auc_ci, pre_red_per_act_f1_ci = '--', '--'

                if act_name in pre_red_per_activity_auc.keys():
                    pre_red_per_act_auc = round(pre_red_per_activity_auc[act_name]['value'], 2)
                    if 'ci' in pre_red_per_activity_auc[act_name]:
                        pre_red_per_act_auc_ci = ci_string(pre_red_per_activity_auc[act_name]['ci'])

                    pre_red_per_act_f1 = round(pre_red_per_activity_f1_score[act_name]['value'], 2)
                    if 'ci' in pre_red_per_activity_f1_score[act_name]:
                        pre_red_per_act_f1_ci = ci_string(pre_red_per_activity_f1_score[act_name]['ci'])

                # post red button
                post_red_per_act_auc, post_red_per_act_f1 = -1, -1
                post_red_per_act_auc_ci, post_red_per_act_f1_ci = '--', '--'

                if act_name in post_red_per_activity_auc.keys():
                    post_red_per_act_auc = round(post_red_per_activity_auc[act_name]['value'], 2)
                    if 'ci' in post_red_per_activity_auc[act_name]:
                        post_red_per_act_auc_ci = ci_string(post_red_per_activity_auc[act_name]['ci'])

                    post_red_per_act_f1 = round(post_red_per_activity_f1_score[act_name]['value'], 2)
                    if 'ci' in post_red_per_activity_f1_score[act_name]:
                        post_red_per_act_f1_ci = ci_string(post_red_per_activity_f1_score[act_name]['ci'])

                # post red button test phase
                test_post_red_per_act_auc, test_post_red_per_act_f1 = -1, -1
                test_post_red_per_act_auc_ci, test_post_red_per_act_f1_ci = '--', '--'

                if act_name in test_post_red_per_activity_auc.keys():
                    test_post_red_per_act_auc = round(test_post_red_per_activity_auc[act_name]['value'], 2)
                    if 'ci' in test_post_red_per_activity_auc[act_name]:
                        test_post_red_per_act_auc_ci = ci_string(test_post_red_per_activity_auc[act_name]['ci'])

                    test_post_red_per_act_f1 = round(test_post_red_per_activity_f1_score[act_name]['value'], 2)
                    if 'ci' in test_post_red_per_activity_f1_score[act_name]:
                        test_post_red_per_act_f1_ci = ci_string(test_post_red_per_activity_f1_score[act_name]['ci'])

                summary.write(f'       {activity_name:10} ({act_id}): '
                  f'{pre_red_per_act_auc:5} {pre_red_per_act_auc_ci:12}   {pre_red_per_act_f1:5} {pre_red_per_act_f1_ci:12}         '
                  f'{post_red_per_act_auc:5}  {post_red_per_act_auc_ci:12}   {post_red_per_act_f1:5} {post_red_per_act_f1_ci:12}         '
                  f'{test_post_red_per_act_auc:5}  {test_post_red_per_act_auc_ci:12}   {test_post_red_per_act_f1:5} {test_post_red_per_act_f1_ci:12}       \n')
                
            summary.write(f"{'-'*60}\n")


def print_confusion_matrices(results_dict, out_file):
    test_ids = sorted(list(results_dict['Red_button'].keys()))
    phases = ['pre_red_btn', 'post_red_btn', 'test_post_red_btn']
    phases_abrv = ['pre_rb', 'pos_rb', 'test']
    with PdfPages(out_file) as pdf_pages:
        # plot confusion matrix of species
        for test_id in test_ids:
            for i, phase in enumerate(phases):
                species_cm = results_dict['Confusion_matrices'][test_id]['species'][phase]['cm']
                species_ids = results_dict['Confusion_matrices'][test_id]['species'][phase]['species_ids']

                if species_cm is None or species_ids is None:
                    continue

                species_id2name_mapping = results_dict['Species_id2name'][test_id]

                species_names = [species_id2name_mapping[j] for j in sorted(species_ids)]


                # Print count version
                
                fig = plt.figure('', figsize=[11, 8])
                # ax = plt.axes()
                ax = fig.add_subplot(111)
                sns.heatmap(species_cm,
                            annot=True,
                            fmt="d",
                            cmap='Blues',)
                ax.xaxis.set_ticklabels(species_names, fontsize=8, rotation=45)
                ax.yaxis.set_ticklabels(species_names, fontsize=8,  rotation='horizontal')
                ax.set_ylabel("True label")
                ax.tick_params(axis='x', which='major', labelbottom=False, bottom=False, top=True, labeltop=True)
                ax.set_title(f"{test_id}: {phases_abrv[i]} species CM")
                fig.tight_layout()
                pdf_pages.savefig()
                plt.close()
                '''

                fig = plt.figure('pre red button', figsize=[11, 8])
                # ax = plt.axes()
                ax = fig.add_subplot(211)
                sns.heatmap(pre_red_species_cm,
                            annot=True,
                            fmt="d",
                            cmap='Blues',)
                ax.xaxis.set_ticklabels(pre_red_species_names, fontsize=8, rotation=45)
                ax.yaxis.set_ticklabels(pre_red_species_names, fontsize=8,  rotation='horizontal')
                ax.set_ylabel("True label")
                ax.set_title(f"{test_id}: pre red btn species confusion matrix")
                ax.tick_params(axis='x', which='major', labelbottom=False, bottom=False, top=True, labeltop=True)

                # fig = plt.figure('post red button', figsize=[11, 8])
                # ax = plt.axes()
                ax = fig.add_subplot(212)
                sns.heatmap(post_red_species_cm,
                            annot=True,
                            fmt="d",
                            cmap='Blues',)
                ax.xaxis.set_ticklabels(post_red_species_names, fontsize=8, rotation=45)
                ax.yaxis.set_ticklabels(post_red_species_names, fontsize=8, rotation='horizontal')
                ax.set_ylabel("True label")
                ax.set_title(f"{test_id}: post red btn species confusion matrix")
                ax.tick_params(axis='x', which='major', labelbottom=False, bottom=False, top=True, labeltop=True)
                fig.tight_layout()
                pdf_pages.savefig()
                plt.close()
                '''

        # plot confusion matrix of activities
        for test_id in test_ids:
            for i, phase in enumerate(phases):
                activity_cm = results_dict['Confusion_matrices'][test_id]['activity'][phase]['cm']
                activity_ids = results_dict['Confusion_matrices'][test_id]['activity'][phase]['activity_ids']

                if activity_cm is None or activity_ids is None:
                    continue

                activity_id2name_mapping = results_dict['Activity_id2name'][test_id]

                # activity_names = [activity_id2name_mapping[j] for j in sorted(activity_ids)]
                activity_names = []
                for j in sorted(activity_ids):
                    if j in activity_id2name_mapping.keys():
                        activity_names.append(activity_id2name_mapping[j])
                    else:
                        activity_names.append('activity '+str(j))

                # Print count version
                
                fig = plt.figure('', figsize=[11, 8])
                # ax = plt.axes()
                ax = fig.add_subplot(111)
                sns.heatmap(activity_cm,
                            annot=True,
                            fmt="d",
                            cmap='Blues',)
                ax.xaxis.set_ticklabels(activity_names, fontsize=8, rotation=45)
                ax.yaxis.set_ticklabels(activity_names, fontsize=8,  rotation='horizontal')
                ax.set_ylabel("True label")
                ax.tick_params(axis='x', which='major', labelbottom=False, bottom=False, top=True, labeltop=True)
                ax.set_title(f"{test_id}: {phases_abrv[i]} activities CM")
                fig.tight_layout()
                pdf_pages.savefig()
                plt.close()


def print_reliability_diagrams(results_dict, out_file):
    test_ids = sorted(list(results_dict['Red_button'].keys()))
    phases = ['pre_red_btn', 'post_red_btn', 'test_post_red_btn']
    phases_abrv = ['pre_rb', 'pos_rb', 'test']
    num_bins = 40
    with PdfPages(out_file) as pdf_pages:
        # plot confusion matrix of species
        for test_id in test_ids:
            for i, phase in enumerate(phases):
                y_true = results_dict['Prediction_confidence'][test_id]['species'][phase]['ground_true']
                pred_prob = results_dict['Prediction_confidence'][test_id]['species'][phase]['confidence']

                if y_true is None or pred_prob is None:
                    continue
                
                fig = plt.figure('', figsize=[10, 10])
                # ax = plt.axes()
                # ax = fig.add_subplot(111)
                ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
                ax2 = plt.subplot2grid((3, 1), (2, 0))

                prob_true, prob_pred = calibration_curve(y_true, pred_prob, n_bins=num_bins)
                ax1.plot(prob_pred, prob_true, "s-")
                ax1.axline([0, 0], [1, 1], color='gray', ls='--')

                ax2.hist(pred_prob, range=(0, 1), bins=num_bins, histtype="step", lw=2)

                ax1.set_ylabel("Fraction of positives")
                ax1.set_xlabel("Mean predicted value")
                ax1.set_title(f"{test_id}: {phases_abrv[i]} species (Reliability Curve)")

                ax2.set_xlabel("Mean predicted value")
                ax2.set_ylabel("Count")
                # ax2.legend(loc="upper center", ncol=2)

                fig.tight_layout()
                pdf_pages.savefig()
                plt.close()

        # plot confusion matrix of activities
        for test_id in test_ids:
            for i, phase in enumerate(phases):
                y_true = results_dict['Prediction_confidence'][test_id]['activity'][phase]['ground_true']
                pred_prob = results_dict['Prediction_confidence'][test_id]['activity'][phase]['confidence']

                if y_true is None or pred_prob is None:
                    continue
                
                fig = plt.figure('', figsize=[10, 10])
                # ax = plt.axes()
                ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
                ax2 = plt.subplot2grid((3, 1), (2, 0))

                prob_true, prob_pred = calibration_curve(y_true, pred_prob, n_bins=num_bins)
                ax1.plot(prob_pred, prob_true, "s-")
                ax1.axline([0, 0], [1, 1], color='gray', ls='--')

                ax2.hist(pred_prob, range=(0, 1), bins=num_bins, histtype="step", lw=2)
                
                ax1.set_ylabel("Fraction of positives")
                ax1.set_xlabel("Mean predicted value")
                ax1.set_title(f"{test_id}: {phases_abrv[i]} activities (Reliability Curve)")

                ax2.set_xlabel("Mean predicted value")
                ax2.set_ylabel("Count")
                # ax2.legend(loc="upper center", ncol=2)

                fig.tight_layout()
                pdf_pages.savefig()
                plt.close()


def __print_reliability_diagrams(results_dict, out_file):
    test_ids = sorted(list(results_dict['Red_button'].keys()))
    phases = ['pre_red_btn', 'post_red_btn', 'test_post_red_btn']
    phases_abrv = ['pre_rb', 'pos_rb', 'test']
    with PdfPages(out_file) as pdf_pages:
        # plot confusion matrix of species
        for test_id in test_ids:
            for i, phase in enumerate(phases):
                y_true = results_dict['Prediction_confidence'][test_id]['species'][phase]['ground_true']
                y_pred = results_dict['Prediction_confidence'][test_id]['species'][phase]['pred_class']
                y_conf = results_dict['Prediction_confidence'][test_id]['species'][phase]['confidence']

                if y_true is None or y_pred is None:
                    continue

                fig = reliability_diagram(y_true, y_pred, y_conf, num_bins=10, draw_ece=True,
                          draw_bin_importance="alpha", draw_averages=True,
                          title=f"{test_id}: {phases_abrv[i]} species", figsize=(6, 6), dpi=100, 
                          return_fig=True)
                fig.tight_layout()
                pdf_pages.savefig()
                plt.close()

        # plot confusion matrix of activities
        for test_id in test_ids:
            for i, phase in enumerate(phases):
                y_true = results_dict['Prediction_confidence'][test_id]['activity'][phase]['ground_true']
                y_pred = results_dict['Prediction_confidence'][test_id]['activity'][phase]['pred_class']
                y_conf = results_dict['Prediction_confidence'][test_id]['activity'][phase]['confidence']

                if y_true is None or y_pred is None:
                    continue

                fig = reliability_diagram(y_true, y_pred, y_conf, num_bins=10, draw_ece=True,
                          draw_bin_importance="alpha", draw_averages=True,
                          title=f"{test_id}: {phases_abrv[i]} activities", figsize=(6, 6), dpi=100, 
                          return_fig=True)
                fig.tight_layout()
                pdf_pages.savefig()
                plt.close()


