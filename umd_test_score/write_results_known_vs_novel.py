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
    output = [['Novelty type', 'Red button', 'Declared', 'Result', 'Delay', 'Total', 'Novel', '', '', '', '', '', '']]
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
        ['', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', '', '', '', '', '', '']
    ]


    # Aggregate species presence
    output += [
        ['Average Species Counts', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['', 'Pre-novelty', 'Pre-novelty', '', 'Post-novelty', 'Post-novelty', 'Post-novelty', 'Post-novelty', '', 'Test', 'Test', 'Test', 'Test'],
        ['', 'Average MAE', 'Average MRE', '', 'Average MAE (known)', 'Average MAE (novel)', 'Average MRE (known)', 'Average MRE (novel)', '', 'Average MAE (known)', 'Average MAE (novel)', 'Average MRE (known)', 'Average MRE (novel)']
    ]
    for test_id in test_ids:
        # pre red button
        pre_red_avg_spe_abs_err = results_dict['Aggregate_species_counts'][test_id]['pre_red_btn']['avg_abs_err']['known']['value']
        if 'ci' in results_dict['Aggregate_species_counts'][test_id]['pre_red_btn']['avg_abs_err']['known']:
            pre_red_avg_spe_abs_err_ci = ci_string(results_dict['Aggregate_species_counts'][test_id]['pre_red_btn']['avg_abs_err']['known']['ci'])

        pre_red_avg_spe_rel_err = results_dict['Aggregate_species_counts'][test_id]['pre_red_btn']['avg_rel_err']['known']['value']
        if 'ci' in results_dict['Aggregate_species_counts'][test_id]['pre_red_btn']['avg_rel_err']['known']:
            pre_red_avg_spe_rel_err_ci = ci_string(results_dict['Aggregate_species_counts'][test_id]['pre_red_btn']['avg_rel_err']['known']['ci'])

        # post red button
        # ** known **
        post_red_avg_spe_abs_err_known = results_dict['Aggregate_species_counts'][test_id]['post_red_btn']['avg_abs_err']['known']['value']
        if 'ci' in results_dict['Aggregate_species_counts'][test_id]['post_red_btn']['avg_abs_err']['known']:
            post_red_avg_spe_abs_err_ci_known = ci_string(results_dict['Aggregate_species_counts'][test_id]['post_red_btn']['avg_abs_err']['known']['ci'])

        post_red_avg_spe_rel_err_known = results_dict['Aggregate_species_counts'][test_id]['post_red_btn']['avg_rel_err']['known']['value']
        if 'ci' in results_dict['Aggregate_species_counts'][test_id]['post_red_btn']['avg_rel_err']['known']:
            post_red_avg_spe_rel_err_ci_known = ci_string(results_dict['Aggregate_species_counts'][test_id]['post_red_btn']['avg_rel_err']['known']['ci'])

        # ** novel **
        post_red_avg_spe_abs_err_novel = results_dict['Aggregate_species_counts'][test_id]['post_red_btn']['avg_abs_err']['novel']['value']
        if 'ci' in results_dict['Aggregate_species_counts'][test_id]['post_red_btn']['avg_abs_err']['novel']:
            post_red_avg_spe_abs_err_ci_novel = ci_string(results_dict['Aggregate_species_counts'][test_id]['post_red_btn']['avg_abs_err']['novel']['ci'])

        post_red_avg_spe_rel_err_novel = results_dict['Aggregate_species_counts'][test_id]['post_red_btn']['avg_rel_err']['novel']['value']
        if 'ci' in results_dict['Aggregate_species_counts'][test_id]['post_red_btn']['avg_rel_err']['novel']:
            post_red_avg_spe_rel_err_ci_novel = ci_string(results_dict['Aggregate_species_counts'][test_id]['post_red_btn']['avg_rel_err']['novel']['ci'])

        # post red button test phase
        # ** known **
        test_post_red_avg_spe_abs_err_known = results_dict['Aggregate_species_counts'][test_id]['test_post_red_btn']['avg_abs_err']['known']['value']
        if 'ci' in results_dict['Aggregate_species_counts'][test_id]['test_post_red_btn']['avg_abs_err']['known']:
            test_post_red_avg_spe_abs_err_ci_known = ci_string(results_dict['Aggregate_species_counts'][test_id]['test_post_red_btn']['avg_abs_err']['known']['ci'])

        test_post_red_avg_spe_rel_err_known = results_dict['Aggregate_species_counts'][test_id]['test_post_red_btn']['avg_rel_err']['known']['value']
        if 'ci' in results_dict['Aggregate_species_counts'][test_id]['test_post_red_btn']['avg_rel_err']['known']:
            test_post_red_avg_spe_rel_err_ci_known = ci_string(results_dict['Aggregate_species_counts'][test_id]['test_post_red_btn']['avg_rel_err']['known']['ci'])

        # ** novel **
        test_post_red_avg_spe_abs_err_novel = results_dict['Aggregate_species_counts'][test_id]['test_post_red_btn']['avg_abs_err']['novel']['value']
        if 'ci' in results_dict['Aggregate_species_counts'][test_id]['test_post_red_btn']['avg_abs_err']['novel']:
            test_post_red_avg_spe_abs_err_ci_novel = ci_string(results_dict['Aggregate_species_counts'][test_id]['test_post_red_btn']['avg_abs_err']['novel']['ci'])

        test_post_red_avg_spe_rel_err_novel = results_dict['Aggregate_species_counts'][test_id]['test_post_red_btn']['avg_rel_err']['novel']['value']
        if 'ci' in results_dict['Aggregate_species_counts'][test_id]['test_post_red_btn']['avg_rel_err']['novel']:
            test_post_red_avg_spe_rel_err_ci_novel = ci_string(results_dict['Aggregate_species_counts'][test_id]['test_post_red_btn']['avg_rel_err']['novel']['ci'])

        output.append(
            [
                test_id, 
                f'{pre_red_avg_spe_abs_err} {pre_red_avg_spe_abs_err_ci}', 
                f'{pre_red_avg_spe_rel_err} {pre_red_avg_spe_rel_err_ci}', 
                '', 
                f'{post_red_avg_spe_abs_err_known} {post_red_avg_spe_abs_err_ci_known}', 
                f'{post_red_avg_spe_abs_err_novel} {post_red_avg_spe_abs_err_ci_novel}', 
                f'{post_red_avg_spe_rel_err_known} {post_red_avg_spe_rel_err_ci_known}',
                f'{post_red_avg_spe_rel_err_novel} {post_red_avg_spe_rel_err_ci_novel}', 
                '', 
                f'{test_post_red_avg_spe_abs_err_known} {test_post_red_avg_spe_abs_err_ci_known}', 
                f'{test_post_red_avg_spe_abs_err_novel} {test_post_red_avg_spe_abs_err_ci_novel}', 
                f'{test_post_red_avg_spe_rel_err_known} {test_post_red_avg_spe_rel_err_ci_known}',
                f'{test_post_red_avg_spe_rel_err_novel} {test_post_red_avg_spe_rel_err_ci_novel}'
            ]
        )

    output += [
        ['', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', '', '', '', '', '', '']
    ]

        
    # Aggregate species presence
    output += [
        ['Average Species Presence', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['', 'Pre-novelty', 'Pre-novelty', '', 'Post-novelty', 'Post-novelty', 'Post-novelty', 'Post-novelty', '', 'Test', 'Test', 'Test', 'Test'],
        ['', 'Average AUC', 'Average F1', '', 'Average AUC (known)', 'Average AUC (novel)', 'Average F1(known)', 'Average F1(novel)', '', 'Average AUC (known)', 'Average AUC (novel)', 'Average F1 (known)', 'Average F1 (novel)']
    ]

    for test_id in test_ids:
        # pre red button
        pre_red_species_avg_auc_ci, pre_red_species_avg_f1_ci, pre_red_avg_spe_abs_err_ci, pre_red_avg_spe_rel_err_ci = '--', '--', '--', '--'

        pre_red_species_avg_auc = results_dict['Aggregate_species_presence'][test_id]['pre_red_btn']['avg_auc']['known']['value']
        if 'ci' in results_dict['Aggregate_species_presence'][test_id]['pre_red_btn']['avg_auc']['known']:
            pre_red_species_avg_auc_ci = ci_string(results_dict['Aggregate_species_presence'][test_id]['pre_red_btn']['avg_auc']['known']['ci'])

        # pre_red_species_avg_precision = results_dict['Aggregate_species_presence'][test_id]['pre_red_btn']['known']['avg_precision']
        # pre_red_species_avg_recall = results_dict['Aggregate_species_presence'][test_id]['pre_red_btn']['known']['avg_recall']

        pre_red_species_avg_f1 = results_dict['Aggregate_species_presence'][test_id]['pre_red_btn']['avg_f1_score']['known']['value']
        if 'ci' in results_dict['Aggregate_species_presence'][test_id]['pre_red_btn']['avg_f1_score']['known']:
            pre_red_species_avg_f1_ci = ci_string(results_dict['Aggregate_species_presence'][test_id]['pre_red_btn']['avg_f1_score']['known']['ci'])


        # post red button
        # ** known **
        post_red_species_avg_auc_ci_known, post_red_species_avg_f1_ci_known = '--', '--'
        post_red_avg_spe_abs_err_ci_known, post_red_avg_spe_rel_err_ci_known = '--', '--'

        post_red_species_avg_auc_known = results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['avg_auc']['known']['value']
        if 'ci' in results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['avg_auc']['known']:
            post_red_species_avg_auc_ci_known = ci_string(results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['avg_auc']['known']['ci'])

        post_red_species_avg_f1_known = results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['avg_f1_score']['known']['value']
        if 'ci' in results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['avg_f1_score']['known']:
            post_red_species_avg_f1_ci_known = ci_string(results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['avg_f1_score']['known']['ci'])
        
        # ** novel **
        post_red_species_avg_auc_ci_novel, post_red_species_avg_f1_ci_novel = '--', '--'
        post_red_avg_spe_abs_err_ci_novel, post_red_avg_spe_rel_err_ci_novel = '--', '--'

        post_red_species_avg_auc_novel = results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['avg_auc']['novel']['value']
        if 'ci' in results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['avg_auc']['novel']:
            post_red_species_avg_auc_ci_novel = ci_string(results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['avg_auc']['novel']['ci'])

        post_red_species_avg_f1_novel = results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['avg_f1_score']['novel']['value']
        if 'ci' in results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['avg_f1_score']['novel']:
            post_red_species_avg_f1_ci_novel = ci_string(results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['avg_f1_score']['novel']['ci'])


        # post red button test phase
        # ** known **
        test_post_red_species_avg_auc_ci_known, test_post_red_species_avg_f1_ci_known = '--', '--'
        test_post_red_avg_spe_abs_err_ci_known, test_post_red_avg_spe_rel_err_ci_known = '--', '--'

        test_post_red_species_avg_auc_known = results_dict['Aggregate_species_presence'][test_id]['test_post_red_btn']['avg_auc']['known']['value']
        if 'ci' in results_dict['Aggregate_species_presence'][test_id]['test_post_red_btn']['avg_auc']['known']:
            test_post_red_species_avg_auc_ci_known = ci_string(results_dict['Aggregate_species_presence'][test_id]['test_post_red_btn']['avg_auc']['known']['ci'])

        test_post_red_species_avg_f1_known = results_dict['Aggregate_species_presence'][test_id]['test_post_red_btn']['avg_f1_score']['known']['value']
        if 'ci' in results_dict['Aggregate_species_presence'][test_id]['test_post_red_btn']['avg_f1_score']['known']:
            test_post_red_species_avg_f1_ci_known = ci_string(results_dict['Aggregate_species_presence'][test_id]['test_post_red_btn']['avg_f1_score']['known']['ci'])

        # ** novel **
        test_post_red_species_avg_auc_ci_novel, test_post_red_species_avg_f1_ci_novel = '--', '--'
        test_post_red_avg_spe_abs_err_ci_novel, test_post_red_avg_spe_rel_err_ci_novel = '--', '--'

        test_post_red_species_avg_auc_novel = results_dict['Aggregate_species_presence'][test_id]['test_post_red_btn']['avg_auc']['novel']['value']
        if 'ci' in results_dict['Aggregate_species_presence'][test_id]['test_post_red_btn']['avg_auc']['novel']:
            test_post_red_species_avg_auc_ci_novel = ci_string(results_dict['Aggregate_species_presence'][test_id]['test_post_red_btn']['avg_auc']['novel']['ci'])

        test_post_red_species_avg_f1_novel = results_dict['Aggregate_species_presence'][test_id]['test_post_red_btn']['avg_f1_score']['novel']['value']
        if 'ci' in results_dict['Aggregate_species_presence'][test_id]['test_post_red_btn']['avg_f1_score']['novel']:
            test_post_red_species_avg_f1_ci_novel = ci_string(results_dict['Aggregate_species_presence'][test_id]['test_post_red_btn']['avg_f1_score']['novel']['ci'])
        
        output.append(
            [
                test_id, 
                f'{pre_red_species_avg_auc} {pre_red_species_avg_auc_ci}', 
                f'{pre_red_species_avg_f1} {pre_red_species_avg_f1_ci}', 
                '', 
                f'{post_red_species_avg_auc_known} {post_red_species_avg_auc_ci_known}',
                f'{post_red_species_avg_auc_novel} {post_red_species_avg_auc_ci_novel}', 
                f'{post_red_species_avg_f1_known} {post_red_species_avg_f1_ci_known}',  
                f'{post_red_species_avg_f1_novel} {post_red_species_avg_f1_ci_novel}', 
                '', 
                f'{test_post_red_species_avg_auc_known} {test_post_red_species_avg_auc_ci_known}',
                f'{test_post_red_species_avg_auc_novel} {test_post_red_species_avg_auc_ci_novel}', 
                f'{test_post_red_species_avg_f1_known} {test_post_red_species_avg_f1_ci_known}', 
                f'{test_post_red_species_avg_f1_novel} {test_post_red_species_avg_f1_ci_novel}'
            ]
        )

    output += [
        ['', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', '', '', '', '', '', '']
    ]


    # Aggregate activity presence
    output += [
        ['Average Activity Presence', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['', 'Pre-novelty', 'Pre-novelty', '', 'Post-novelty', 'Post-novelty', 'Post-novelty', 'Post-novelty', '', 'Test', 'Test', 'Test', 'Test'],
        ['', 'Average AUC', 'Average F1', '', 'Average AUC (known)', 'Average AUC (novel)', 'Average F1 (known)', 'Average F1 (novel)', '', 'Average AUC (known)', 'Average AUC (novel)', 'Average F1 (known)', 'Average F1 (novel)']
    ]

    # ** print average activity presence metrics
    for test_id in test_ids:
        # -- pre red button
        pre_red_activity_avg_auc_ci, pre_red_activity_avg_f1_ci = '--', '--'

        pre_red_activity_avg_auc = results_dict['Aggregate_activity_presence'][test_id]['pre_red_btn']['avg_auc']['known']['value']
        if 'ci' in results_dict['Aggregate_activity_presence'][test_id]['pre_red_btn']['avg_auc']['known']:
            pre_red_activity_avg_auc_ci = ci_string(results_dict['Aggregate_activity_presence'][test_id]['pre_red_btn']['avg_auc']['known']['ci'])

        pre_red_activity_avg_f1 = results_dict['Aggregate_activity_presence'][test_id]['pre_red_btn']['avg_f1_score']['known']['value']
        if 'ci' in results_dict['Aggregate_activity_presence'][test_id]['pre_red_btn']['avg_f1_score']['known']:
            pre_red_activity_avg_f1_ci = ci_string(results_dict['Aggregate_activity_presence'][test_id]['pre_red_btn']['avg_f1_score']['known']['ci'])

        # -- post red button
        # ** known **
        post_red_activity_avg_auc_ci_known, post_red_activity_avg_f1_ci_known = '--', '--'

        post_red_activity_avg_auc_known = results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_auc']['known']['value']
        if 'ci' in results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_auc']['known']:
            post_red_activity_avg_auc_ci_known = ci_string(results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_auc']['known']['ci'])

        post_red_activity_avg_f1_known = results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_f1_score']['known']['value']
        if 'ci' in results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_f1_score']['known']:
            post_red_activity_avg_f1_ci_known = ci_string(results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_f1_score']['known']['ci'])

        # ** novel **
        post_red_activity_avg_auc_ci_novel, post_red_activity_avg_f1_ci_novel = '--', '--'

        post_red_activity_avg_auc_novel = results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_auc']['novel']['value']
        if 'ci' in results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_auc']['novel']:
            post_red_activity_avg_auc_ci_novel = ci_string(results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_auc']['novel']['ci'])

        post_red_activity_avg_f1_novel = results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_f1_score']['novel']['value']
        if 'ci' in results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_f1_score']['novel']:
            post_red_activity_avg_f1_ci_novel = ci_string(results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_f1_score']['novel']['ci'])


        # -- post red button test phase
        # ** known **
        test_post_red_activity_avg_auc_ci_known, test_post_red_activity_avg_f1_ci_known = '--', '--'

        test_post_red_activity_avg_auc_known = results_dict['Aggregate_activity_presence'][test_id]['test_post_red_btn']['avg_auc']['known']['value']
        if 'ci' in results_dict['Aggregate_activity_presence'][test_id]['test_post_red_btn']['avg_auc']['known']:
            test_post_red_activity_avg_auc_ci_known = ci_string(results_dict['Aggregate_activity_presence'][test_id]['test_post_red_btn']['avg_auc']['known']['ci'])

        test_post_red_activity_avg_f1_known = results_dict['Aggregate_activity_presence'][test_id]['test_post_red_btn']['avg_f1_score']['known']['value']
        if 'ci' in results_dict['Aggregate_activity_presence'][test_id]['test_post_red_btn']['avg_f1_score']['known']:
            test_post_red_activity_avg_f1_ci_known = ci_string(results_dict['Aggregate_activity_presence'][test_id]['test_post_red_btn']['avg_f1_score']['known']['ci'])

        # ** novel **
        test_post_red_activity_avg_auc_ci_novel, test_post_red_activity_avg_f1_ci_novel = '--', '--'

        test_post_red_activity_avg_auc_novel = results_dict['Aggregate_activity_presence'][test_id]['test_post_red_btn']['avg_auc']['novel']['value']
        if 'ci' in results_dict['Aggregate_activity_presence'][test_id]['test_post_red_btn']['avg_auc']['novel']:
            test_post_red_activity_avg_auc_ci_novel = ci_string(results_dict['Aggregate_activity_presence'][test_id]['test_post_red_btn']['avg_auc']['novel']['ci'])

        test_post_red_activity_avg_f1_novel = results_dict['Aggregate_activity_presence'][test_id]['test_post_red_btn']['avg_f1_score']['novel']['value']
        if 'ci' in results_dict['Aggregate_activity_presence'][test_id]['test_post_red_btn']['avg_f1_score']['novel']:
            test_post_red_activity_avg_f1_ci_novel = ci_string(results_dict['Aggregate_activity_presence'][test_id]['test_post_red_btn']['avg_f1_score']['novel']['ci'])

        output.append(
            [
                test_id, 
                f'{pre_red_activity_avg_auc} {pre_red_activity_avg_auc_ci}', 
                f'{pre_red_activity_avg_f1} {pre_red_activity_avg_f1_ci}', 
                '', 
                f'{post_red_activity_avg_auc_known} {post_red_activity_avg_auc_ci_known}', 
                f'{post_red_activity_avg_auc_novel} {post_red_activity_avg_auc_ci_novel}', 
                f'{post_red_activity_avg_f1_known} {post_red_activity_avg_f1_ci_known}',
                f'{post_red_activity_avg_f1_novel} {post_red_activity_avg_f1_ci_novel}', 
                '', 
                f'{test_post_red_activity_avg_auc_known} {test_post_red_activity_avg_auc_ci_known}', 
                f'{test_post_red_activity_avg_auc_novel} {test_post_red_activity_avg_auc_ci_novel}', 
                f'{test_post_red_activity_avg_f1_known} {test_post_red_activity_avg_f1_ci_known}',
                f'{test_post_red_activity_avg_f1_novel} {test_post_red_activity_avg_f1_ci_novel}'
            ]
        )
  
    output += [
        ['', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', '', '', '', '', '', '']
    ]

    # Per species count errors
    output.append(['Per Species Counts', '', '', '', '', '', '', '', '', '', '', '', ''])
    for test_id in test_ids:
        # species counts
        # ***  known  ***
        pre_red_per_species_count_abs_err = results_dict['Species_counts'][test_id]['pre_red_btn']['abs_err']
        pre_red_per_species_count_rel_err = results_dict['Species_counts'][test_id]['pre_red_btn']['rel_err']

        post_red_per_species_count_abs_err = results_dict['Species_counts'][test_id]['post_red_btn']['abs_err']
        post_red_per_species_count_rel_err = results_dict['Species_counts'][test_id]['post_red_btn']['rel_err']

        test_post_red_per_species_count_abs_err = results_dict['Species_counts'][test_id]['test_post_red_btn']['abs_err']
        test_post_red_per_species_count_rel_err = results_dict['Species_counts'][test_id]['test_post_red_btn']['rel_err']

        # ***  novel  ***
        # post_red_per_species_count_abs_err = results_dict['Species_counts'][test_id]['post_red_btn']['abs_err']
        # post_red_per_species_count_rel_err = results_dict['Species_counts'][test_id]['post_red_btn']['rel_err']

        # test_post_red_per_species_count_abs_err_novel = results_dict['Species_counts'][test_id]['test_post_red_btn']['abs_err']['novel']
        # test_post_red_per_species_count_rel_err_novel = results_dict['Species_counts'][test_id]['test_post_red_btn']['rel_err']['novel']


        species_id2name_mapping = results_dict['Species_id2name'][test_id]
        species_name2id_mapping = dict((v,k) for k,v in species_id2name_mapping.items())

        output += [
            [f'{test_id}', '', '', '', '', '', '', '', '', '', '', '', ''],
            ['', 'Pre-novelty', 'Pre-novelty', '', 'Post-novelty', 'Post-novelty', 'Post-novelty', 'Post-novelty', '', 'Test', 'Test', 'Test', 'Test'],
            ['', 'Mean Abs Err', 'Mean Rel Err', '', 'MAE (known)', 'MAE (novel)', 'MRE (known)', 'MRE (novel)', '', 'MAE (known)', 'MAE (novel)', 'MRE (known)', 'MRE (novel)']
        ]

        species_present = list(
            set(
                list(pre_red_per_species_count_abs_err.keys()) + 
                list(post_red_per_species_count_abs_err.keys()) + 
                list(test_post_red_per_species_count_abs_err.keys())
            )
        )
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
            # ** known **
            pre_red_per_spe_cnt_abs_err_ci, pre_red_per_spe_cnt_rel_err_ci = '--', '--'
            pre_red_per_spe_cnt_abs_err = round(pre_red_per_species_count_abs_err[spe_name]['known']['value'], 2)
            if 'ci' in pre_red_per_species_count_abs_err[spe_name]['known']:
                pre_red_per_spe_cnt_abs_err_ci = ci_string(pre_red_per_species_count_abs_err[spe_name]['known']['ci'])

            pre_red_per_spe_cnt_rel_err = round(pre_red_per_species_count_rel_err[spe_name]['known']['value'], 2)
            if 'ci' in pre_red_per_species_count_rel_err[spe_name]['known']:
                pre_red_per_spe_cnt_rel_err_ci = ci_string(pre_red_per_species_count_rel_err[spe_name]['known']['ci'])


            # post red button
            # ** known **
            post_red_per_spe_cnt_abs_err_ci_known, post_red_per_spe_cnt_rel_err_ci_known = '--', '--'

            post_red_per_spe_cnt_abs_err_known = round(post_red_per_species_count_abs_err[spe_name]['known']['value'], 2)
            if 'ci' in post_red_per_species_count_abs_err[spe_name]['known']:
                post_red_per_spe_cnt_abs_err_ci_known = ci_string(post_red_per_species_count_abs_err[spe_name]['known']['ci'])

            post_red_per_spe_cnt_rel_err_known = round(post_red_per_species_count_rel_err[spe_name]['known']['value'], 2)
            if 'ci' in post_red_per_species_count_rel_err[spe_name]['known']:
                post_red_per_spe_cnt_rel_err_ci_known = ci_string(post_red_per_species_count_rel_err[spe_name]['known']['ci'])

            # ** novel **
            post_red_per_spe_cnt_abs_err_ci_novel, post_red_per_spe_cnt_rel_err_ci_novel = '--', '--'

            post_red_per_spe_cnt_abs_err_novel = round(post_red_per_species_count_abs_err[spe_name]['novel']['value'], 2)
            if 'ci' in post_red_per_species_count_abs_err[spe_name]['novel']:
                post_red_per_spe_cnt_abs_err_ci_novel = ci_string(post_red_per_species_count_abs_err[spe_name]['novel']['ci'])

            post_red_per_spe_cnt_rel_err_novel = round(post_red_per_species_count_rel_err[spe_name]['novel']['value'], 2)
            if 'ci' in post_red_per_species_count_rel_err[spe_name]['novel']:
                post_red_per_spe_cnt_rel_err_ci_novel = ci_string(post_red_per_species_count_rel_err[spe_name]['novel']['ci'])


            # post red button test phase
            # ** known **
            test_post_red_per_spe_cnt_abs_err_ci_known, test_post_red_per_spe_cnt_rel_err_ci_known = '--', '--'

            test_post_red_per_spe_cnt_abs_err_known = round(test_post_red_per_species_count_abs_err[spe_name]['known']['value'], 2)
            if 'ci' in test_post_red_per_species_count_abs_err[spe_name]['known']:
                test_post_red_per_spe_cnt_abs_err_ci_known = ci_string(test_post_red_per_species_count_abs_err[spe_name]['known']['ci'])

            test_post_red_per_spe_cnt_rel_err_known = round(test_post_red_per_species_count_rel_err[spe_name]['known']['value'], 2)
            if 'ci' in test_post_red_per_species_count_rel_err[spe_name]['known']:
                test_post_red_per_spe_cnt_rel_err_ci_known = ci_string(test_post_red_per_species_count_rel_err[spe_name]['known']['ci'])

            # ** novel **
            test_post_red_per_spe_cnt_abs_err_ci_novel, test_post_red_per_spe_cnt_rel_err_ci_novel = '--', '--'

            test_post_red_per_spe_cnt_abs_err_novel = round(test_post_red_per_species_count_abs_err[spe_name]['novel']['value'], 2)
            if 'ci' in test_post_red_per_species_count_abs_err[spe_name]['novel']:
                test_post_red_per_spe_cnt_abs_err_ci_novel = ci_string(test_post_red_per_species_count_abs_err[spe_name]['novel']['ci'])

            test_post_red_per_spe_cnt_rel_err_novel = round(test_post_red_per_species_count_rel_err[spe_name]['novel']['value'], 2)
            if 'ci' in test_post_red_per_species_count_rel_err[spe_name]['novel']:
                test_post_red_per_spe_cnt_rel_err_ci_novel = ci_string(test_post_red_per_species_count_rel_err[spe_name]['novel']['ci'])

            output.append(
                [
                    f'{species_name} ({spe_id}):', 
                    f'{pre_red_per_spe_cnt_abs_err} {pre_red_per_spe_cnt_abs_err_ci}', 
                    f'{pre_red_per_spe_cnt_rel_err} {pre_red_per_spe_cnt_rel_err_ci}', 
                    '', 
                    f'{post_red_per_spe_cnt_abs_err_known} {post_red_per_spe_cnt_abs_err_ci_known}', 
                    f'{post_red_per_spe_cnt_abs_err_novel} {post_red_per_spe_cnt_abs_err_ci_novel}', 
                    f'{post_red_per_spe_cnt_rel_err_known} {post_red_per_spe_cnt_rel_err_ci_known}',
                    f'{post_red_per_spe_cnt_rel_err_novel} {post_red_per_spe_cnt_rel_err_ci_novel}', 
                    '', 
                    f'{test_post_red_per_spe_cnt_abs_err_known} {test_post_red_per_spe_cnt_abs_err_ci_known}', 
                    f'{test_post_red_per_spe_cnt_abs_err_novel} {test_post_red_per_spe_cnt_abs_err_ci_novel}', 
                    f'{test_post_red_per_spe_cnt_rel_err_known} {test_post_red_per_spe_cnt_rel_err_ci_known}',
                    f'{test_post_red_per_spe_cnt_rel_err_novel} {test_post_red_per_spe_cnt_rel_err_ci_novel}'
                ]
            )

        output.append(['', '', '', '', '', '', '', '', '', '', '', '', ''])


    output += [
        ['', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', '', '', '', '', '', '']
    ]

    with open(os.path.join(output_path, f'summary.csv'), 'w') as myfile:
     
        # using csv.writer method from CSV package
        write = csv.writer(myfile)
        write.writerows(output)

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


