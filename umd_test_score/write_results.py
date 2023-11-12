import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict
from pathlib import Path
import seaborn as sns


def percent_string(num, denom=None):
    if num < 0:
        return '  --  '
    elif denom is None:
        return f'{100 * num:6.2f}%'
    return f'{100 * num / denom:6.2f}%'


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


        # Aggregate species presence
        summary.write(f'               ************************  Aggreate species presence and counts  ***********************   \n')
        summary.write(f'               ------------  Pre  -------------          |          ------------  Post  ------------   \n')
        summary.write(f'               Avg_AUC  Avg_F1   Avg_abs_err   Avg_rel   |    Avg_AUC  Avg_F1   Avg_abs_err   Avg_rel    \n')

        for test_id in test_ids:
            pre_red_species_avg_auc = results_dict['Aggregate_species_presence'][test_id]['pre_red_btn']['avg_auc']
            pre_red_species_avg_precision = results_dict['Aggregate_species_presence'][test_id]['pre_red_btn']['avg_precision']
            pre_red_species_avg_recall = results_dict['Aggregate_species_presence'][test_id]['pre_red_btn']['avg_recall']
            pre_red_species_avg_f1 = results_dict['Aggregate_species_presence'][test_id]['pre_red_btn']['avg_f1_score']
            pre_red_avg_spe_abs_err = results_dict['Aggregate_species_counts'][test_id]['pre_red_btn']['avg_abs_err']
            pre_red_avg_spe_rel_err = results_dict['Aggregate_species_counts'][test_id]['pre_red_btn']['avg_rel_err']

            post_red_species_avg_auc = results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['avg_auc']
            post_red_species_avg_precision = results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['avg_precision']
            post_red_species_avg_recall = results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['avg_recall']
            post_red_species_avg_f1 = results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['avg_f1_score']
            post_red_avg_spe_abs_err = results_dict['Aggregate_species_counts'][test_id]['post_red_btn']['avg_abs_err']
            post_red_avg_spe_rel_err = results_dict['Aggregate_species_counts'][test_id]['post_red_btn']['avg_rel_err']

            summary.write(f'{test_id}: '
                  f'{pre_red_species_avg_auc:9}   {pre_red_species_avg_precision:9}   {pre_red_avg_spe_abs_err:9}   {pre_red_avg_spe_rel_err:9}   ' 
                  f'{post_red_species_avg_auc:9}   {post_red_species_avg_precision:9}   {post_red_avg_spe_abs_err:9}   {post_red_avg_spe_rel_err:7} \n')
        
        summary.write('\n\n')

        # Aggregate activity presence
        summary.write(f'              *******************  Aggregate activity presence  ******************   \n')
        summary.write(f'                    --------  Pre  --------    |     --------  Post  --------   \n')
        summary.write(f'                      Avg_AUC     Avg_F1       |       avg_AUC     Avg_F1         \n')
        

        for test_id in test_ids:
            pre_red_activity_avg_auc = results_dict['Aggregate_activity_presence'][test_id]['pre_red_btn']['avg_auc'] 
            pre_red_activity_avg_precision = results_dict['Aggregate_activity_presence'][test_id]['pre_red_btn']['avg_precision']
            pre_red_activity_avg_recall = results_dict['Aggregate_activity_presence'][test_id]['pre_red_btn']['avg_recall']
            pre_red_activity_avg_f1 = results_dict['Aggregate_activity_presence'][test_id]['pre_red_btn']['avg_f1_score']

            post_red_activity_avg_auc = results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_auc']
            post_red_activity_avg_precision = results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_precision']
            post_red_activity_avg_recall = results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_recall']
            post_red_activity_avg_f1 = results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['avg_f1_score']

            summary.write(f'{test_id}:            '
                    f'{pre_red_activity_avg_auc:9}  {pre_red_activity_avg_f1:9}         {post_red_activity_avg_auc:9}    {post_red_activity_avg_f1:7} \n')

        summary.write(f"\n\n{'_'*150}\n\n")

        # Per species presence
        summary.write(f'                             *********  Per species count and presence  *********  \n')

        for test_id in test_ids:
            pre_red_per_species_count_abs_err = results_dict['Species_counts'][test_id]['pre_red_btn']['abs_err']
            pre_red_per_species_count_rel_err = results_dict['Species_counts'][test_id]['pre_red_btn']['rel_err']

            post_red_per_species_count_abs_err = results_dict['Species_counts'][test_id]['post_red_btn']['abs_err']
            post_red_per_species_count_rel_err = results_dict['Species_counts'][test_id]['post_red_btn']['rel_err']


            pre_red_per_species_auc = results_dict['Per_species_presence'][test_id]['pre_red_btn']['auc']
            pre_red_per_species_precision = results_dict['Per_species_presence'][test_id]['pre_red_btn']['precision']
            pre_red_per_species_recall = results_dict['Per_species_presence'][test_id]['pre_red_btn']['recall']
            pre_red_per_species_f1_score = results_dict['Per_species_presence'][test_id]['pre_red_btn']['f1_score']

            post_red_per_species_auc = results_dict['Per_species_presence'][test_id]['post_red_btn']['auc']
            post_red_per_species_precision = results_dict['Per_species_presence'][test_id]['post_red_btn']['precision']
            post_red_per_species_recall = results_dict['Per_species_presence'][test_id]['post_red_btn']['recall']
            post_red_per_species_f1_score = results_dict['Per_species_presence'][test_id]['post_red_btn']['f1_score']

            species_id2name_mapping = results_dict['Species_id2name'][test_id]
            species_name2id_mapping = dict((v,k) for k,v in species_id2name_mapping.items())

            summary.write(f'\n                               --------------  Pre  -------------   |   --------------  Post  ----------    \n')
            summary.write(f'                                 AUC    F1    Abs_err   Rel_err     |     AUC    F1     Abs_err   Rel_err    \n')
            summary.write(f'{test_id}: \n')

            species_present = list(set(list(pre_red_per_species_auc.keys()) + list(post_red_per_species_auc.keys())))
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

                summary.write(f'      {species_name:15} ({spe_id}): '
                  f'{round(pre_red_per_species_auc[spe_name], 2):7} {round(pre_red_per_species_f1_score[spe_name], 2):7} {round(pre_red_per_species_count_abs_err[spe_name], 2):7}   '
                  f'{round(pre_red_per_species_count_rel_err[spe_name], 2):9}      {round(post_red_per_species_auc[spe_name], 2):7} {round(post_red_per_species_f1_score[spe_name], 2):7}   '
                  f'{round(post_red_per_species_count_abs_err[spe_name], 2):7} {round(post_red_per_species_count_rel_err[spe_name], 2):7} \n')
                
            summary.write(f"{'-'*100}\n")
        
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

            activity_id2name_mapping = results_dict['Activity_id2name'][test_id]
            activity_name2id_mapping = dict((v,k) for k,v in activity_id2name_mapping.items())


            summary.write(f'                        ------  Pre  ------    |    ------  Post  ------   \n')
            summary.write(f'                            AUC       F1       |       AUC        F1     \n')
            summary.write(f'{test_id}: \n')

            present_activities = list(set(list(pre_red_per_activity_auc.keys()) + list(post_red_per_activity_auc.keys())))
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

                summary.write(f'       {activity_name:10} ({act_id}): '
                  f'{round(pre_red_per_activity_auc[act_name], 2):9}  {round(pre_red_per_activity_f1_score[act_name], 2):9}       {round(post_red_per_activity_auc[act_name], 2):9}   {round(post_red_per_activity_f1_score[act_name], 2):7} \n')
                
            summary.write(f"{'-'*60}\n")

    '''
    with open(os.path.join(output_path, f'confusion_matrices.log'), 'w') as conf_mat:
            conf_mat.write(f'                  ***********************  Species prediction confusion matrix  ************************   \n')

            for test_id in test_ids:
                pre_red_species_ids = results_dict['Species_confusion_matrices'][test_id]['pre_red_btn']['species_ids']
                pre_red_species_cm = results_dict['Species_confusion_matrices'][test_id]['pre_red_btn']['cm']

                post_red_species_ids = results_dict['Species_confusion_matrices'][test_id]['post_red_btn']['species_ids']
                post_red_species_cm = results_dict['Species_confusion_matrices'][test_id]['post_red_btn']['cm']

                species_id2name_mapping = results_dict['Species_id2name'][test_id]

                # conf_mat.write(f'                             -------  Pre  ------ | -----  Post  -----   \n')
                # conf_mat.write(f'                               BS    AUC   Acc    |    BS   AUC   Acc    \n')
                conf_mat.write(f'{test_id}: \n')
                conf_mat.write(f'              pre red button: \n\n')

                # write confusion matrix species
                for i in pre_red_species_ids:
                    species_name = 'unknown'
                    if i in species_id2name_mapping:
                        species_name = species_id2name_mapping[i]
                    elif i == 0:
                        # this is an empty images
                        species_name = species_id2name_mapping['--']
                        conf_mat.write(f'                               | {species_name:10} |')
                        continue
                    conf_mat.write(f' {species_name:10} |')
                conf_mat.write(f'\n                 ' + '-'*200 + '\n')

                for i in pre_red_species_ids:
                    species_name = 'unknown'
                    if i in species_id2name_mapping:
                        species_name = species_id2name_mapping[i]
                    elif i == 0:
                        # this is an empty images
                        species_name = species_id2name_mapping['--']
                    conf_mat.write(f'              | {species_name:15} |')

                    for j in pre_red_species_ids:
                        conf_mat.write(f' {pre_red_species_cm[i, j]:10} |')
                    conf_mat.write(f'\n                 ' + '-'*200 + '\n')
                
                conf_mat.write(f"{'_'*100}\n\n")
    '''


def print_confusion_matrices(results_dict, out_file):
    test_ids = sorted(list(results_dict['Red_button'].keys()))

    with PdfPages(out_file) as pdf_pages:
        # plot confusion matrix of species
        for test_id in test_ids:
            species_ids = results_dict['Confusion_matrices'][test_id]['species']['species_ids']
            species_cm = results_dict['Confusion_matrices'][test_id]['species']['cm']

            species_id2name_mapping = results_dict['Species_id2name'][test_id]

            species_names = [species_id2name_mapping[i] for i in sorted(species_ids)]


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
            ax.set_title(f"{test_id}: species confusion matrix")
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
            activity_ids = results_dict['Confusion_matrices'][test_id]['activity']['activity_ids']
            activity_cm = results_dict['Confusion_matrices'][test_id]['activity']['cm']

            activity_id2name_mapping = results_dict['Activity_id2name'][test_id]

            activity_names = [activity_id2name_mapping[i] for i in sorted(activity_ids)]


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
            ax.set_title(f"{test_id}: activities confusion matrix")
            fig.tight_layout()
            pdf_pages.savefig()
            plt.close()


