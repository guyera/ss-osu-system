import os
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
        summary.write(f'              Red Decl Res Delay                  ***********************  Species count  ************************   \n')
        summary.write(f'                                  Total    Novel  ------------  Pre  ----------- | ------------  Post  ----------   \n')
        summary.write(f'                                                  separate_novel  combined_novel | separate_novel  combined_novel    \n')
        for test_id in test_ids:
            red_button_pos = results_dict['Red_button'][test_id]
            sys_declare_pos = results_dict['Red_button_declared'][test_id]
            detect = results_dict['Red_button_result'][test_id]
            delay = results_dict['Delay'][test_id]
            total = results_dict['Total'][test_id]
            novel = results_dict['Novel'][test_id]

            pre_red_species_count_err2 = results_dict['Species_counts'][test_id]['pre_red_btn']['all_species']
            pre_red_species_count_err1 = results_dict['Species_counts'][test_id]['pre_red_btn']['combined_novel_species']

            post_red_species_count_err2 = results_dict['Species_counts'][test_id]['post_red_btn']['all_species']
            post_red_species_count_err1 = results_dict['Species_counts'][test_id]['post_red_btn']['combined_novel_species']

            summary.write(f'{test_id}: '
                  f'{red_button_pos:8} {sys_declare_pos:3} {detect:3} {delay:3}    ' 
                  f'{total:3}    {novel:3} {pre_red_species_count_err2:15}    {pre_red_species_count_err1:10}            '
                  f'{post_red_species_count_err2:7}    {post_red_species_count_err1:10} \n')


        summary.write('\n\n')

        # Aggregate species presence
        summary.write(f'              ********************  Aggreate species presence  *******************   \n')
        summary.write(f'              ------------  Pre  ------------- |  ------------  Post  ------------   \n')
        summary.write(f'               Mean_BS   Mean_AUC   Mean_Acc   |    Mean_BS   Mean_AUC   Mean_Acc    \n')

        for test_id in test_ids:
            pre_red_species_mean_brier_score = results_dict['Aggregate_species_presence'][test_id]['pre_red_btn']['mean_bs']
            pre_red_mean_spe_presence_acc = results_dict['Aggregate_species_presence'][test_id]['pre_red_btn']['mean_acc']
            pre_red_mean_spe_presence_auc = results_dict['Aggregate_species_presence'][test_id]['pre_red_btn']['mean_auc']

            post_red_species_mean_brier_score = results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['mean_bs']
            post_red_mean_spe_presence_acc = results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['mean_acc']
            post_red_mean_spe_presence_auc = results_dict['Aggregate_species_presence'][test_id]['post_red_btn']['mean_auc']

            summary.write(f'{test_id}: '
                  f'{pre_red_species_mean_brier_score:11}   {pre_red_mean_spe_presence_auc:10}   {pre_red_mean_spe_presence_acc:10}' 
                  f'{post_red_species_mean_brier_score:10}   {post_red_mean_spe_presence_auc:10}   {post_red_mean_spe_presence_acc:10} \n')
        
        summary.write('\n\n')

        # Aggregate activity presence
        summary.write(f'              *******************  Aggregate activity presence  ******************   \n')
        summary.write(f'              ------------  Pre  ------------- |  ------------  Post  ------------   \n')
        summary.write(f'               Mean_BS   Mean_AUC   Mean_Acc   |    Mean_BS   Mean_AUC   Mean_Acc    \n')

        for test_id in test_ids:
            pre_red_activity_mean_brier_score = results_dict['Aggregate_activity_presence'][test_id]['pre_red_btn']['mean_bs'] 
            pre_red_mean_activity_acc = results_dict['Aggregate_activity_presence'][test_id]['pre_red_btn']['mean_acc']
            pre_red_mean_activity_auc = results_dict['Aggregate_activity_presence'][test_id]['pre_red_btn']['mean_auc']

            post_red_activity_mean_brier_score = results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['mean_bs']
            post_red_mean_activity_acc = results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['mean_acc']
            post_red_mean_activity_auc = results_dict['Aggregate_activity_presence'][test_id]['post_red_btn']['mean_auc']

            summary.write(f'{test_id}: '
                  f'{pre_red_activity_mean_brier_score:11}   {pre_red_mean_activity_auc:10}   {pre_red_mean_activity_acc:10}' 
                  f'{post_red_activity_mean_brier_score:10}   {post_red_mean_activity_auc:10}   {post_red_mean_activity_acc:10} \n')

        summary.write(f"\n\n{'_'*150}\n\n")

        # Per species presence
        summary.write(f'                             *********  Per species presence  *********  \n')
        # summary.write(f'              -------  Pre  ------ | -----  Post  -----   \n')
        # summary.write(f'                BS    AUC   Acc    |    BS   AUC   Acc    \n')
        # summary.write(f'{test_id}: \n')

        for test_id in test_ids:
            pre_red_per_species_brier_score = results_dict['Per_species_presence'][test_id]['pre_red_btn']['bs']
            pre_red_per_species_presence_acc = results_dict['Per_species_presence'][test_id]['pre_red_btn']['acc']
            pre_red_per_species_auc = results_dict['Per_species_presence'][test_id]['pre_red_btn']['auc']

            post_red_per_species_brier_score = results_dict['Per_species_presence'][test_id]['post_red_btn']['bs']
            post_red_per_species_presence_acc = results_dict['Per_species_presence'][test_id]['post_red_btn']['acc']
            post_red_per_species_auc = results_dict['Per_species_presence'][test_id]['post_red_btn']['auc']

            species_id2name_mapping = results_dict['Species_id2name'][test_id]

            summary.write(f'                             -------  Pre  ------ | -----  Post  -----   \n')
            summary.write(f'                               BS    AUC   Acc    |    BS   AUC   Acc    \n')
            summary.write(f'{test_id}: \n')
            
            for i in range(len(pre_red_per_species_brier_score)):
                species_name = 'unknown'
                if i in species_id2name_mapping:
                    species_name = species_id2name_mapping[i]
                elif i == 0:
                    # this is an empty images
                    species_name = species_id2name_mapping['--']

                summary.write(f'      {species_name:15} ({i}): '
                  f'{round(pre_red_per_species_brier_score[i], 2):9} {percent_string(pre_red_per_species_auc[i]):7} '
                  f'{percent_string(pre_red_per_species_presence_acc[i]):7} {round(post_red_per_species_brier_score[i], 2):7} '
                  f'{percent_string(post_red_per_species_auc[i]):7} {percent_string(post_red_per_species_presence_acc[i]):7} \n')
                
            summary.write(f"{'-'*60}\n")
        
        summary.write(f"{'_'*100}\n\n")

        # Per activity presence
        summary.write(f'                        *********  Per Activity presence  *********  \n')
        # summary.write(f'              -------  Pre  ------ | -----  Post  -----   \n')
        # summary.write(f'                BS    AUC   Acc    |    BS   AUC   Acc    \n')
        

        for test_id in test_ids:
            pre_red_per_activity_brier_score = results_dict['Per_activity_presence'][test_id]['pre_red_btn']['bs']
            pre_red_per_activity_acc = results_dict['Per_activity_presence'][test_id]['pre_red_btn']['acc']
            pre_red_per_activity_auc = results_dict['Per_activity_presence'][test_id]['pre_red_btn']['auc']

            post_red_per_activity_brier_score = results_dict['Per_activity_presence'][test_id]['post_red_btn']['bs']
            post_red_per_activity_acc = results_dict['Per_activity_presence'][test_id]['post_red_btn']['acc']
            post_red_per_activity_auc = results_dict['Per_activity_presence'][test_id]['post_red_btn']['auc']

            activity_id2name_mapping = results_dict['Activity_id2name'][test_id]

            summary.write(f'                        -------  Pre  ------ | -----  Post  -----   \n')
            summary.write(f'                          BS    AUC   Acc    |    BS   AUC   Acc    \n')
            summary.write(f'{test_id}: \n')
            for i in range(len(pre_red_per_activity_brier_score)):
                activity_name = 'unknown'
                if i in activity_id2name_mapping:
                    activity_name = activity_id2name_mapping[i]

                summary.write(f'       {activity_name:10} ({i}): '
                  f'{round(pre_red_per_activity_brier_score[i], 2):9} {percent_string(pre_red_per_activity_auc[i]):7} '
                  f'{percent_string(pre_red_per_activity_acc[i]):7} {round(post_red_per_activity_brier_score[i], 2):7} '
                  f'{percent_string(post_red_per_activity_auc[i]):7} {percent_string(post_red_per_activity_acc[i]):7} \n')
                
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
            pre_red_species_ids = results_dict['Species_confusion_matrices'][test_id]['pre_red_btn']['species_ids']
            pre_red_species_cm = results_dict['Species_confusion_matrices'][test_id]['pre_red_btn']['cm']

            post_red_species_ids = results_dict['Species_confusion_matrices'][test_id]['post_red_btn']['species_ids']
            post_red_species_cm = results_dict['Species_confusion_matrices'][test_id]['post_red_btn']['cm']

            species_id2name_mapping = results_dict['Species_id2name'][test_id]

            # pre_red_species_names = [species_id2name_mapping[i] if i != 0 else species_id2name_mapping['--'] for i in sorted(pre_red_species_ids)]
            # post_red_species_names = [species_id2name_mapping[i] if i != 0 else species_id2name_mapping['--'] for i in sorted(post_red_species_ids)]
            pre_red_species_names = [species_id2name_mapping[i] for i in sorted(pre_red_species_ids)]
            post_red_species_names = [species_id2name_mapping[i] for i in sorted(post_red_species_ids)]


            # Print count version
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
            plt.title(f"{test_id}: pre red btn species confusion matrix")
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

        # plot confusion matrix of activities
        for test_id in test_ids:
            pre_red_activity_ids = results_dict['Activity_confusion_matrices'][test_id]['pre_red_btn']['activity_ids']
            pre_red_activity_cm = results_dict['Activity_confusion_matrices'][test_id]['pre_red_btn']['cm']

            post_red_activity_ids = results_dict['Activity_confusion_matrices'][test_id]['post_red_btn']['activity_ids']
            post_red_activity_cm = results_dict['Activity_confusion_matrices'][test_id]['post_red_btn']['cm']

            activity_id2name_mapping = results_dict['Activity_id2name'][test_id]

            pre_red_activity_names = [activity_id2name_mapping[i] for i in pre_red_activity_ids]
            post_red_activity_names = [activity_id2name_mapping[i] for i in post_red_activity_ids]

            # print pre red button confusion matrix
            fig = plt.figure('pre red button', figsize=[8, 6])
            # ax = plt.axes()
            ax = fig.add_subplot(211)
            sns.heatmap(pre_red_activity_cm,
                        annot=True,
                        fmt="d",
                        cmap='Blues',)
            ax.xaxis.set_ticklabels(pre_red_activity_names, fontsize=8, rotation='horizontal')
            ax.yaxis.set_ticklabels(pre_red_activity_names, fontsize=8,  rotation='horizontal')
            ax.set_ylabel("True label")
            ax.set_title(f"{test_id}: pre red btn activities confusion matrix")
            ax.tick_params(axis='x', which='major', labelbottom=False, bottom=False, top=True, labeltop=True)

            # print post red button confusion matrix
            ax = fig.add_subplot(212)
            sns.heatmap(post_red_activity_cm,
                        annot=True,
                        fmt="d",
                        cmap='Blues',)
            ax.xaxis.set_ticklabels(post_red_activity_names, fontsize=8, rotation='horizontal')
            ax.yaxis.set_ticklabels(post_red_activity_names, fontsize=8, rotation='horizontal')
            ax.set_ylabel("True label")
            ax.set_title(f"{test_id}: post red btn activities confusion matrix")
            ax.tick_params(axis='x', which='major', labelbottom=False, bottom=False, top=True, labeltop=True)
            fig.tight_layout()
            pdf_pages.savefig()
            plt.close()


