# This software was created by Oregon State University under Army Research
# Office (ARO) Award Number W911NF-22-2-0149. ARO, as the Federal awarding
# agency, reserves a royalty-free, nonexclusive and irrevocable right to
# reproduce, publish, or otherwise use this software for Federal purposes, and
# to authorize others to do so in accordance with 2 CFR 200.315(b).

import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from matplotlib.backends.backend_pdf import PdfPages

# Paths for pkl and csv files

Feedback_W_0_5 = [
'/nfs/hpc/share/sail_on3/TestsForPaper/Jan2024_Corrected_Normalization/EWC_100/logs/OND.100.000/OND.100.000.pkl',
'/nfs/hpc/share/sail_on3/TestsForPaper/Jan2024_Corrected_Normalization/End-to-End-102-Feedback_W_0.5/logs/OND.102.000/OND.102.000.pkl',
'/nfs/hpc/share/sail_on3/TestsForPaper/Jan2024_Corrected_Normalization/End-to-End-103-Feedback_W_0.5/logs/OND.103.000/OND.103.000.pkl',
'/nfs/hpc/share/sail_on3/TestsForPaper/Jan2024_Corrected_Normalization/End-to-End-104-Feedback_W_0.5/logs/OND.104.000/OND.104.000.pkl',
'/nfs/hpc/share/sail_on3/TestsForPaper/Jan2024_Corrected_Normalization/End-to-End-1061-Feedback_W_0.5/logs/OND.1061.000/OND.1061.000.pkl',
'/nfs/hpc/share/sail_on3/TestsForPaper/Jan2024_Corrected_Normalization/End-to-End-1062-Feedback_W_0.5/logs/OND.1062.000/OND.1062.000.pkl',
'/nfs/hpc/share/sail_on3/TestsForPaper/Jan2024_Corrected_Normalization/End-to-End-1063-Feedback_W_0.5/logs/OND.1063.000/OND.1063.000.pkl',
'/nfs/hpc/share/sail_on3/TestsForPaper/Jan2024_Corrected_Normalization/End-to-End-1064-Amin-Feedback_W_0.5/logs/OND.1064.000/OND.1064.000.pkl',

]



csv_paths = [
'/nfs/hpc/share/sail_on3/final/test_trials/api_tests/OND/image_classification/OND.100.000_single_df.csv',
'/nfs/hpc/share/sail_on3/final/test_trials/api_tests/OND/image_classification/OND.102.000_single_df.csv',
'/nfs/hpc/share/sail_on3/final/test_trials/api_tests/OND/image_classification/OND.103.000_single_df.csv',
'/nfs/hpc/share/sail_on3/final/test_trials/api_tests/OND/image_classification/OND.104.000_single_df.csv',
'/nfs/hpc/share/sail_on3/final/test_trials/api_tests/OND/image_classification/OND.1061.000_single_df.csv',
'/nfs/hpc/share/sail_on3/final/test_trials/api_tests/OND/image_classification/OND.1062.000_single_df.csv',
'/nfs/hpc/share/sail_on3/final/test_trials/api_tests/OND/image_classification/OND.1063.000_single_df.csv',
'/nfs/hpc/share/sail_on3/final/test_trials/api_tests/OND/image_classification/OND.1064.000_single_df.csv',
]

captions = [
    'OND.100.000',
    'OND.102.000',
    'OND.103.000',
    'OND.104.000',
    'OND.1061.000',
    'OND.1062.000',
    'OND.1063.000',
    'OND.1064.000',
]

all_exp = [Feedback_W_0_5]
all_exp_names = ['Feedback_W_0_5']

for exp, exp_name in zip(all_exp, all_exp_names):
    pkl_paths = exp
    with PdfPages(f'./ROC_curves/roc_curve_{exp_name}.pdf') as pdf:
        for pkl_path, csv_path, caption in zip(pkl_paths, csv_paths, captions):
            print(exp_name, '  ', caption)
            print(f'\t {pkl_path[-27:], csv_path[-27:]}')
            if caption not in pkl_path:
                print(f'ROC Not calculated for : \t {pkl_path[-27:]}')
                continue
        

            with open(pkl_path, 'rb') as f:
                log = pkl.load(f)
            if len(log['queries']) == 0:
                print(f'{csv_path} has no queries')
                continue

            test_data_csv_path = csv_path
            # novelty_check = pd.read_csv(test_data_csv_path)['novel'].tolist()[-len(log['queries']):]
            novelty_check = pd.read_csv(test_data_csv_path)['novel'].tolist()[-2500:]
            p_ni_raw = log['p_ni_raw'][3000-len(log['queries']):]
            # import ipdb; ipdb.set_trace()
            # Data before and after training
            gt_queries_before_training = novelty_check[:-1000]
            # p_ni_raw_before_training  = log['p_ni_raw'][3000-len(log['queries']):-1000]
            p_ni_raw_before_training  = log['p_ni_raw'][3000-2500:-1000]
            gt_queries_After_training = novelty_check[-1000:]
            p_ni_raw_After_training  = log['p_ni_raw'][-1000:]

            # Create subplots
            fig, axs = plt.subplots(2, figsize=(10, 15))

            # Plot for data before training
            fpr, tpr, _ = roc_curve(gt_queries_before_training, p_ni_raw_before_training, pos_label = 1)
            roc_auc = auc(fpr, tpr)
            axs[0].plot(fpr, tpr, color='green', lw=2, label='Post-novelty Before Test (AUC = %0.2f)' % roc_auc)
            axs[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axs[0].set_title('Post-novelty Before Test')
            axs[0].set_xlabel('False Positive Rate')
            axs[0].set_ylabel('True Positive Rate')
            axs[0].legend(loc="lower right")

            # Plot for data after training
            fpr, tpr, _ = roc_curve(gt_queries_After_training, p_ni_raw_After_training)
            roc_auc = auc(fpr, tpr)
            axs[1].plot(fpr, tpr, color='red', lw=2, label='Test Set (AUC = %0.2f)' % roc_auc)
            axs[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axs[1].set_title('Test Data')
            axs[1].set_xlabel('False Positive Rate')
            axs[1].set_ylabel('True Positive Rate')
            axs[1].legend(loc="lower right")

            # Adjust layout and save to PDF
            plt.figtext(0.8, 0.5, caption, wrap=True, verticalalignment='center', fontsize=12, rotation='vertical')

            plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust the rect to make space for caption
            pdf.savefig(fig)
            plt.close(fig)


