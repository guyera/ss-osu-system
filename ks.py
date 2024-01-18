import numpy as np
import sys
import pickle as pkl

from tqdm import tqdm
import pandas
from scipy.stats import ks_2samp

def sliding_window_p_vals(sample, prefix_size, window_size):
    prefix = sample[:prefix_size]
    start = prefix_size + window_size - 1
    end = sample.shape[0]
    p_vals = []
    for i in range(start, end):
        window = sample[i - window_size + 1 : i + 1]
        # p_val = ks_2samp(prefix, window, alternative='greater')[1]
        p_val = ks_2samp(
            prefix,
            window,
            alternative='greater',
            method='exact'
        )[1]
        p_vals.append(p_val)
    return np.array(p_vals)

def permutation_trial(n_itr, prefix_size, window_size, p_ni_vals):
    n = len(p_ni_vals)
    min_pvals = []
    for itr in tqdm(range(n_itr), leave=False):
        sample = np.random.choice(p_ni_vals, size=n, replace=True)
        p_vals = sliding_window_p_vals(sample, prefix_size, window_size)
        min_pval = p_vals.min()
        min_pvals.append(min_pval)
    return np.array(min_pvals)

with open('OND.100.000.pkl', 'rb') as f:
    log = pkl.load(f)
p_ni_vals = log['p_ni_raw']

prefix_size = 60

window_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
window_sizes = [40]
n_window_sizes = len(window_sizes)


print('With exact method!')
df_thresholds = pandas.DataFrame(
    columns=['window_size', 'median_pvalue', 'q10_pvalue']
)
df_thresholds['window_size'] = window_sizes
df_thresholds.set_index('window_size', inplace=True)
for window_size in tqdm(window_sizes, leave=False):
    min_pvals = permutation_trial(1000, prefix_size, window_size, p_ni_vals)
    median_cutoff = np.median(min_pvals)
    q10_cutoff = np.quantile(min_pvals, 0.1)
    df_thresholds.loc[window_size, 'median_pvalue'] = median_cutoff
    df_thresholds.loc[window_size, 'q10_pvalue'] = q10_cutoff

print(df_thresholds)
