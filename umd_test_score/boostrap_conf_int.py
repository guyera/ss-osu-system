import pandas as pd
import numpy as np
import sklearn.metrics as metrics

from helpers import species_count_error

np.seterr('raise')


def boostrap_conf_interval(y_true, y_pred, metric_name, is_abs_err=True, n_samples=500, alpha=0.05, seed=10000):
    """
    Computes estimate confidence interval of a metric by simulation
    Arguments:
        y_pred: model prediction (can be presence or counts)
        y_true: ground truth labels
        metric_name: name of the metric being computed
        is_abs_err: check if computing absolute error or not (only used when metric_name == 'count_err')
        n_samples: number of samples to draw for simulation
        alpha: confidence level
        seed: random state for repreducibility
    """
    np.random.seed(seed)

    assert 0 < alpha < 1
    lower_c = alpha*100/2
    upper_c = (1-alpha/2)*100

    assert metric_name in ('pre/rec/f1', 'avg_pre/rec/f1', 'auc', 'avg_auc', 'count_err', 'avg_count_err')

    
    if 'pre/rec/f1' in metric_name and 'avg' in metric_name:
        sample_score = {
            'avg_precision': [],
            'avg_recall': [],
            'avg_f1_score': []
        }
    elif 'pre/rec/f1' in metric_name:
        sample_score = {
            'precision': [],
            'recall': [],
            'f1_score': []
        }
    else:
        sample_score = []

    # import pdb; pdb.set_trace();

    if 'avg' not in metric_name:
        for _ in range(n_samples):
            # sample indices with replacement
            indices = np.random.randint(0, len(y_true), len(y_true))
            if metric_name == 'auc' and len(set(y_true[indices])) < 2:
                # all classes are required in the ground truth for AUC
                continue
            
            if metric_name.lower() == 'auc':
                score = metrics.roc_auc_score(y_true[indices],  y_pred[indices])
                sample_score.append(score)
            elif metric_name.lower() == 'pre/rec/f1':
                try:
                    pre, rec, f1, _ = metrics.precision_recall_fscore_support(
                        y_true[indices],  
                        y_pred[indices],
                        average='binary',
                        zero_division=0.0
                    )
                    sample_score['precision'].append(pre)
                    sample_score['recall'].append(rec)
                    if f1 != 0:
                        sample_score['f1_score'].append(f1)
                except Exception as ex:
                    print('==>> Exception:', ex)
                    continue

            elif metric_name == 'count_err':
                if is_abs_err:
                    total = len(y_true[indices][y_true[indices] > 0])
                else:
                    total = sum(y_true[indices])

                if total == 0:
                    continue

                score = species_count_error(
                    y_true[indices], 
                    y_pred[indices], 
                    metric='AE'
                ) / total
                sample_score.append(score)
            else:
                raise ValueError(f'metric_name "{metric_name}" unknown!')

        if isinstance(sample_score, list):
            return np.percentile(sample_score, (lower_c, upper_c)) if len(sample_score) > 0 else -1
        return {
            'precision': np.percentile(sample_score['precision'], (lower_c, upper_c)) if len(sample_score['precision']) > 0 else -1,
            'recall': np.percentile(sample_score['recall'], (lower_c, upper_c)) if len(sample_score['recall']) > 0 else -1,
            'f1_score': np.percentile(sample_score['f1_score'], (lower_c, upper_c)) if len(sample_score['f1_score']) > 0 else -1
        }

    # ** computing the confidence interval for the average of the metric over all species or activities
    # --------
    # sample indices with replacement
    for _ in range(n_samples):
        assert isinstance(y_true, pd.DataFrame) and isinstance(y_pred, pd.DataFrame)
        y_true.reset_index(drop=True, inplace=True)
        y_pred.reset_index(drop=True, inplace=True)

        if 'pre/rec/f1' in metric_name:
            col_scores = {
                'precision': [],
                'recall': [],
                'f1_score': []
            }
        else:
            col_scores = []

        indices = np.random.randint(0, len(y_true), len(y_true))
        for col in y_true.columns:
                       
            if metric_name.lower() == 'avg_auc':
                if len(y_true[col].iloc[indices].unique()) < 2:
                    # all classes are required in the ground truth for AUC
                    continue

                try:
                    col_score = metrics.roc_auc_score(y_true[col].iloc[indices],  y_pred[col].iloc[indices])
                except Exception as ex:
                    print('>> This exception happened when computing avg_auc:', ex)
                    continue

                col_scores.append(col_score)
            elif metric_name.lower() == 'avg_pre/rec/f1':
                try:
                    pre, rec, f1, _ = metrics.precision_recall_fscore_support(
                        y_true[col].iloc[indices].to_numpy(),  
                        y_pred[col].iloc[indices].to_numpy(),
                        average='binary',
                        zero_division=0.0
                    )
                    col_scores['precision'].append(pre)
                    col_scores['recall'].append(rec)
                    if f1 != 0:
                        col_scores['f1_score'].append(f1)
                except Exception as ex:
                    print('** The following exception happened:', ex)
                    continue
            elif metric_name == 'avg_count_err':
                if is_abs_err:
                    total_col = y_true[col].iloc[indices].astype(bool).sum()
                else:
                    total_col = y_true[col].iloc[indices].sum()

                if total_col == 0:
                    continue

                col_score = species_count_error(
                    y_true[col].iloc[indices], 
                    y_pred[col].iloc[indices], 
                    metric='AE'
                ) / total_col
                col_scores.append(col_score)
            else:
                raise ValueError(f'metric_name "{metric_name}" unknown!')

        if metric_name.lower() == 'avg_pre/rec/f1':
            sample_score['avg_precision'].append(np.mean(col_scores['precision']))
            sample_score['avg_recall'].append(np.mean(col_scores['recall']))
            sample_score['avg_f1_score'].append(np.mean(col_scores['f1_score']))
        else:
            try:
                sample_score.append(np.mean(col_scores))
            except Exception as ex:
                continue
                print(col_scores)
                print('\n The following exception happened in boostrap_conf_int():', ex)

    if isinstance(sample_score, list):
        return np.percentile(sample_score, (lower_c, upper_c)) if len(sample_score) > 0 else -1
    return {
        'avg_precision': np.percentile(sample_score['avg_precision'], (lower_c, upper_c)) if len(sample_score['avg_precision']) > 0 else -1,
        'avg_recall': np.percentile(sample_score['avg_recall'], (lower_c, upper_c)) if len(sample_score['avg_recall']) > 0 else -1,
        'avg_f1_score': np.percentile(sample_score['avg_f1_score'], (lower_c, upper_c)) if len(sample_score['avg_f1_score']) > 0 else -1
    }
