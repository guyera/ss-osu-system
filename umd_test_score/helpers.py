import numpy as np
import pandas as pd
import sklearn.metrics as metrics


def percent_string(num, denom=None):
    if denom is None:
        return f'{100 * num:6.2f}%'
    return f'{100 * num / denom:6.2f}%'


def species_count_error(grd_trth, pred, metric='MAE'):
    '''
    Computes the squared error, absolute error, mean squared error, 
    mean absolute error or root mean squared error between the ground truth and the predicion
    '''
    assert metric in ('AE', 'MAE')  # ('RE', 'AE', 'MAE', 'MRE')
    assert len(grd_trth) == len(pred)

    grd_trth = np.array(grd_trth)
    pred = np.array(pred)
    err = grd_trth - pred

    if np.sum(np.abs(err)) == 0:
        return -1

    if metric == 'AE' or metric == 'MAE':
        return np.sum(np.abs(err)) / len(grd_trth)
    elif metric == 'MRE':
        sum_err = np.sum(np.abs(err), axis=1)
        num_boxes = np.sum(grd_trth, axis=1)
        rel_err = np.divide(sum_err, 2 * num_boxes)
        if len(rel_err) == 0 or len(grd_trth) == 0:
            return -1
        return np.sum(rel_err) / len(grd_trth)
    return -1

