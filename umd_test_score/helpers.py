import numpy as np


def percent_string(num, denom=None):
    if denom is None:
        return f'{100 * num:6.2f}%'
    return f'{100 * num / denom:6.2f}%'


def species_count_error(grd_trth_vec, pred_vec, metric='MSE'):
    '''
    Computes the squared error, absolute error, mean squared error, 
    mean absolute error or root mean squared error between the ground truth and the predicion
    '''
    assert metric in ('SE', 'AE', 'RMSE', 'MSE', 'MAE')
    assert len(grd_trth_vec) == len(pred_vec)

    grd_trth_vec = np.array(grd_trth_vec).ravel()
    pred_vec = np.array(pred_vec).ravel()
    diff = grd_trth_vec - pred_vec
    sqrt_diff = np.power(diff, 2)

    if metric == 'SE':
        return sum(sqrt_diff)
    elif metric == 'AE':
        return sum(np.abs(diff))
    elif metric == 'MSE':
        return np.mean(sqrt_diff)
    elif metric == 'MAE':
        return np.mean(np.abs(diff))
    return np.sqrt(np.mean(sqrt_diff))