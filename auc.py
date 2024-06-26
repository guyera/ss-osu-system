"""                                                                             
Created on Tue Nov  6 10:06:52 2018                                             
                                                                                
@author: yandexdataschool                                                       
                                                                                
Original Code found in:                                                         
https://github.com/yandexdataschool/roc_comparison                              
                                                                                
updated: Raul Sanchez-Vazquez                                                   
                                                                                
updated: Alexander Guyer                                                        
"""

"""                                                                             
This is a fast DeLong implementation of AUC computations with standard          
deviations. It is a heuristic which asymptotically approximates                 
AUC summaries of bootstrapping procedures with a lower computational            
cost.                                                                           
                                                                                
compute_auc() is the most important public-facing function.                     
"""                                                                             

# The below disclaimer applies to some, but not all, of the source code written
# in this file (not all of this source code was authored by OSU):
# 
# This software was created by Oregon State University under Army Research
# Office (ARO) Award Number W911NF-22-2-0149. ARO, as the Federal awarding
# agency, reserves a royalty-free, nonexclusive and irrevocable right to
# reproduce, publish, or otherwise use this software for Federal purposes, and
# to authorize others to do so in accordance with 2 CFR 200.315(b).
                                                                                
import numpy as np                                                              
import scipy.stats                                                              
from scipy import stats                                                         
                                                                                
# AUC comparison adapted from                                                   
# https://github.com/Netflix/vmaf/                                              
def compute_midrank(x):                                                         
    """Computes midranks.                                                       
    Args:                                                                       
       x - a 1D numpy array                                                     
    Returns:                                                                    
       array of midranks                                                        
    """                                                                         
    J = np.argsort(x)                                                           
    Z = x[J]                                                                    
    N = len(x)                                                                  
    T = np.zeros(N, dtype=np.float)                                             
    i = 0                                                                       
    while i < N:                                                                
        j = i                                                                   
        while j < N and Z[j] == Z[i]:                                           
            j += 1                                                              
        T[i:j] = 0.5*(i + j - 1)                                                
        i = j                                                                   
    T2 = np.empty(N, dtype=np.float)                                            
    # Note(kazeevn) +1 is due to Python using 0-based indexing                  
    # instead of 1-based in the AUC formula in the paper                        
    T2[J] = T + 1                                                               
    return T2

def compute_midrank_weight(x, sample_weight):                                   
    """Computes midranks.                                                       
    Args:                                                                       
       x - a 1D numpy array                                                     
    Returns:                                                                    
       array of midranks                                                        
    """                                                                         
    J = np.argsort(x)                                                           
    Z = x[J]                                                                    
    cumulative_weight = np.cumsum(sample_weight[J])                             
    N = len(x)                                                                  
    T = np.zeros(N, dtype=np.float)                                             
    i = 0                                                                       
    while i < N:                                                                
        j = i                                                                   
        while j < N and Z[j] == Z[i]:                                           
            j += 1                                                              
        T[i:j] = cumulative_weight[i:j].mean()                                  
        i = j                                                                   
    T2 = np.empty(N, dtype=np.float)                                            
    T2[J] = T                                                                   
    return T2 


def fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight):    
    if sample_weight is None:                                                   
        return fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
    else:                                                                       
        return fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight)


def fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight):
    """                                                                         
    The fast version of DeLong's method for computing the covariance of         
    unadjusted AUC.                                                             
    Args:                                                                       
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first                  
    Returns:                                                                    
       (AUC value, DeLong covariance)                                           
    Reference:                                                                  
     @article{sun2014fast,                                                      
       title={Fast Implementation of DeLong's Algorithm for                     
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},                                          
       journal={IEEE Signal Processing Letters},                                
       volume={21},                                                             
       number={11},                                                             
       pages={1389--1393},                                                      
       year={2014},                                                             
       publisher={IEEE}                                                         
     }                                                                          
    """                                                                         
    # Short variables are named as they are in the paper                        
    m = label_1_count                                                           
    n = predictions_sorted_transposed.shape[1] - m                              
    positive_examples = predictions_sorted_transposed[:, :m]                    
    negative_examples = predictions_sorted_transposed[:, m:]                    
    k = predictions_sorted_transposed.shape[0]                                  
                                                                                
    tx = np.empty([k, m], dtype=np.float)                                       
    ty = np.empty([k, n], dtype=np.float)                                       
    tz = np.empty([k, m + n], dtype=np.float)                                   
    for r in range(k):                                                          
        tx[r, :] = compute_midrank_weight(positive_examples[r, :], sample_weight[:m])
        ty[r, :] = compute_midrank_weight(negative_examples[r, :], sample_weight[m:])
        tz[r, :] = compute_midrank_weight(predictions_sorted_transposed[r, :], sample_weight)
    total_positive_weights = sample_weight[:m].sum()                            
    total_negative_weights = sample_weight[m:].sum()                            
    pair_weights = np.dot(sample_weight[:m, np.newaxis], sample_weight[np.newaxis, m:])
    total_pair_weights = pair_weights.sum()                                     
    aucs = (sample_weight[:m]*(tz[:, :m] - tx)).sum(axis=1) / total_pair_weights
    v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights                       
    v10 = 1. - (tz[:, m:] - ty[:, :]) / total_positive_weights                  
    sx = np.cov(v01)                                                            
    sy = np.cov(v10)                                                            
    delongcov = sx / m + sy / n                                                 
    return aucs, delongcov 

def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):        
    """                                                                         
    The fast version of DeLong's method for computing the covariance of         
    unadjusted AUC.                                                             
    Args:                                                                       
       predictions_sorted_transposed: 2D numpy.array[n_classifiers, n_examples] 
          sorted such as the examples with label "1" are first                  
    Returns:                                                                    
       (AUC value, DeLong covariance)                                           
    Reference:                                                                  
     @article{sun2014fast,                                                      
       title={Fast Implementation of DeLong's Algorithm for                     
              Comparing the Areas Under Correlated Receiver Oerating            
              Characteristic Curves},                                           
       author={Xu Sun and Weichao Xu},                                          
       journal={IEEE Signal Processing Letters},                                
       volume={21},                                                             
       number={11},                                                             
       pages={1389--1393},                                                      
       year={2014},                                                             
       publisher={IEEE}                                                         
     }                                                                          
    """                                                                         
    # Short variables are named as they are in the paper                        
    m = label_1_count                                                           
    n = predictions_sorted_transposed.shape[1] - m                              
    positive_examples = predictions_sorted_transposed[:, :m]                    
    negative_examples = predictions_sorted_transposed[:, m:]                    
    k = predictions_sorted_transposed.shape[0]                                  
                                                                                
    tx = np.empty([k, m], dtype=np.float)                                       
    ty = np.empty([k, n], dtype=np.float)                                       
    tz = np.empty([k, m + n], dtype=np.float)                                   
    for r in range(k):                                                          
        tx[r, :] = compute_midrank(positive_examples[r, :])                     
        ty[r, :] = compute_midrank(negative_examples[r, :])                     
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])         
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n             
    v01 = (tz[:, :m] - tx[:, :]) / n                                            
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m                                      
    sx = np.cov(v01)                                                            
    sy = np.cov(v10)                                                            
    delongcov = sx / m + sy / n                                                 
    return aucs, delongcov  


def calc_pvalue(aucs, sigma):                                                   
    """Computes log(10) of p-values.                                            
    Args:                                                                       
       aucs: 1D array of AUCs                                                   
       sigma: AUC DeLong covariances                                            
    Returns:                                                                    
       log10(pvalue)                                                            
    """                                                                         
    l = np.array([[1, -1]])                                                     
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))          
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth, sample_weight):               
    assert np.array_equal(np.unique(ground_truth), [0, 1])                      
    order = (-ground_truth).argsort()                                           
    label_1_count = int(ground_truth.sum())                                     
    if sample_weight is None:                                                   
        ordered_sample_weight = None                                            
    else:                                                                       
        ordered_sample_weight = sample_weight[order]                            
                                                                                
    return order, label_1_count, ordered_sample_weight  


def delong_roc_variance(ground_truth, predictions, sample_weight=None):         
    """                                                                         
    Computes ROC AUC variance for a single set of predictions                   
    Args:                                                                       
       ground_truth: np.array of 0 and 1                                        
       predictions: np.array of floats of the probability of being class 1      
    """                                                                         
    order, label_1_count, ordered_sample_weight = \
        compute_ground_truth_statistics(ground_truth, sample_weight)        
    predictions_sorted_transposed = predictions[np.newaxis, order]              
    aucs, delongcov = fastDeLong(predictions_sorted_transposed,                 
                                 label_1_count, ordered_sample_weight)          
    assert len(aucs) == 1, ("There is a bug in the code, please forward this"   
                            " to the developers")                               
    return aucs[0], delongcov 

                                                                                
"""                                                                             
Description: Computes AUC, AUC covariance, and an AUC confidence interval       
                                                                                
Parameters:                                                                     
    y_true: Numpy 1d array of the true classifications (binary, 0/1) of the data
    y_pred: Numpy 1d array of the predicted data scores. Higher values should   
        correspond to 1's in the y_true array, and lower values should          
        correspond to 0's.                                                      
    ci_confidence: Confidence score (in [0, 1]) of confidence interval          
"""                                                                             
def compute_auc(y_true: 'np.ndarray[np.int32]', y_pred: 'np.ndarray[np.int32]', 
        ci_confidence: float = None):                                           
    auc, auc_cov = delong_roc_variance(y_true, y_pred)                          
    if ci_confidence is None:                                                   
        return auc, auc_cov, None                                               
    if auc_cov == 0.0:                                                          
        return auc, auc_cov, [auc, auc]                                         
    auc_std = np.sqrt(auc_cov)                                                  
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - ci_confidence) / 2)          
    ci = stats.norm.ppf(lower_upper_q, loc = auc, scale = auc_std)              
    ci[ci > 1] = 1                                                              
    return auc, auc_cov, ci  
