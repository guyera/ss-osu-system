###########################################################
# Supervised Novelty Detector Training for Sail-On Phase 2
###########################################################
# Author: Thomas Noel 
###########################################################
# According to Sailon Document:
#    
#    Given: Set of supervised novelty instances
#
#    Train 3 MLPs, one for each S,V,O, on latent
#    representations and their associated unsupervised
#    novelty scores, 
#
###########################################################

import argparse
import random
import torch
import torchvision
import torchvision.transforms as T
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
import math
from adaptation.MLP_Fusion import MLP_Fusion
#import matplotlib.pyplot as plt

from torch.autograd import Variable
from scipy.spatial import distance
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

#from typing import Tuple

# For debugging
##from torch.utils.tensorboard import SummaryWriter

def get_dataloaders(data):
    ''' Constructs the dataloaders '''
    
    i, X, y = data['i'], data['X'], data['y']
    num_folds = data['num_folds']
    num_examples = data['num_examples']
    anom_scores = data['anom_scores']
    img_batch_size = data['img_batch_size']
    num_workers = data['num_workers']

    holdout_lo = round(i * (1/num_folds) * num_examples)
    holdout_hi = round((i+1) * (1/num_folds) * num_examples)

    val_idxs   = [i for i in range(holdout_lo, holdout_hi)]
    train_idxs = [i for i in range(num_examples) if i not in val_idxs]

    train_X_i = X[train_idxs]
    train_a_i = anom_scores[train_idxs]
    train_y_i = y[train_idxs]
        
    val_X_i = X[val_idxs]
    val_a_i = anom_scores[val_idxs]
    val_y_i = y[val_idxs]       

    # Construct the datasets and dataloaders
    train_Xa_i    = torch.hstack((train_X_i,train_a_i))
    train_dataset = torch.utils.data.TensorDataset(train_Xa_i, train_y_i)
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=img_batch_size,
        num_workers=num_workers, 
        pin_memory=True
    )
        
    val_Xa_i    = torch.hstack((val_X_i,val_a_i))
    val_dataset = torch.utils.data.TensorDataset(val_Xa_i, val_y_i)
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=img_batch_size,
        num_workers=num_workers, 
        pin_memory=True
    )

    return train_dataloader, val_dataloader


def train(model, loader, criterion, optimizer, config):                                         
    ''' Train the model '''        
    epochs = config['num_epochs']
    device = config['device']

    total_batches = len(loader) * epochs
    example_ct = 0   
    batch_ct   = 0 

    for epoch in tqdm(range(epochs)):
        for _, (images, labels) in enumerate(loader):

            loss = train_batch(images, labels, 
                               model, optimizer, 
                               criterion, device)
            example_ct +=  len(images)
            batch_ct   +=  1

            # Report loss every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)             


def train_log(loss, example_ct, epoch):
    ''' Print out loss info '''
    loss = float(loss)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")


def train_batch(images, labels, model, optimizer, criterion, device):                                   
    ''' Train the model on one batch '''
    images, labels = images.to(device), labels.to(device)   

    # Forward pass ->
    outputs = model(images)
    outputs = torch.reshape(outputs, (-1,))
    labels  = torch.reshape(labels,  (-1,))
    loss = criterion(outputs, labels.float())

    # Backward pass <-
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()
    
    return loss 


def test(model, test_loader, data_params):                                                                   
    ''' Test the model on the test set '''
    device = data_params['device']

    model.eval()
    # NOTE: If testing is done multiple times throughout training, need to make 
    # sure that model.train() is called somewhere before continuing to train.

    with torch.no_grad():
        correct, total = 0, 0
        y_hat, y = torch.tensor([]).to(device), torch.tensor([]).to(device)
        for images, labels in test_loader:
           images, labels = images.to(device), labels.to(device)
           outputs = model(images)
           # torch.round since binary classification
           #_, predicted = torch.max(outputs.data, 1)
           predicted = torch.round(outputs.data).T[0]
           total += labels.size(0)
           correct += (predicted == torch.round(labels.T)).sum().item()
          
           y_hat = torch.cat((y_hat, outputs))
           y     = torch.cat((y, labels)) 

        # Put these on the cpu so that sklearn can 
        # do its magic
        y = y.cpu()
        y = y.detach().numpy()
        y_hat = y_hat.cpu()
        y_hat = y_hat.detach().numpy()
        auc = roc_auc_score(np.round(y), np.round(y_hat))

        print(f"Accuracy of the model on the {total} " +
              f"test images: {100 * correct / total}%")

        print(f"AUC of the model on the {total} " + 
              f"test images: {auc}")
 
    return auc


def train_supervised_model(X, anom_scores, y):
    ''' 
    Train a supervised MLP anomaly detector
    using X, a, and y
    '''  

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     

    num_folds = 5
    num_examples = X.shape[0]
    num_features = X.shape[1]
    num_workers = 1
    img_batch_size= 64
    
    num_epochs = 15
    lrate = 0.0546 
    mu = 0.9    

    scores = []    
    models = []   
 
    # Doing 5-fold cross-validation
    # and then averaging each member of the ensemble to 
    # get a score.
    for i in range(num_folds):   
        data_params = dict(
            i=i, num_folds=num_folds,
            num_examples=num_examples,
            X=X, anom_scores=anom_scores, y=y,
            img_batch_size=img_batch_size,
            num_workers=num_workers,
            num_epochs=num_epochs,
            device=device
        )

        # Load the data
        train_dataloader, val_dataloader = get_dataloaders(data_params)
        
        # Get the model
        model_i = MLP_Fusion(n=num_features)        
        model_i.train()
        model_i.to(device)       
 
        # Define the loss function and the optimizer
        criterion = nn.BCELoss()
        optimizer = optim.SGD(model_i.parameters(), lr=lrate, momentum=mu)
        
        # Train the model
        train(model_i, train_dataloader, criterion, optimizer, data_params)

        # Compute the AUC
        auc = test(model_i, val_dataloader, data_params)        

        # Store the model and AUC for this split
        scores.append(auc)
        models.append(model_i)

    mean_AUC = sum(scores) / len(scores)

    return mean_AUC, models    
    

def train_supervised_models(S_X, V_X, O_X, S_a, V_a, O_a, S_y, V_y, O_y):
    ''' 
    Expects feature tensors {S,V,O}_X, unsupervised novelty score 
    vectors (tensors) {S,V,O}_a, and target vectors (tensors) {S,V,O}_y. 
    Trains an MLP model for each S, V, and O and returns supervised 
    anomaly scores for each instance.

    Returns: S_AUC_scores, V_AUC_scores, O_AUC_scores 
    '''
     
    S_AUC_scores, S_models = train_supervised_model(S_X, S_a, S_y)
    V_AUC_scores, V_models = train_supervised_model(V_X, V_a, V_y)
    O_AUC_scores, O_models = train_supervised_model(O_X, O_a, O_y)

    return ((S_AUC_scores, V_AUC_scores, O_AUC_scores),
            (S_models, V_models, O_models))


def eval_supervised_ensemble(models, X, a):
    '''
    Computes a supervised anomaly score for each instance
    given an ensemble of MLPs.
    '''

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     

    ensemble_size = len(models)    

    num_examples = X.shape[0]
    num_features = X.shape[1]

    X_concat = torch.hstack((X, a)) 

    scores = None

    for model in models:   
        model.eval()
        scores_i = model(X_concat)
        scores_i = torch.reshape(scores_i, (-1,1))
        
        if scores is None:
            scores = scores_i
        else:
            scores = torch.hstack((scores, scores_i))
    
    # Now that we have a scores tensor of size (num_examples x num_models),
    # compute the average for every instance.
    nov_scores = torch.mean(scores, 1)
    
    return nov_scores
                

def eval_supervised(S_X, V_X, O_X, S_a, V_a, O_a, models):
    '''
    Evaluates the given instances via the given models ensemble.
    NOTE: models contains 3 lists of models packed into a tuple
                ([s_model_1, s_model_2, …, s_model_5],
                 [v_model_1, v_model_2, …, v_model_5],
                 [o_model_1, o_model_2, …, o_model_5])
    
    Returns: A tuple of the following form indicating instancewise
             novelty scores.
                 (subject_nov_scores, verb_nov_scores, object_nov_scores)
    '''
    
    S = 0
    V = 1
    O = 2

    subject_nov_scores = eval_supervised_ensemble(models[S], S_X, S_a)
    verb_nov_scores    = eval_supervised_ensemble(models[V], V_X, V_a)
    object_nov_scores  = eval_supervised_ensemble(models[O], O_X, O_a) 

    return (subject_nov_scores, verb_nov_scores, object_nov_scores)
