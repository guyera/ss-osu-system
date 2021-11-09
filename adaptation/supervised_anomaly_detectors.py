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
#from .MLP_Fusion import MLP_Fusion
#from noveltydetection.utils import compute_partial_auc
from .MLP_Fusion import MLP_Fusion
import noveltydetectionfeatures
#import matplotlib.pyplot as plt

from torch.autograd import Variable
from scipy.spatial import distance
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

#from typing import Tuple

# For debugging
##from torch.utils.tensorboard import SummaryWriter

def get_split_dataloaders(data):
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

    #train_nom_idxs  = list(torch.nonzero(torch.flatten(torch.tensor(train_y_i).clone().detach())))
    train_nom_idxs  = list(torch.nonzero(torch.flatten(train_y_i.clone().detach())))
    train_anom_idxs = [i for i in range(len(train_y_i)) if i not in train_nom_idxs]       

    train_y_i_nom = train_y_i[train_nom_idxs]
    train_y_i_anom = train_y_i[train_anom_idxs]
    train_X_i_nom = train_X_i[train_nom_idxs]
    train_X_i_anom = train_X_i[train_anom_idxs]
    train_a_i_nom = train_a_i[train_nom_idxs]
    train_a_i_anom = train_a_i[train_anom_idxs]
    
    val_X_i = X[val_idxs]
    val_a_i = anom_scores[val_idxs]
    val_y_i = y[val_idxs]       

    #val_nom_idxs  = list(torch.nonzero(torch.flatten(torch.tensor(val_y_i).clone().detach())))
    val_nom_idxs  = list(torch.nonzero(torch.flatten(val_y_i.clone().detach())))
    val_anom_idxs = [i for i in range(len(val_y_i)) if i not in val_nom_idxs]

    val_y_i_nom = val_y_i[val_nom_idxs]
    val_y_i_anom = val_y_i[val_anom_idxs]
    val_X_i_nom = val_X_i[val_nom_idxs]
    val_X_i_anom = val_X_i[val_anom_idxs]
    val_a_i_nom = val_a_i[val_nom_idxs]
    val_a_i_anom = val_a_i[val_anom_idxs]

    # Construct the datasets and dataloaders
    train_Xa_i_nom = torch.hstack((train_X_i_nom,train_a_i_nom))
    train_dataset_nom = torch.utils.data.TensorDataset(train_Xa_i_nom, train_y_i_nom)
    train_dataloader_nom = torch.utils.data.DataLoader(
        dataset=train_dataset_nom,
        batch_size=img_batch_size,
        num_workers=num_workers, 
        pin_memory=False
    )
    
    train_Xa_i_anom = torch.hstack((train_X_i_anom,train_a_i_anom))
    train_dataset_anom = torch.utils.data.TensorDataset(train_Xa_i_anom,train_y_i_anom)
    train_dataloader_anom = torch.utils.data.DataLoader(
        dataset=train_dataset_anom,
        batch_size=img_batch_size,
        num_workers=num_workers, 
        pin_memory=False
    )
        
    val_Xa_i_nom = torch.hstack((val_X_i_nom,val_a_i_nom))
    val_dataset_nom = torch.utils.data.TensorDataset(val_Xa_i_nom,val_y_i_nom)
    val_dataloader_nom = torch.utils.data.DataLoader(
        dataset=val_dataset_nom,
        batch_size=img_batch_size,
        num_workers=num_workers, 
        pin_memory=False
    )

    val_Xa_i_anom = torch.hstack((val_X_i_anom,val_a_i_anom))
    val_dataset_anom = torch.utils.data.TensorDataset(val_Xa_i_anom,val_y_i_anom)
    val_dataloader_anom = torch.utils.data.DataLoader(
        dataset=val_dataset_anom,
        batch_size=img_batch_size,
        num_workers=num_workers, 
        pin_memory=False
    )

    val_Xa_i = torch.hstack((val_X_i, val_a_i))
    val_dataset = torch.utils.data.TensorDataset(val_Xa_i, val_y_i)
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=img_batch_size,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_dataloader_nom, train_dataloader_anom, val_dataloader_nom, val_dataloader_anom, val_dataloader


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
        pin_memory=False
    )
        
    val_Xa_i    = torch.hstack((val_X_i,val_a_i))
    val_dataset = torch.utils.data.TensorDataset(val_Xa_i, val_y_i)
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=img_batch_size,
        num_workers=num_workers, 
        pin_memory=False
    )

    return train_dataloader, val_dataloader


def train(model, nom_loader, anom_loader, criterion, optimizer, config):                                         
    ''' Train the model '''        
    epochs = config['num_epochs']
    device = config['device']

    total_batches = (len(nom_loader)+len(anom_loader)) * epochs
    example_ct = 0   
    batch_ct   = 0 

    progress = tqdm(                                                                                                                     
        total = epochs,
        desc = 'Training classifier',
        leave = False
    )

    # Guarantees that loader_a is the shorter of the two.
    # Done so that there will be no need to handle a 
    # StopIteration Exception.
    if len(nom_loader) <= len(anom_loader):
        loader_a = nom_loader
        loader_b = anom_loader
    else:
        loader_a = anom_loader
        loader_b = nom_loader

    for epoch in range(epochs):
        progress.set_description("epoch {}".format(epoch))
        
        #nom_iterator = iter(nom_loader)
        b_iterator = iter(loader_b)

        #for _, (anom_images, anom_labels) in enumerate(anom_loader):
        for i, (a_images, a_labels) in enumerate(loader_a):
            
            #try:
            #    (nom_images, nom_labels) = next(nom_iterator)
            #except StopIteration:
            #    nom_iterator = iter(nom_loader)
            #    (nom_images, nom_labels) = next(nom_iterator)
            
            (b_images, b_labels) = next(b_iterator)
            
            loss = train_batch(a_images, a_labels, 
                               b_images, b_labels,
                               model, optimizer, 
                               criterion, device)

            #loss = train_batch(nom_images, nom_labels, 
            #                   anom_images, anom_labels,
            #                   model, optimizer, 
            #                   criterion, device)
            
            
            #example_ct +=  len(nom_images) + len(anom_images)
            example_ct +=  len(a_images) + len(b_images)
            batch_ct   +=  1

            # Report loss every 25th batch
            #if ((batch_ct + 1) % 25) == 0:
            #    train_log(loss, example_ct, epoch)             


def train_log(loss, example_ct, epoch):
    ''' Print out loss info '''
    loss = float(loss)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")


def train_batch(a_images, a_labels, b_images, b_labels, model, optimizer, criterion, device):                                   
    ''' Train the model on one batch '''
    #nom_images, nom_labels = nom_images.to(device), nom_labels.to(device)   
    #anom_images, anom_labels = anom_images.to(device), anom_labels.to(device)

    a_images, a_labels = a_images.to(device), a_labels.to(device)   
    b_images, b_labels = b_images.to(device), b_labels.to(device)

    # Forward pass ->
    #nom_outputs = model(nom_images)
    #anom_outputs = model(anom_images)    
    a_outputs = model(a_images)
    b_outputs = model(b_images)    

    #outputs = torch.cat((nom_outputs, anom_outputs), dim=0)
    #labels  = torch.cat((nom_labels,  anom_labels), dim=0)
    outputs = torch.cat((a_outputs, b_outputs), dim=0)
    labels  = torch.cat((a_labels,  b_labels), dim=0)

    outputs = torch.reshape(outputs, (-1,))
    labels  = torch.reshape(labels,  (-1,))
    loss = criterion(outputs, labels.float())

    # Backward pass <-
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()
    
    return loss 


def test(model, test_loader, test_nom_loader, test_anom_loader, data_params):                                                                   
    ''' Test the model on the test set '''
    device = data_params['device']

    model.eval()
    # NOTE: If testing is done multiple times throughout training, need to make 
    # sure that model.train() is called somewhere before continuing to train.

    with torch.no_grad():
        correct, total = 0, 0
        nom_correct, nom_total = 0, 0
        anom_correct, anom_total = 0, 0
        y_hat, y = torch.tensor([]).to(device), torch.tensor([]).to(device)     
        y_hat_nom, y_nom = torch.tensor([]).to(device), torch.tensor([]).to(device)     
        y_hat_anom, y_anom = torch.tensor([]).to(device), torch.tensor([]).to(device)     

        for images, labels in test_nom_loader:
           images, labels = images.to(device), labels.to(device)
           outputs = model(images)
           # torch.round since binary classification
           #_, predicted = torch.max(outputs.data, 1)
           predicted = torch.round(outputs.data).T[0]
           nom_total += labels.size(0)
           labels = labels.float()
           nom_correct += (predicted == torch.round(labels.T)).sum().item()
          
           y_hat_nom = torch.cat((y_hat_nom, outputs))
           y_nom     = torch.cat((y_nom, labels)) 

        for images, labels in test_anom_loader:
           images, labels = images.to(device), labels.to(device)
           outputs = model(images)
           # torch.round since binary classification
           #_, predicted = torch.max(outputs.data, 1)
           predicted = torch.round(outputs.data).T[0]
           anom_total += labels.size(0)
           labels = labels.float()
           anom_correct += (predicted == torch.round(labels.T)).sum().item()
          
           y_hat_anom = torch.cat((y_hat_anom, outputs))
           y_anom     = torch.cat((y_anom, labels)) 

        for images, labels in test_loader:
           images, labels = images.to(device), labels.to(device)
           outputs = model(images)
           # torch.round since binary classification
           #_, predicted = torch.max(outputs.data, 1)
           predicted = torch.round(outputs.data).T[0]
           total += labels.size(0)
           labels = labels.float()
           correct += (predicted == torch.round(labels.T)).sum().item()
          
           y_hat = torch.cat((y_hat, outputs))
           y     = torch.cat((y, labels)) 
        
        scores = y_hat
        nom_scores  = y_hat_nom
        anom_scores = y_hat_anom

        if len(nom_scores) == 0:
            nom_scores = torch.tensor([[0]])

        if len(anom_scores) == 0:
            anom_scores = torch.tensor([[1]]) 

        # Since anomalies are zero, these are put in an
        # order contrary to the method signature.
        auc = compute_partial_auc(anom_scores, nom_scores) 

        print(f"Accuracy of the supervised ensemble member on the {total} " +
              f"test images: {100 * correct / total}%")

        print(f"AUROC (at fpr=0.25) of the supervised ensemble member on the {total} " + 
              f"test images: {auc}")
 
    return auc, scores


def compute_partial_auc(nominal_scores, novel_scores):
    nominal_trues = torch.zeros_like(nominal_scores)
    novel_trues = torch.ones_like(novel_scores)

    trues = torch.cat((nominal_trues.cpu(), novel_trues.cpu()), dim = 0).data.cpu().numpy()
    scores = torch.cat((nominal_scores.cpu(), novel_scores.cpu()), dim = 0).data.cpu().numpy()

    auc = roc_auc_score(trues, scores, max_fpr = 0.25)

    return auc


def train_supervised_model(X, anom_scores, y):
    ''' 
    Train a supervised MLP anomaly detector
    using X, a, and y
    '''  

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     

    num_folds = 5
    num_examples = len(X)
    num_features = len(X[0])
    num_workers = 1
    img_batch_size= 64
    
    num_epochs = 15
    lrate = 0.0546 
    mu = 0.9    

    scores = None
    aucs   = []    
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
        dataloaders = get_split_dataloaders(data_params)
        
        train_nom_dl = dataloaders[0]
        train_anom_dl = dataloaders[1]
        val_nom_dl = dataloaders[2]
        val_anom_dl = dataloaders[3]     
        val_dl = dataloaders[4]  
 
        # Get the model
        model_i = MLP_Fusion(n=num_features)        
        model_i.train()
        model_i.to(device)       
 
        # Define the loss function and the optimizer
        criterion = nn.BCELoss()
        optimizer = optim.SGD(model_i.parameters(), lr=lrate, momentum=mu)
        
        # Train the model
        train(model_i, train_nom_dl, train_anom_dl, criterion, optimizer, data_params)

        # Compute the AUC with max_fpr at 0.25
        auc, scores_split_i = test(model_i, val_dl, val_nom_dl, val_anom_dl, data_params)        

        # Store the model and AUC for this split
        if scores is None:
            scores = scores_split_i
        else:
            scores = torch.vstack((scores,scores_split_i))
        aucs.append(auc)
        models.append(model_i)

    mean_AUC = sum(aucs) / len(aucs)

    return mean_AUC, scores, models    
    

def train_supervised_models(S_X, V_X, O_X, S_a, V_a, O_a, S_y, V_y, O_y):
    ''' 
    Expects feature tensors {S,V,O}_X, unsupervised novelty score 
    vectors (tensors) {S,V,O}_a, and target vectors (tensors) {S,V,O}_y. 
    Trains an MLP model for each S, V, and O and returns supervised 
    anomaly scores for each instance.

    Returns: S_AUC_scores, V_AUC_scores, O_AUC_scores 
    '''

    S_auc, S_nov_scores, S_models = train_supervised_model(S_X, S_a, S_y)
    V_auc, V_nov_scores, V_models = train_supervised_model(V_X, V_a, V_y)
    O_auc, O_nov_scores, O_models = train_supervised_model(O_X, O_a, O_y)

    return ((S_auc, V_auc, O_auc),
            (S_nov_scores, V_nov_scores, O_nov_scores),
            (S_models, V_models, O_models))


def eval_supervised_ensemble(models, X, a):
    '''
    Computes a supervised anomaly score for each instance
    given an ensemble of MLPs.

    X: a list of feature vectors

    a: a list of anomaly scores corresponding to each feature vector    

    returns a novelty score vector of the same length as the input.

    '''

    if len(X) != len(a):
        raise ValueError("Feature list and anomaly score list must be same length.")

    N = len(X)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     

    ensemble_size = len(models)

    none_idxs = []

    X_o = X.copy()
    a_o = a.copy()

    # Need to try torch.stack on X
    
    # Gather None idxs
    for i in range(N):
        if a[i] is None or X[i] is None:
            none_idxs.append(i)

    # Remove the None elements from X and a
    for i, idx in enumerate(none_idxs):
        X.pop(idx)
        a.pop(idx)

        # Adjust idxs since we are popping from list.
        # This implies that we must reintroduce elements
        # from the end to the beginning (test this carefully)
        for j in range(i+1,len(none_idxs)):
            none_idxs[j] -= 1

    a = torch.tensor(a)
    a = torch.reshape(a, (len(a),1))
    #a = torch.unsqueeze(torch.tensor(a), dim=1)
    ##a = torch.unsqueeze(a.clone().detach(), dim=1)
    
    #a.to(device)

    # This will be a list of tensors
    X = [feature_vec.tolist() for feature_vec in X]
    X = torch.tensor(X)
    #X.to(device)

    num_examples = X.shape[0]
    num_features = X.shape[1]

    X_concat = torch.hstack((X, a)) 
    X_concat = X_concat.to(device)

    scores = None

    for model in models:   
        model.to(device)
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
    nov_scores = nov_scores.tolist()
    
    reversed_none_idxs = reversed(none_idxs)    

    for idx in reversed_none_idxs:
        nov_scores.insert(idx, None)   
 
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


# For debugging and tests
if __name__ == '__main__':
    # TODO: Set this up with real data.

    num_examples = 1142
    num_features = 512
    S_X = torch.rand(num_examples, num_features)
    V_X = torch.rand(num_examples, num_features)
    O_X = torch.rand(num_examples, num_features)

    S_a = torch.rand(num_examples, 1)
    V_a = torch.rand(num_examples, 1)
    O_a = torch.rand(num_examples, 1)
    S_y = torch.sigmoid(torch.abs(torch.normal(0,1,size=(num_examples, 1))))
    S_y = torch.bernoulli(S_y)
    V_y = torch.sigmoid(torch.abs(torch.normal(0,1,size=(num_examples, 1))))
    V_y = torch.bernoulli(V_y)
    O_y = torch.sigmoid(torch.abs(torch.normal(0,1,size=(num_examples, 1))))
    O_y = torch.ones_like(torch.bernoulli(O_y))
    AUC, scores, models = train_supervised_models(S_X,V_X,O_X,
                                          S_a,V_a,O_a,
                                          S_y,V_y,O_y)
    
    S_a = [random.uniform(0,1) for _ in range(num_examples)]
    V_a = [random.uniform(0,1) for _ in range(num_examples)]
    O_a = [random.uniform(0,1) for _ in range(num_examples)]
    S_none_idxs = random.sample([i for i in range(num_examples)], 5)
    V_none_idxs = random.sample([i for i in range(num_examples)], 5)
    O_none_idxs = random.sample([i for i in range(num_examples)], 5)
    S_X = S_X.tolist()
    V_X = V_X.tolist()
    O_X = O_X.tolist() 

    for i in range(num_examples):
        if i in S_none_idxs:
            S_X[i] = None
            S_a[i] = None
        if i in V_none_idxs:
            V_X[i] = None
            V_a[i] = None
        if i in O_none_idxs:
            O_X[i] = None
            O_a[i] = None

    scores = eval_supervised(S_X, V_X, O_X, S_a, V_a, O_a, models)
    print("S Scores: {}".format(scores[0][0:5]))
    print("V Scores: {}".format(scores[1][0:5]))
    print("O Scores: {}".format(scores[2][0:5]))

    print("S score shape: {}".format(scores[0].shape))
    print("V score shape: {}".format(scores[1].shape))
    print("O score shape: {}".format(scores[2].shape)) 
