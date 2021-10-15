#!/usr/bin/env python3

# Thomas Noel
# Oregon State University
#
# NOTE: This constitutes the adaptation phase of the SAIL-ON 
# phase 2 plan. 
#
# INPUT:  -> Latent representations z_{j,h} for j=1,...,5 and h=1,2,3. 
#            There are 5 networks and 3 h-values corresponding to 
#            subject (S), verb (V), or object (O).
#
#         -> Anomaly scores a_{j,h} for j=1,...,5 and h=1,2,3. 
#
# OUTPUT: 
#
# 
# NOTES:
# -> Build 5 (number of networks) x 3 (one for each S,V,O) = 15 MLPs
#    that take in (z_{j,h}, a_{j,h}) and predict a binary anomaly
#    score. 
#
# -> Will tentatively use max-anomaly query strategy
#
# -> Will retain the examples already seen (even nominals) so that 
#    these can be added to the train set that includes anomalies
#    once we get to this phase of training. For example, if the 
#    number of anomaly instances exceeds the number of nominals in
#    the set of all examples queried so far, we can draw from 
#    the set of known nominals (pre-red-button) to balance our 
#    training set.
#    
# -> For sure can ask for feedback on one of S,V,O for a request
#    for labels in each batch of size F, where one S,V,O is 
#    selected on a per-instance basis in the feedback requests.
#
#    (e.g. If requests for S/V are made for the same image, it
#    counts as 2 items for the budget F, rendering the remaining
#    query budget F_{n+1} = F_{n} - 2, where F_n is the remaining budget 
#    at the point this double query is made.)
#
# -> **We will have dependency on the model class, so make sure that 
#    that is accessible in the final submission. 

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
import matplotlib.pyplot as plt
import wandb

from torch.autograd import Variable
from scipy.spatial import distance
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression
from typing import Tuple
from pyod.models.loda import LODA
from torch.utils.tensorboard import SummaryWriter
from MLP_Fusion import MLP_Fusion
from tqdm import tqdm

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def model_pipeline(hyperparameters):

    project = "SAIL-ON Adaptation"
    
    # Start wandb
    # NOTE: wandb is only here during dev. REMOVE WHEN DONE WITH DEV.
    with wandb.init(project=project, config=hyperparameters):
        # Want logging to match execution
        config = wandb.config
       
        #oracle_percents = list(range(1, config.max_percent_to_oracle+1, config.to_oracle_interval))
        oracle_percents = [10]
        #oracle_percents = [100] # Pure oracle experiment
        for oracle_percent in oracle_percents:     
            oracle_proportion = 0.01 * oracle_percent            

            # make model, data, and optimizer
            model, train_loader, test_loader, criterion, optimizer = make(oracle_proportion, config)
            print(model)
            print("Oracle Percent: {}".format(oracle_proportion))

            # Train the model
            train(model, train_loader, criterion, optimizer, config)

            # Test the model's performance
            auc = test(model, test_loader)

            wandb.log({"AUC": auc, "Oracle_Percents": oracle_percent})

    return model
        

def make(oracle_proportion, config):
    
    # Make the model
    # (NOTE: The backbone model returns logits)
    backbone_model = ResNet(config.num_channels, config.nom_classes, block=ResNetBasicBlock, depths=[2,2,2,2])       
    backbone_model.load_state_dict(torch.load(config.model_path))
    
    # Attach MLP at end of ResNet-18
    backbone = backbone_model.get_encoder()
    model = AnomalyDetector(backbone)
    
    if config.thawed_layers != 'all': 
        model = freeze_layers(backbone, model, config)
    
    model = model.to(device)
    model.train()
    
    # Make the data
    train, test  = get_data(oracle_proportion, backbone_model, config)
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader  = make_loader(test,  batch_size=config.batch_size)

    # Make the loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), 
                          lr=config.learning_rate, momentum=config.momentum)      
   
    return model, train_loader, test_loader, criterion, optimizer
 

def get_data(oracle_proportion, backbone_model, config):
    ''' Make the Train and Test Datasets '''

    transform = T.Compose([
        T.ToTensor()
    ])

    try:
        raw_train_dataset = eval("torchvision.datasets.{}('data', train=True, transform=transform, download=True)".format(config.dataset))
        test_dataset   = eval("torchvision.datasets.{}('data', train=False, transform=transform, download=True)".format(config.dataset))
    except:
        raise NotImplementedError('{} is not a dataset that is currently supported.'.format(config.dataset)) 

    # Relabel the targets to indicate nominal (0) or anomaly (1) status
    raw_train_targets = raw_train_dataset.__dict__['targets']
    raw_train_dataset.__dict__['targets'] = [1 if label in config.anom_labels else 0 for label in raw_train_targets]    
    binary_train_targets = raw_train_dataset.__dict__['targets']

    test_targets = test_dataset.__dict__['targets']
    test_dataset.__dict__['targets'] = [1 if label in config.anom_labels else 0 for label in test_targets]

    val_low   = int(0.8*len(raw_train_dataset))
    val_high  = len(raw_train_dataset)

    train_indices = torch.tensor([i for i in range(0, val_low) if binary_train_targets[i] == 0])
    
    oracle_params = dict(
        selection_strategy = config.selection_strategy,
        max_proportion = config.max_percent_to_oracle,
        temp_fn = config.temp_fn,
        proportion = oracle_proportion,
        backbone_model = backbone_model, # if applicable
        data = raw_train_dataset,        # if applicable
        batch_size = config.batch_size,  # if applicable
        val_low = val_low,
        val_high = val_high)              

    val_indices = get_oracle_indices(oracle_params) 


    # TODO: Make function that analyzes anomaly choices here;
    # Should print out 
    #
    #    1) Confusion matrix (at first binary, later 
    #       implement for all 10 classes)
    #
    #    2) # Class instances detected v. Oracle Iterations
    #       (at first binary, later implement for all 10 classes)
        
    conf_matrix = generate_confusion_matrices(config,
                                              oracle_proportion, 
                                              val_indices, raw_train_targets)

    # Stitch together known train examples and labeled oracle examples
    if config.with_train:
        final_train_indices = torch.cat((train_indices, val_indices))
    else:
        final_train_indices = val_indices
    
    final_train_dataset = torch.utils.data.Subset(raw_train_dataset, final_train_indices) 

    return final_train_dataset, test_dataset 


def generate_confusion_matrices(config, proportion, indices, targets):
    # Note that since max_oracle_proportion is < 40%, that 
    # we should theoretically be be getting anomalies 
    # consistently using our max anomaly selection strategy
    
    # NOTE: Might need to take alternate approach if generating
    # confusion matrix for balanced or random selection strategies

    # Starting with the max anomaly selection strategy

    anom_labels = config.anom_labels
    classes = config.cifar_classes
    selection_strategy = config.selection_strategy
    split = config.split

    # 0.4 is the max anomaly proportion possible according to 
    # our experimental setup
    if proportion < 0.4:
        # Use indices to index into targets vector
        
        # Generate binary labels and numpyify the two target vectors
        bin_targets = np.array([1 if label in anom_labels else 0 for label in targets])
        targets = np.array(targets)

        # Index into the target vectors
        y_true = bin_targets[indices]
        
        # NOTE: This will be different for the balanced selection approach
        if selection_strategy == 'batch_max_anom':
            y_pred = np.ones((len(indices),))
        elif selection_strategy == 'batch_balanced':
            # TODO
            y_pred = [1 if i % 2 == 0 else 0 for i in range(len(indices))]        
            y_pred = np.array(y_pred)

        bin_confusion_matrix = confusion_matrix(y_true, y_pred)
        mc_confusion_matrix = multiclass_confusion(split, anom_labels, classes, targets[indices], y_pred)
         
        mc_confusion_matrix.to_csv('confusion_matrices/conf_matrix_{}_split_{}.csv'.format(selection_strategy, split), index=False)
    else:
        raise ValueError("Max anomaly proportion is 0.4")


def multiclass_confusion(split, anom_labels, classes, y_true, y_pred):
    ''' NOTE: y_true[i] \in {0,1,2,...,9}, whereas y_pred[i] \in {0,1}, 
        where 0 is 'nominal' and 1 is 'anomalous' '''
   
    NOMINAL   = 0
    ANOMALOUS = 1
 
    conf_matrix_np = np.zeros((2, len(classes)))
    nom_labels = [i for i in range(len(classes)) if i not in anom_labels]

    classes_np = np.array(classes)
    labels  = np.append(np.array(nom_labels), np.array(anom_labels))
    classes_np = classes_np[labels]

    # 2 rows by n columns, where n is the number of classes
    for i, label in enumerate(labels):
        for j, pred_label in enumerate(y_pred):
            # If this is an instance of type <label>
            if y_true[j] == label:
                # Top row is nominal, bottom is anomalous
                if pred_label == NOMINAL:
                    conf_matrix_np[NOMINAL,i]   += 1 
                else:
                    conf_matrix_np[ANOMALOUS,i] += 1       

    conf_matrix = pd.DataFrame(conf_matrix_np, columns=list(classes_np))
    
    return conf_matrix


def get_oracle_indices(params):
    ''' Need to get appropriate indices in range '''
 
    if params['selection_strategy'] == 'random':
        if not os.path.isfile(params['temp_fn']):
            indices = torch.randperm(params['val_high'] - params['val_low']) + params['val_low']
            torch.save(indices, params['temp_fn']) 
        
        # NOTE: indices contains indexes of ALL elements in val_set
        indices = torch.load(params['temp_fn'])
        indices = indices[:int(params['proportion']*len(indices))]                 
    
    elif params['selection_strategy'] == 'batch_max_anom':
        if not os.path.isfile(params['temp_fn']):
            write_ordered_anomaly_indices(params, order='max_anom')
 
        # NOTE: indices contains indexes of ALL elements in val_set
        indices = torch.load(params['temp_fn'])
        indices = indices[:int(params['proportion']*len(indices))]                
        
 

    elif params['selection_strategy'] == 'batch_balanced':
        # Use max logit method to extract anomaly scores
        # and select an equal number of nominal and 
        # anomaly indices
        if not os.path.isfile(params['temp_fn']):
            write_ordered_anomaly_indices(params, order='balanced_anom')

        # NOTE: indices contains indexes of ALL elements in val_set
        indices = torch.load(params['temp_fn'])
        indices = indices[:int(params['proportion']*len(indices))]                
    
    elif params['selection_strategy'] == 'minibatch_max_anom':
        # "" "" but select indices based on pool of ALL examples
        indices = None
    elif params['selection_strategy'] == 'minibatch_balanced':
        # "" "" but select indices based on pool of ALL examples
        indices = None
    else:
        raise NotImplementedError('{} is a selection strategy that is not yet supported.'.format(strategy))

    # Cleanup if this is the last time the index file is accessed
    if int(params['proportion']*100) == params['max_proportion']:
        os.remove(params['temp_fn']) 

    return indices.cpu()


def write_ordered_anomaly_indices(params, order):
    val_indices = [i for i in range(params['val_low'], params['val_high'])]
    val_data = torch.utils.data.Subset(params['data'], val_indices)
            
    # Make a sequential sampler so that that elements are drawn in order
    sampler = torch.utils.data.SequentialSampler(val_data)
            
    # Use max logit method to extract anomaly scores
    # and return indices corresponding to 'most anomalous'
    #
    # MaxC_{θ}(x) = - argmax_{k}θ(k)^T E(x)
    #
    # NOTE: This formula seems to exclude CNN's since it 
    # assumes that the E(x) will be input to a linear layer,
    # when, for example, the ResNet-18 is simply     

    # One large batch 
    batch_size = len(val_data)
    loader = make_loader(val_data, batch_size, shuffle=False, sampler=sampler) 
    model = params['backbone_model']
    model.to(device)
    model.eval()
    with torch.no_grad():
        # Should only be one batch
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # Risheek describes the max logit anomaly signal as,
            # "the negative of the maximum of the logits."
            # Here we're identifying our max logits and 
            # negating them to get our anomaly scores.
            values, _ = torch.max(outputs, dim=1)
            values = -values
            sorted_values, sorted_index_indices = values.sort(descending=True)
            # Have an offset here because our "validation set" starts at 
            # index params['val_low'] in the training set provided by 
            # the PyTorch builtin
            indices = sorted_index_indices + params['val_low']
    
    if order == 'max_anom':
        torch.save(indices, params['temp_fn'])

    elif order == 'balanced_anom':
        # Need to construct sequence of [anom, nom, anom, nom, ...]
        # and save it. Grab anoms from beginning of indices and 
        # nominals from end.
        anom_idx = 0
        nom_idx  = 9999
        balanced_indices = []
        ordered_indices = indices.tolist()
        # While the indices have not yet met at the beginning of the tensor.
        # NOTE: the postcondition is that the anom_idx will be greater than 
        # nom_idx
        while anom_idx < nom_idx:
            balanced_indices.append(ordered_indices[anom_idx])
            balanced_indices.append(ordered_indices[nom_idx])
            anom_idx += 1
            nom_idx  -= 1
        torch.save(torch.tensor(balanced_indices), params['temp_fn'])

    else:
        raise NotImplementedError('{} is not a valid ordering strategy'.format(order))


def make_loader(dataset, batch_size, shuffle=True, sampler=None):
    ''' Make a Dataloader from the Specified Dataset '''
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size,
                                         shuffle=shuffle, sampler=sampler,
                                         pin_memory=True, num_workers=8)
    return loader 


def train(model, loader, criterion, optimizer, config):
    ''' Train the model '''    
    # wandb will watch gradients, weights, etc..
    wandb.watch(model, criterion, log='all', log_freq=10)
    
    total_batches = len(loader) * config.epochs
    example_ct = 0  # Number of examples seen  
    batch_ct   = 0

    for epoch in tqdm(range(config.epochs)):
        for _, (images, labels) in enumerate(loader):

            #if batch_ct == total_batches-1:
            #    my_images = wandb.Image(images[0:20], caption=str(labels[0:20]))
            #    wandb.log({"examples": my_images})

            loss = train_batch(images, labels, model, optimizer, criterion)
            example_ct +=  len(images)
            batch_ct   +=  1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)


def train_batch(images, labels, model, optimizer, criterion):
    ''' Train the model on one batch '''

    images, labels = images.to(device), labels.to(device)   

    # Forward pass ->
    outputs = model(images)
    outputs = torch.reshape(outputs, (-1,))
    loss = criterion(outputs, labels.float())

    # Backward pass <-
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()
    
    return loss 
     

def train_log(loss, example_ct, epoch):
    ''' Send relevant info to weights and biases '''
    loss = float(loss)
    
    # Some wandb logging magic
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")


def test(model, test_loader):
    ''' Test the model on the test set '''
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
           correct += (predicted == labels).sum().item()
           
           y_hat = torch.cat((y_hat, outputs))
           y     = torch.cat((y, labels)) 

        # Put these on the cpu so that sklearn can 
        # do its magic
        y = y.cpu()
        y = y.detach().numpy()
        y_hat = y_hat.cpu()
        y_hat = y_hat.detach().numpy()
        auc = roc_auc_score(y, y_hat)

        print(f"Accuracy of the model on the {total} " +
              f"test images: {100 * correct / total}%")

        print(f"AUC of the model on the {total} " + 
              f"test images: {auc}")
 
        wandb.log({"test_accuracy": correct / total})

    # Save the model in the exchangeable ONNX format
    #torch.onnx.export(model, images, "model.onnx")
    #wandb.save("model.onnx")    

    return auc


def adaptation(Z, a):
''' Making a binary anomaly prediction for each network
    for each S,V,O.
    
    INPUTS:    
        Z: A mx5x3xn tensor of latent vectors, where
           Z[i][j] corresponds to a latent vector from 
           head j of network i, where head j corresponds 
           to S, V, or O. Additionally, m is the number
           of examples
 
        A: A 5x3 matrix of anomaly scores where 
           A[i][j] corresponds to the anomaly score
           corresponding to network i and action part j
           which can be S, V, or O.  

    OUTPUTS:
        P: A 5x3 matrix of probabilities, where probability
           P[i][j] corresponds to the probability that that
           the latent representation from network i corresponds
           to an anomaly in action part j.
'''
    # For i in 1..5
    #     For j in 1..3
    #         z <- Z[i][j]
    #         a <- A[i][j]
    #
 
    pass


class Adaptor:
    ''' Adapts anomaly detection prediction to oracle feedback. '''   

    # TODO: Generalize MLP 
    def __init__(self, S_z_len, V_z_len, O_z_len, D, F):
        '''
        Initializes our feedback adaptor.

        INPUTS:

            S_z_len: The length of the subject latent vectors

            V_z_len: The length of the verb latent vectors

            O_z_len: The length of the object latent vectors 

            D: The dataset on which the models are being trained.
               This should be a tuple containing features X and 
               labels y.

            F: Our query budget per minibatch. Assuming that our 
               query budget is fixed across all trials.
        '''
        self._D = D
        self._F = F
        self._S_MLP = MLP_Fusion(n=S_z_len)
        self._V_MLP = MLP_Fusion(n=V_z_len)
        self._O_MLP = MLP_Fusion(n=O_z_len)


    def get_feedback(self, S, V, O):
        ''' 
        Gets feedback from the oracle. The number of total
        queries per minibatch must be <= F. 

        INPUTS:

          S: A binary vector of length "batch_size". 
             True at index i means that we want subject
             feedback for image i. Each True spends a 
             point of our budget F.

          V: A binary vector of length "batch_size". 
             True at index i means that we want verb
             feedback for image i. Each True spends a 
             point of our budget F.
          
          O: A binary vector of length "batch_size".
             True at index i means that we want object
             feedback for image i. Each True spends a 
             point of our budget F.
        '''
        # Pass selected queries to the oracle and 
        # then add queried instances with their
        # labels to the existing dataset.
        pass
    
   
    def select_queries(self, a):
        ''' 
        Chooses a appropriate queries based on the 
        anomaly scores associated with S, V, and O for 
        each representation in the given minibatch.
        
        INPUTS:

            a: Anomaly scores for S, V, and O for each 
               representation in the given minibatch.

        OUTPUTS:

            S, V, O: Binary vectors 
        '''
        # Could need intermediate query strategy selection 
        # computation here.
        pass


    def new_round(self, Z_new, a_new):
        ''' 
        Gets feedback and trains new MLPs on new dataset 

        INPUTS:

            Z_new: Latent vectors of size batch_size x n.

            a_new: Anomaly scores associated with each latent representation.
               Should be of size batch_size x 3, where we have an anomaly
               score for S, V, and O corresponding to each latent vector.
            
        '''

        pass


    def train_mlp(self, MLP, X, y):
        ''' Train specified MLP on given data '''
        MLP.train()
        # TODO

    def compute_anomaly_scores(self, Z_test, a_test): 
        ''' 
        Returns anomaly scores for S, V, and O for each 
        element in the given test set.

        INPUTS:
        
            Z_test: A torch tensor of test set representations.

            a_test: A torch tensor of test set representation anomaly scores.
                    Assumed order of columns of a_test are S, V, and O, where
                    rows correspond to instances.

        OUTPUTS:

            a_out: A |Z_test| x 3 matrix of anomaly scores        
        ''' 

        # TODO: Determine if we need y_test here (i.e.
        #       if we're going to be measuring test acc 
        #       or AUC or the like.

        X_S = torch.cat((Z_test, a_test[:,0]), dim=1)
        assert(X_S.shape == (Z_test.shape[0],Z_test.shape[1]+1))
        
        X_V = torch.cat((Z_test, a_test[:,1]), dim=1)
        assert(X_V.shape == (Z_test.shape[0],Z_test.shape[1]+1))

        X_O = torch.cat((Z_test, a_test[:,2]), dim=1)
        assert(X_O.shape == (Z_test.shape[0],Z_test.shape[1]+1))

        anom_scores_S = self.compute_anomaly_vector(self._S_MLP, X_S) 
        anom_scores_V = self.compute_anomaly_vector(self._V_MLP, X_V) 
        anom_scores_S = self.compute_anomaly_vector(self._O_MLP, X_O) 
 
        # TODO: Stitch the anomaly scores together and return them :)     

        
    def compute_anomaly_vector(self, MLP, X):
        ''' Compute anomaly scores with the given MLP for each instance '''
        MLP.eval()



# NOTE: Main in this case just acts as an initializer for 
#       preparing 
#          -> synthetic data, 
#          -> hyperparameters 
#          -> etc..
# 
#       Will need to determine how this is expected to
#       fit into the pipeline. Will the adaptation phase
#       have its own main, being a sovereign program, or
#       will it just be expected to be a function with
#       latent vectors and anomaly scores as input 
#       and a final anomaly score as output?
#
#       Also need to consider how rounds will look.
#       -> Round definition? 
#       -> Should I create something that will be 
#       invoked everytime that we have an opportunity to 
#       ask for feedback?
#       -> **How often will we be able to ask for feedback??
#           - Once per batch, F items (F is the requested fixed number)
#       -> How do we make queries? 
#       -> What will be the format of the received queries?

# Maybe, can use CIFAR 10 latents as "Synthetic data"??

def main():
    
    # Build argument parser
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--learning_rate", default=0.0546)
    parser.add_argument("--momentum", default=0.9)
    parser.add_argument("--with_train", default=False)
    parser.add_argument("--num_epochs", default=15)    

    parser.add_argument("--temp_fn", 
                        help="Filename where oracle indices will be stored.", 
                        default='oracle_indices.pt')
     
    # Valid selection strategies include:
    #   'minibatch_max_anom', 
    #   'batch_max_anom', 
    #   'minibatch_balanced', 
    #   'batch_balanced', 
    #   'random'
    parser.add_argument("--selection_strategy", 
                        help="method used to select instances to query", 
                        default='random')

    args = parser.parse_args()

    # Seed the numpy rng
    np.random.seed(0)

    # For development purposes only;
    # TODO: Remove when development is complete
    wandb.login()

    for i in range(NUM_SPLITS):
        hyperparams = dict(
            epochs=int(args.num_epochs),
            num_channels=3,
            batch_size=64,
            momentum=float(args.momentum),
            learning_rate=float(args.learning_rate),
            with_train=bool(args.with_train),
            selection_strategy=args.selection_strategy,
            temp_fn=args.temp_fn+'.pt')

        model = model_pipeline(hyperparams)
                
if __name__ == '__main__':
    main()
