import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from collections import OrderedDict
from functools import partial
from tqdm import tqdm


class MLP_Fusion(nn.Module):

    def __init__(self, n=512):
        super(MLP, self).__init__()
        
        self.n = n
        l1_out_dim = 100
        l2_out_dim = 25

        # n is the dimension of the latent
        # vector. We add 1 to this dimension, and all
        # other inputs, because we want to additionally
        # throw in an anomaly score (a scalar) at each layer.
        self.fc1 = nn.Linear(n+1,          l1_out_dim) 
        self.fc2 = nn.Linear(l1_out_dim+1, l2_out_dim)
        self.fc3 = nn.Linear(l2_out_dim+1, 1) 
    
    def forward(self, x, a):
        ''' 
        Computes the forward pass through all-fusion MLP

        INPUTS:
            x: A minibatch of representations. This 
               should have dimensions batch_size x n
            
            a: A vector of anomaly scores corresponding to
               each instance in the minibatch. Should be 
               vector of length batch_size.
        '''
        x_concat = torch.cat((x, a), dim=1)
        assert(x_concat.shape[0] == x.shape[0] and
               x_concat.shape[1] == x.shape[1]+1) 
        x = F.relu(self.fc1(x_concat))

        x_concat = torch.cat((x, a), dim=1)          
        assert(x_concat.shape[0] == x.shape[0] and
               x_concat.shape[1] == x.shape[1]+1) 
        x = F.relu(self.fc2(x_concat))

        x_concat = torch.cat((x, a), dim=1)                 
        assert(x_concat.shape[0] == x.shape[0] and
               x_concat.shape[1] == x.shape[1]+1) 
        x = torch.sigmoid(self.fc3(x_concat))

        return x
     
    def get_layer(self, layer_requested):
        layer = None
        if layer_requested == 1:
            layer = self.fc1
        elif layer_requested == 2:
            layer = self.fc2
        else:
            raise NotImplementedError('The layer requested is invalid')

        return layer 
