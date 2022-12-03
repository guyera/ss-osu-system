#################################################
# Formulate a set of images for which we would
# like oracle feedback given a budget F.
#################################################

import numpy as np
import torch
import torchvision

def smallest_non_none(A):
    smallest = None

    for i in range(len(A)):
        if A[i] != None:
            if smallest == None or A[i] < smallest:
                smallest = A[i]

    return smallest


def select_queries(budget, P_type, P_N, A_S, A_V, A_O):
    ''' 
    Formulates a set of images for which we'll be asking
    for oracle feedback.

    Returns: A list of image indices indicating which images
             shall be sent to the oracle.
    '''
    # Assert length of list is nonzero
    if P_N.shape[0] == 0:
        raise ValueError("Must pass in a nonzero number of queries")

    indices = torch.argsort(P_N, descending=True)
    selection = indices[:budget]
    return selection.tolist()

if __name__ == "__main__":
    # For Testing
    budget = 2
    P_type = torch.tensor([0.2,0.2,0.2,0.2])
    P_N = torch.tensor([[0.01],[0.4],[0.78],[0.21],[0.5]])
    A_S = [None, -4283, -3, -5.5, None]
    A_V = [-5, -2.2, 7, 90, None]
    A_O = [None, 12, 17.23, -8, -0.01]
    print(select_queries(budget, P_type, P_N, A_S, A_V, A_O))
