# This software was created by Oregon State University under Army Research
# Office (ARO) Award Number W911NF-22-2-0149. ARO, as the Federal awarding
# agency, reserves a royalty-free, nonexclusive and irrevocable right to
# reproduce, publish, or otherwise use this software for Federal purposes, and
# to authorize others to do so in accordance with 2 CFR 200.315(b).

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


def select_queries(budget, p_ni_query_threshold, P_N, bbox_counts):
    ''' 
    Formulates a set of images for which we'll be asking
    for oracle feedback.

    Returns: A list of image indices indicating which images
             shall be sent to the oracle.
    '''
    # Assert length of list is nonzero
    if P_N.shape[0] == 0:
        raise ValueError("Must pass in a nonzero number of queries")
    sorted_indices = torch.argsort(P_N, descending=True)
    # line 40 is Just for Exp 2 and Exp 5 where sorting is not required
    # sorted_indices = torch.randperm(len(P_N), device=P_N.device)
    non_empty_mask = bbox_counts > 0
    sorted_non_empty_mask = non_empty_mask[sorted_indices]
    sorted_non_empty_indices = sorted_indices[sorted_non_empty_mask]
    budgeted_selection = sorted_non_empty_indices[:budget]
    budgeted_p_ni_values = P_N[budgeted_selection]
    selection = budgeted_selection[budgeted_p_ni_values >= p_ni_query_threshold]
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
