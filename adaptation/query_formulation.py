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

    # Assert that arguments have same length
    if not (P_N.shape[0] == len(A_S) and len(A_S) == len(A_V) and len(A_V) == len(A_O)):
        raise ValueError("P_N, A_S, A_V, and A_O must have same length")
    
    N = len(P_N)

    TYPE_1, TYPE_2 = 0, 1
    TYPE_3, type_5 = 2, 4
    
    types_retained = [TYPE_1, TYPE_2, TYPE_3, type_5]
    P_type_new = P_type[types_retained]
 
    # Since a lower index element was removed, need
    # to readjust the type 5 index
    type_5 = 3

    P_type_new = P_type_new / torch.sum(P_type_new)

    A_S_min = smallest_non_none(A_S)
    A_V_min = smallest_non_none(A_V)
    A_O_min = smallest_non_none(A_O)

    S_nones = 0
    V_nones = 0
    O_nones = 0

    for i in range(N):
        if A_S[i] == None:
            A_S[i] = A_S_min - 1
            S_nones += 1
        
        if A_V[i] == None:
            A_V[i] = A_V_min - 1
            V_nones += 1
        
        if A_O[i] == None:
            A_O[i] = A_O_min - 1
            O_nones += 1
        
    if budget > (N - S_nones) or budget > (N - V_nones) or budget > (N - O_nones):
        raise ValueError("Invalid budget; too many NoneType Instances")

    w_s = P_type_new[TYPE_1]
    w_v = P_type_new[TYPE_2] + P_type_new[type_5]
    w_o = P_type_new[TYPE_3]

    S_tensor = torch.unsqueeze(torch.tensor(A_S), dim=1)
    V_tensor = torch.unsqueeze(torch.tensor(A_V), dim=1)
    O_tensor = torch.unsqueeze(torch.tensor(A_O), dim=1)

    u_s = w_s * S_tensor
    u_v = w_v * V_tensor
    u_o = w_o * O_tensor

    U_mat = torch.hstack((torch.hstack((u_s,u_v)),u_o)) 
    u_vec, _ = torch.max(U_mat,1)

    # Sort u_vec
    u_vec_sorted, indices = torch.sort(u_vec,0,descending=True)

    selection = indices[0:budget]

    return selection.tolist()

if __name__ == "__main__":
    budget = 2
    P_type = torch.tensor([0.2,0.2,0.2,0.2,0.2])
    P_N = torch.tensor([[0.01],[0.4],[0.78],[0.21],[0.5]])
    A_S = [None, -4283, -3, -5.5, None]
    A_V = [-5, -2.2, 7, 90, None]
    A_O = [None, 12, 17.23, -8, -0.01]
    select_queries(budget, P_type, P_N, A_S, A_V, A_O)
