#################################################
# Formulate a set of images for which we would
# like oracle feedback given a budget F.
#################################################

import numpy as np
import torch
import torchvision

def select_queries(budget, P_type, P_N, A_S, A_V, A_O):
    ''' 
    Formulates a set of images for which we'll be asking
    for oracle feedback.

    Returns: A list of image indices indicating which images
             shall be sent to the oracle.
    '''

    TYPE_1, TYPE_2 = 0, 1
    TYPE_3, type_5 = 2, 4
    
    N = len(P_type)

    types_retained = [TYPE_1, TYPE_2, TYPE_3, type_5]
    P_type_new = P_type[:,types_retained]
 
    # Since a lower index element was removed, need
    # to readjust the type 5 index
    type_5 = 3

    for i in range(N):
        P_type_new[i] = P_type_new[i] / torch.sum(P_type_new[i])

    w_s = torch.reshape(P_type_new[:,TYPE_1], (N,1))
    w_v = torch.reshape(P_type_new[:,TYPE_2] + P_type_new[:,type_5], (N,1))
    w_o = torch.reshape(P_type_new[:,TYPE_3], (N,1))

    u_s = w_s * A_S
    u_v = w_v * A_V
    u_o = w_o * A_O

    U_mat = torch.hstack((torch.hstack((u_s,u_v)),u_o)) 
    u_vec, _ = torch.max(U_mat,1)

    # Sort u_vec
    u_vec_sorted, indices = torch.sort(u_vec,0,descending=True)

    selection = indices[0:budget]

    # TODO: -> "If P(type) assigned high probabilities to more than
    #           one type, do the top 5 queries include cases of 
    #           those types?"

    return selection.tolist()


# For debugging and tests
if __name__ == '__main__':
    # TODO: Set this up with real data.

    budget = 10
    N = 100
    num_types = 5

    P_type = torch.rand(N, num_types)
    for i in range(N):
        P_type[i] = P_type[i] / torch.sum(P_type[i])

    P_N = torch.rand(N, 1)
    
    s_n = (1/torch.rand(N,1))*torch.rand(N,1) 
    v_n = (1/torch.rand(N,1))*torch.rand(N,1) 
    o_n = (1/torch.rand(N,1))*torch.rand(N,1) 

    selections = select_queries(budget, P_type, P_N, s_n, v_n, o_n) 
    print(selections) 
