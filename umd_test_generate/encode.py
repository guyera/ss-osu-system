#########################################################################
# Copyright 2008-2022 by Raytheon BBN Technologies.  
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this code and associated documentation files (the "Code"), to deal
# in the Code without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Code, and to permit persons to whom the Code is
# furnished to do so, subject to the following conditions:
#
# This copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Code.
#
# BBN makes no warranties regarding the fitness of the code for any use or
# purpose.
#########################################################################

import numpy as np
import itertools
import pandas as pd

df = pd.read_csv('../../umd/annotations/val_dataset_v1_train.csv')
print(f'Max subject id: {df["subject_id"].max()}')
print(f'Max berb id: {df["verb_id"].max()}')
print(f'Max object id: {df["object_id"].max()}')

nsub = 5
nobj = 13
nvrb = 8

triple_list_umd = list(itertools.product(np.arange(-1, nsub+1), np.arange(0, nvrb+1), np.arange(-1, nobj+1)))
triple_count = len(triple_list_umd)
print(f'{triple_count=}')
# for i, triple in enumerate(triple_list_umd):
#     print(f'{i}: {triple}')

triple_dict = {}
for i, triple in enumerate(triple_list_umd):
    triple_dict[triple] = i

pred = np.zeros(triple_count)

pred[triple_dict[(-1, 0, 2)]] = 0.25
pred[triple_dict[(-1, 0, 4)]] = 0.15

print(','.join([f'{val:f}' for val in pred]))

# triple_list = []
# # First handle cases for s=-1, where we know v=0 and obj>=0
# for o in range(0, nobj+1):
#     triple_list.append((-1, 0, o))
# # Now the rest
# for s in range(0, nsub+1):
#     for v in range (0, nvrb+1):
#         for o in range(-1, nobj+1):
#             triple_list.append((s, v, o))
# print(f'{len(triple_list)=}')
# for i, triple in enumerate(triple_list):
#     print(f'{i}: {triple}')
