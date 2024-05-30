"""
Check if val data includes any combinations of known_ids
that do not also occur in the training.
"""
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

from pathlib import Path
import pandas as pd

def find_combos(df):
    svo_combos = set()
    sv_combos = set()
    for tup in train_df.itertuples():
        s = tup.subject_id
        v = tup.verb_id
        o = tup.object_id
        if s > 0 and v > 0:
            if o > 0:
                svo_combos.add((s, v, o))
            else:
                # including both cases with o==-1 and o==0
                sv_combos.add((s, v))
    return svo_combos, sv_combos

train_csv = Path('../../umd/dataset_v3_4_train.csv')
train_df = pd.read_csv(train_csv)
val_csv = Path('../../umd/dataset_v3_4_val.csv')
val_df = pd.read_csv(val_csv)

train_svo_combos, train_sv_combos = find_combos(train_df)
val_svo_combos, val_sv_combos = find_combos(val_df)

print(f'In train, found {len(train_svo_combos)} unique SVO combos and {len(train_sv_combos)} unique SV combos.')
print(f'In val, found {len(train_svo_combos)} unique SVO combos and {len(train_sv_combos)} unique SV combos.')
print(f'Count of val svo combos that were not found in training: {len(val_svo_combos - train_svo_combos)}')
