"""
Load my test source data pickle file.
Then process the single_df.csv files in the old UMD tests,
and report how many examples of each class are found.
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

import pickle
from pathlib import Path
import pandas as pd
from enums import NoveltyType

data_source_file = Path('../test_source_data_v3_4_TNT_val.pkl')
val_tests_dir = Path('../../umd/downloads_2021-11-18/val_tests_v3_3')

# First load the source data

with open(data_source_file, 'rb') as handle:
    test_source_data = pickle.load(handle)

known = set()
for item in test_source_data.known:
    known.add(item.new_image_path)

novel = {novelty_type : set() for novelty_type in NoveltyType if novelty_type != NoveltyType.NO_NOVEL}
for novelty_type in NoveltyType:
    if novelty_type == NoveltyType.NO_NOVEL:
        continue
    for item in test_source_data.novelty_bins[novelty_type]:
        novel[novelty_type].add(item.new_image_path)

print(f'Found {len(known)} known.')
for novelty_type in NoveltyType:
    if novelty_type == NoveltyType.NO_NOVEL:
        continue
    print(f'Found {len(novel[novelty_type])} {novelty_type.name}.')

other = set()
triples = {}
for csv_file in val_tests_dir.glob('*single_df.csv'):
    df = pd.read_csv(csv_file)
    print(f'{csv_file.name}   ', end='')
    known_count = novel_s_count = novel_v_count = novel_o_count = other_count = total = 0
    for tup in df.itertuples():
        total += 1
        item = tup.new_image_path
        triple = (tup[3], tup[9], tup[6])
        if item in known:
            known_count += 1
        elif item in novel[NoveltyType.NOVEL_S]:
            novel_s_count += 1
        elif item in novel[NoveltyType.NOVEL_V]:
            novel_v_count += 1
        elif item in novel[NoveltyType.NOVEL_O]:
            novel_o_count += 1
        else:
            other_count += 1
            other.add(item)
            triples[item] = triple
    print(f'{known_count:4d} {novel_s_count:4d} {novel_v_count:4d} '
          f'{novel_o_count:4d} {other_count:4d} {total:4d}')

print(f'\nOther items ({len(other)} total):')
for item in sorted(list(other)):
    triple= triples[item]
    print(f'   {item:32}  {triple}')
