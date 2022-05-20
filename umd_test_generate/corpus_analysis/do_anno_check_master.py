"""
Run this from the umd dir to process the annotation files in umd/annotation
and produce analysis files in umd/annotation_anal.
"""

modified = True
if modified:
    in_file = 'dataset_v4_2_val_master_modified.csv'
    out_file = 'annotations_anal_v4_2_master_modified.txt'
else:
    in_file = 'dataset_v4_2_val_master.csv'
    out_file = 'annotations_anal_v4_2_master.txt'

import pandas as pd
import pdb
from contextlib import redirect_stdout

print_combos_file = None

slices = ['train', 'val']

verb_synonyms = {'carry' : 'carrying',
                 'catch' : 'catching',
                 'ride' : 'riding',
                 'wear' : 'wearing',
                 'looking' : 'watching',
                 'touching' : 'carrying',
                 }

noun_synonyms = {'bow-tie': 'tie',
                 'bike': 'bicycle',
                 'motor bike': 'bicycle',
                 'cap': 'hat',
                 'backpack' : 'bag',
                 }

skip_values = ['absent', 'novel', 'novel/unknown']

df = pd.read_csv(in_file)
# Replace missing SVO_name values with null
df['subject_name'] = df['subject_name'].fillna('NULL')
df['object_name'] = df['object_name'].fillna('NULL')
df['verb_name'] = df['verb_name'].fillna('NULL')

# apply verb synonyms
for key, val in verb_synonyms.items():
    hits = df['verb_name'] == key
    df.loc[hits, 'verb_name'] = val

# apply noun synonyms to subj and obj
for key, val in noun_synonyms.items():
    s_hits = df['subject_name'] == key
    df.loc[s_hits, 'subject_name'] = val
    o_hits = df['object_name'] == key
    df.loc[o_hits, 'object_name'] = val

total_count = len(df)

with open(out_file, 'w') as f:

    f.write(f'====== Data ({total_count} images) =======\n')

    subjects = sorted(list(df['subject_name'].unique()))
    subj_master_ids = sorted(list(df['master_subject_id'].unique()))
    verbs = sorted(list(df['verb_name'].unique()))
    verb_master_ids = sorted(list(df['master_verb_id'].unique()))
    objects = sorted(list(df['object_name'].unique()))
    obj_master_ids = sorted(list(df['master_object_id'].unique()))

    # Print unique vals
    f.write(f'\nUnique individual SVO values:\n\n')
    f.write(f'Subjects (ignoring novel and absent)\n')
    for subject in subjects:
        if subject not in skip_values:
            f.write(f'  {subject}\n')
    f.write(f'\n')
    f.write('Verbs (ignoring novel/unknown)\n')
    for verb in verbs:
        if verb not in skip_values:
            f.write(f'  {verb}\n')
    f.write(f'\n')
    f.write('Objects (ignoring novel and absent)\n')
    for object in objects:
        if object not in skip_values:
            f.write(f'  {object}\n')

    # print names by master ids
    f.write(f'\nSubject names by master id:\n')
    for id in subj_master_ids:
        hits = df['master_subject_id'] == id
        f.write(f'Master id {id}: {list(df[hits]["subject_name"].unique())}\n')
        f.write(f'Master id {id}: IDs {list(df[hits]["subject_id"].unique())}\n')
    f.write(f'\nVerb names by master id:\n')
    for id in verb_master_ids:
        f.write(f'Master id: {id}\n')
        hits = df['master_verb_id'] == id
        hits_df = df[hits]
        f.write(f'  Names:\n')
        names = list(hits_df["verb_name"].unique())
        for name in names:
            name_count = len(hits_df[hits_df["verb_name"]==name])
            f.write(f'    {name}: {name_count}\n')
        f.write(f'  IDs:\n')
        ids = list(hits_df["verb_id"].unique())
        for id in ids:
            id_count = len(hits_df[hits_df["verb_id"]==id])
            f.write(f'    {id}: {id_count}\n')
    f.write(f'\nObject names by master id:\n')
    for id in obj_master_ids:
        hits = df['master_object_id'] == id
        f.write(f'Master id {id}: {list(df[hits]["object_name"].unique())}\n')
        f.write(f'Master id {id}: IDs {list(df[hits]["object_id"].unique())}\n')

    # Look at master_id 0 cases
    if modified:
        f.write(f"\nLooking at the master_id=0 'missing V' cases, with 5 samples of each:\n")
        hits_df = df[df['master_verb_id']==0]
        both_hits = hits_df[(hits_df["subject_ymin"] != -1) & (hits_df["object_ymin"] != -1)]
        both_count = len(both_hits)
        s_only_hits = hits_df[(hits_df["subject_ymin"] != -1) & (hits_df["object_ymin"] == -1)]
        s_only_count = len(s_only_hits)
        o_only_hits = hits_df[(hits_df["subject_ymin"] == -1) & (hits_df["object_ymin"] != -1)]
        o_only_count = len(o_only_hits)
        f.write(f"{both_count=}\n")
        for i, tup in enumerate(both_hits.itertuples()):
            f.write(f"  {tup.new_image_path}\n")
            if i >= 5:
                break
        f.write(f"{s_only_count=}\n")
        for i, tup in enumerate(s_only_hits.itertuples()):
            f.write(f"  {tup.new_image_path}\n")
            if i >= 5:
                break
        f.write(f"{o_only_count=}\n")
        for i, tup in enumerate(o_only_hits.itertuples()):
            f.write(f"  {tup.new_image_path}\n")
            if i >= 5:
                break

        f.write(f"\nLooking at s-only novel V cases:\n")
        hits_df = df[df['master_verb_id']>=9]
        s_only_hits = hits_df[(hits_df["subject_ymin"] != -1) & (hits_df["object_ymin"] == -1)]
        for i, tup in enumerate(s_only_hits.itertuples()):
            f.write(f"  {tup.new_image_path}\n")
            if i >= 5:
                break
        
    # Print unique vals with counts
    f.write(f'\nIndividual SVO values with counts (in decreasing order):\n\n')
    f.write(f'Subjects\n')
    f.write(f"{df['subject_name'].value_counts().to_string()}\n")
    f.write(f'\n')
    f.write(f'Master subject ids\n')
    f.write(f'{df["master_subject_id"].value_counts().to_string()}\n')
    f.write('Verbs\n')
    f.write(f"{df['verb_name'].value_counts().to_string()}\n")
    f.write(f'\n')
    f.write(f'Master verb ids\n')
    f.write(f'{df["master_verb_id"].value_counts().to_string()}\n')
    f.write('Objects\n')
    f.write(f"{df['object_name'].value_counts().to_string()}\n")
    f.write(f'Master object ids\n')
    f.write(f'{df["master_object_id"].value_counts().to_string()}\n')

