"""
Process the training and validation annotation files
supplied by UMD
and produce a textual analysis file.
Set the input_dir, annotation_version, and out_file values appropriately.
"""

import pandas as pd
import pdb
from contextlib import redirect_stdout

source_dir = "../../../../umd"
annotation_version = "dataset_v4_2_"
out_file = "../../../../umd/annotation_anal_test.txt"

print_combos_file = None

slices = ['train', 'val']

skip_values = ['absent', 'novel', 'novel/unknown']

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

def collapse_synonyms(df):
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

def find_unique_names(df):
    subjects = sorted(list(df['subject_name'].unique()))
    verbs = sorted(list(df['verb_name'].unique()))
    objects = sorted(list(df['object_name'].unique()))
    return subjects, verbs, objects

def print_unique_vals(subjects, verbs, objects):
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

def print_vals_with_counts(subjects, verbs, objects):
    # Print unique vals with counts
    f.write(f'\nIndividual SVO values with counts (in decreasing order):\n\n')
    f.write(f'Subjects\n')
    f.write(f"{df['subject_name'].value_counts().to_string()}\n")
    f.write(f'\n')
    f.write('Verbs\n')
    f.write(f"{df['verb_name'].value_counts().to_string()}\n")
    f.write(f'\n')
    f.write('Objects\n')
    f.write(f"{df['object_name'].value_counts().to_string()}\n")
                            
    
    

with open(out_file, 'w') as f:

    for slice in slices:

        df = pd.read_csv(f'{source_dir}/{annotation_version}{slice}.csv')
        # Replace missing SVO_name values with null
        df['subject_name'] = df['subject_name'].fillna('NULL')
        df['object_name'] = df['object_name'].fillna('NULL')
        df['verb_name'] = df['verb_name'].fillna('NULL')

        total_count = len(df)
        
        f.write(f'====== {slice} Data ({total_count} images) =======\n')

        f.write(f"\n***Contents before replacing synonyms***\n")

        subjects, verbs, objects = find_unique_names(df)
        print_unique_vals(subjects, verbs, objects)
        print_vals_with_counts(subjects, verbs, objects)

        collapse_synonyms(df)
        f.write(f"\n***Contents after replacing synonyms***\n")

        subjects, verbs, objects = find_unique_names(df)
        print_unique_vals(subjects, verbs, objects)
        print_vals_with_counts(subjects, verbs, objects)

        if slice == 'train':
            novel_subject = len(df[df['subject_name']=='novel'])
            novel_object = len(df[df['object_name']=='novel'])
            f.write(f'\nNon-novel train count: {total_count - novel_subject - novel_object}\n')


        f.write(f'\nCounts of SVO triples (in decreasing order):\n\n')
        # print counts
        df2 = df.copy()
        df2['svo'] = df2['subject_name'].astype(str) + '_' + df2['verb_name'].astype(str) + '_' + df2['object_name'].astype(str)
        # f.write('SVO')
        # f.write(df2['svo'])
        #df2 = df2.groupby(['subject_name', 'verb_name', 'object_name'])['svo'].nunique().reset_index()
        svos = df2['svo'].value_counts()
        f.write(f'{svos.to_string()}\n')

        f.write(f'\n')
        
        # maybe print combos
        if print_combos_file:
            with open(f'annotation_{slice}_combos.txt', 'w') as combos_f:
                for subject in subjects:
                    for object in objects:
                        for verb in verbs:
                            combos_f.write(f'{subject},{verb},{object}\n')

