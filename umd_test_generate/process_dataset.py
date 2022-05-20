from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import random
import json
import pickle

from item import Item
from enums import InstanceType, NoveltyType
from data import Data
from feed import Feed
#########################################################################
# Copyright 2008-2022 by Raytheon BBN Technologies.  All Rights Reserved
#########################################################################

from test_soucce_data import TestSourceData

random.seed(42)

# test_header = 'new_image_path,subject_name,subject_id,master_subject_id,' \
#               'object_name,object_id,master_object_id,verb_name,verb_id,master_verb_id,'\
#               'image_width,image_height,subject_ymin,subject_xmin,subject_ymax,subject_xmax,' \
#               'object_ymin,object_xmin,object_ymax,object_xmax,novel'

def process_dataset(train_csv, val_csv, log):
    data = Data()
    log.write(f'*** Loading train data {train_csv.name} ***\n\n')
    data.load_train(train_csv, log)
    data.debug_print_known(log)
    log.write(f'\n***Loading val data {val_csv.name}  ***\n\n')
    data.load_val(val_csv, log)
    data.print_val_counts(log)
    data.debug_list(log)
    return data


def main():
    p = ArgumentParser()
    p.add_argument('--train_csv', default='../../../umd/dataset_v4_2_train.csv')
    p.add_argument('--val_csv', default='../../../umd/dataset_v4_2_val_master.csv')
    p.add_argument('--pickle_out_file', default='../../umd_test_sandbox/test_source_data_v4_2_all_valid_val.pkl')
    p.add_argument('--log_file', default='../../umd_test_sandbox/dataset_analysis_v4_2_all_valid.log')
    args = p.parse_args()
    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv)
    pickle_out_file = Path(args.pickle_out_file)
    log_file = Path(args.log_file)
    with open(log_file, 'w') as log:
        data = process_dataset(train_csv, val_csv, log)
    test_source_data = TestSourceData(data.train_known, data.bins)
    with open(pickle_out_file, 'wb') as handle:
        pickle.dump(test_source_data, handle)


if __name__ == '__main__':
    main()