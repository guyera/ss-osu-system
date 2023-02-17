from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import random
import json
import pickle

from item import Item
from enums import NoveltyType
from data import Data
from feed import Feed
#########################################################################
# Copyright 2008-2022 by Raytheon BBN Technologies.  All Rights Reserved
#########################################################################

from test_soucce_data import TestSourceData

random.seed(42)

def process_dataset(train_csv, val_csv, log):
    data = Data()
    log.write(f'*** Loading train data {train_csv.name} ***\n\n')
    data.load_train(train_csv, log)
    # data.debug_print_known(log)
    log.write(f'\n***Loading val data {val_csv.name}  ***\n\n')
    data.load_val(val_csv, log)
    data.print_val_counts(log)
    data.debug_list(log)
    return data


def main():
    p = ArgumentParser()
    p.add_argument('--train_csv', default='/nfs/hpc/share/sail_on3/final/osu_train_cal_val/train.csv')
    p.add_argument('--val_csv', default='/nfs/hpc/share/sail_on3/final/osu_train_cal_val/valid.csv')
    p.add_argument('--pickle_out_file', default='/nfs/hpc/share/sail_on3/final/osu_train_cal_val/test_trials/osu_test_sandbox/test_source_data_valid.pkl')
    p.add_argument('--log_file', default='/nfs/hpc/share/sail_on3/final/osu_train_cal_val/test_trials/osu_test_sandbox/dataset_analysis_valid.log')
    args = p.parse_args()
    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv)
    pickle_out_file = Path(args.pickle_out_file)
    log_file = Path(args.log_file)
    with open(log_file, 'w') as log:
        data = process_dataset(train_csv, val_csv, log)
    test_source_data = TestSourceData(data.bins)
    with open(pickle_out_file, 'wb') as handle:
        pickle.dump(test_source_data, handle)


if __name__ == '__main__':
    main()

    # python process_dataset.py --train_csv '../../../../sail_on3/svo/umd/dataset_v4_2_train.csv' --val_csv '../../../../sail_on3/svo/dataset_v4_2_val_final.csv' --pickle_out_file '../../../../sail_on3/svo/umd_test_sandbox/test_source_data_v4_2_all_valid_val.pkl' --log_file '../../../../sail_on3/svo/umd_test_sandbox/dataset_analysis_v4_2_all_valid.log'