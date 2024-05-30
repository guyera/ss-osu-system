from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import random
import pickle

from data import Data
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

from test_soucce_data import TestSourceData

random.seed(42)

def process_dataset(train_csv, test_csv, log):
    data = Data()
    log.write(f'*** Loading train data {train_csv.name} ***\n\n')
    data.load_train(train_csv, log)
    # data.debug_print_known(log)
    log.write(f'\n***Loading test data {test_csv.name}  ***\n\n')
    data.load_test(test_csv, log)
    data.print_test_counts(log)
    data.debug_list(log)
    return data


def main():
    p = ArgumentParser()
    p.add_argument('--train_csv', default='/nfs/hpc/share/sail_on3/final/osu_train_cal_val/train.csv')
    p.add_argument('--test_csv', default='/nfs/hpc/share/sail_on3/final/osu_train_cal_val/valid.csv')
    p.add_argument('--pickle_out_file', default='/nfs/hpc/share/sail_on3/final/osu_train_cal_val/test_trials/osu_test_sandbox/test_source_data_valid.pkl')
    p.add_argument('--log_file', default='/nfs/hpc/share/sail_on3/final/osu_train_cal_val/test_trials/osu_test_sandbox/dataset_analysis_valid.log')
    args = p.parse_args()
    train_csv = Path(args.train_csv)
    test_csv = Path(args.test_csv)
    pickle_out_file = Path(args.pickle_out_file)
    log_file = Path(args.log_file)
    with open(log_file, 'w') as log:
        data = process_dataset(train_csv, test_csv, log)
    test_source_data = TestSourceData(data.bins)
    with open(pickle_out_file, 'wb') as handle:
        pickle.dump(test_source_data, handle)


if __name__ == '__main__':
    main()

    # python process_dataset.py --train_csv '../../../../sail_on3/svo/umd/dataset_v4_2_train.csv' --test_csv '../../../../sail_on3/svo/dataset_v4_2_val_final.csv' --pickle_out_file '../../../../sail_on3/svo/umd_test_sandbox/test_source_data_v4_2_all_valid_val.pkl' --log_file '../../../../sail_on3/svo/umd_test_sandbox/dataset_analysis_v4_2_all_valid.log'
