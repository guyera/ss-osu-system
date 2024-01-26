"""
Stores items by class an provides feeds
"""
#########################################################################
# Copyright 2008-2022 by Raytheon BBN Technologies.  All Rights Reserved
#########################################################################

import ast
import pandas as pd

from enums import NoveltyType, EnvironmentType, Id2noveltyType, get_subnovelty_varname
from item import Item

class Data:
    def __init__(self):
        pass

    def load_train(self, csv_file, log):
        df = pd.read_csv(csv_file, quotechar='"', skipinitialspace=True)
        self.train_count = len(df)
        self.train_valid = []
        self.train_invalid = []
        self.known_species = set()
        self.known_activities = set()
        self.kwown_environments = set()

        for _tuple in df.itertuples():
            item = Item(_tuple)

            if not item.valid:
                self.train_invalid.append(item)
                continue

            self.train_valid.append(item)
            self.known_species.add(item.agent1_name)
            self.kwown_environments.add(item.environment_id)

            assert len(item.activities) <= 1
            
            if len(item.activities) > 0:
                self.known_activities.add(item.activities[0])
            
        log.write(f'Found {self.train_count} items, {len(self.train_valid)} valid.\n')
        if self.train_invalid:
            log.write(f'Skipped {len(self.train_invalid)} invalid items.\n')
            for item in self.train_invalid:
                log.write(f'Skipping invalid item: ')
                item.debug_print(log)


    def load_test(self, csv_file, log):
        self.test_invalid = []
        self.bins = {novelty_type: [] for novelty_type in NoveltyType}
        for sub_nov in EnvironmentType:
            self.bins[sub_nov] = []
        df = pd.read_csv(csv_file, quotechar='"', skipinitialspace=True, low_memory=False)
        self.test_count = len(df)
        log.write(f'Found {self.test_count} total test items.\n')
        for _tuple in df.itertuples():
            item = Item(_tuple, val=True)
            if not item.valid:
                self.test_invalid.append(item)
                continue

            self.bins[item.novelty_type].append(item)
            if item.environment_id is not None and item.environment_id != 0:
                if item not in self.bins[get_subnovelty_varname(item.environment_id, item.novelty_type_id)]:
                    self.bins[get_subnovelty_varname(item.environment_id, item.novelty_type_id)].append(item)
        
        # for kb in self.bins:
        #     print('////////// bins:', kb, '******', 'size', len(self.bins[kb]))
            
        if self.test_invalid:
            log.write(f'Skipping {len(self.test_invalid)} invalid test items:\n')
            for item in self.test_invalid:
                log.write(f'Skipping invalid item: ')
                item.debug_print(log)
            log.write('\n')

    def print_test_counts(self, log):
        log.write(f'\nCount of all valid test items: {self.test_count}\n')
        bins_total = 0
        for novelty_type in NoveltyType:
            bins_total += len(self.bins[novelty_type])
        log.write(f'Total assigned to novelty bins: {bins_total}\n')
        log.write(f'\nCounts by novelty type\n')
        log.write(f'  Train: {len(self.train_valid)}\n')
        for novelty_type in NoveltyType:
            log.write(f'  Type {novelty_type.value} {novelty_type.name}: {len(self.bins[novelty_type])}\n')
        
        # write environment subnovelty (type 6) bins to the log file
        for env_subnov in EnvironmentType:
            log.write(f'  Subnovelty type {env_subnov.value} {env_subnov.name}: {len(self.bins[env_subnov])}\n')


    def debug_list(self, log):
        for novelty_type in NoveltyType:
            log.write(f'\n*** Type {novelty_type.value} {novelty_type.name} ***\n')
            # print('>>>> novelty type:', novelty_type, '   size of bin:', len(self.bins[novelty_type]))
            log.write(f"{'filename':30} {'agent1':10}:{'count'} {'agent2':10}:{'count'} {'agent3':10}:{'count'} {'activities'} {'env_id'} \n")

            for item in self.bins[novelty_type]:
                item.debug_print(log)
        # write environment subnovelty (type 6) to the log
        for env_subnov in EnvironmentType:
            log.write(f'\n*** Type {env_subnov.value} {env_subnov.name} ***\n')
            log.write(f"{'filename':30} {'agent1':10}:{'count'} {'agent2':10}:{'count'} {'agent3':10}:{'count'} {'activities'} {'env_id'} \n")
            for item in self.bins[env_subnov]:
                item.debug_print(log)


