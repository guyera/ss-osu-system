"""
Stores items by class an provides feeds
"""
#########################################################################
# Copyright 2008-2022 by Raytheon BBN Technologies.  All Rights Reserved
#########################################################################

import pandas as pd

from enums import NoveltyType, InstanceType
from item import Item

class Data:
    def __init__(self):
        pass

    def load_train(self, csv_file, log):
        df = pd.read_csv(csv_file)
        self.train_count = len(df)
        self.train_valid = []
        self.train_invalid = []
        self.train_known = []
        self.train_novel = []
        self.known_svo = set()
        self.known_sv = set()
        for _tuple in df.itertuples():
            item = Item(_tuple)
            if not item.valid:
                self.train_invalid.append(item)
                continue
            self.train_valid.append(item)
            s = _tuple.subject_id
            v = _tuple.verb_id
            o = _tuple.object_id
            if item.instance_type == InstanceType.BOTH_PRESENT:
                if s > 0 and v > 0 and o > 0:
                    self.known_svo.add((s, v, o))
            if item.instance_type != InstanceType.S_MISSING:
                if s > 0 and v > 0 and o <= 0:
                    self.known_sv.add((s, v))
            if item.is_non_comb_novel():
                self.train_novel.append(item)
            else:
                self.train_known.append(item)
        log.write(f'Found {self.train_count} items, {len(self.train_valid)} valid.\n')
        if self.train_invalid:
            log.write(f'Skipped {len(self.train_invalid)} invalid items.\n')
            for item in self.train_invalid:
                log.write(f'Skipping invalid item: ')
                item.debug_print(log)
        log.write(f'In train.csv, found {len(self.train_known)} non-novel and '
                  f'{len(self.train_novel)} novel instances.\n')
        log.write(f'Found {len(self.known_svo)} distinct known SVO combinations\n')
        log.write(f'Found {len(self.known_sv)} distinct known SV combinations\n')


    def load_val(self, csv_file, log):
        self.val_invalid = []
        self.val_ignored = []
        self.val_bad_combo = []
        self.bins = {novelty_type: [] for novelty_type in NoveltyType}
        df = pd.read_csv(csv_file)
        self.val_count = len(df)
        log.write(f'Found {self.val_count} total val items.\n')
        for _tuple in df.itertuples():
            item = Item(_tuple, val=True)
            if not item.valid:
                self.val_invalid.append(item)
                continue
            # Filtering out s-1v0 cases at Tom's suggestion
            ### Removing this as part of the all_valid change
            # if item.s == -1 and item.v == 0:
            #     self.val_ignored.append(item)
            #     continue

            # if item.novelty_type == NoveltyType.NOVEL_COMB_V_MISS_O:
            #     if ((item.instance_type == InstanceType.BOTH_PRESENT and
            #          (item.s, item.v, item.o) in self.known_svo) or
            #         (item.instance_type == InstanceType.O_MISSING and
            #          (item.s, item.v) in self.known_sv)):
            #             self.val_bad_combo.append(item)
            #             ## LAR HACK!! Disabling this test!!
            #             #continue
            self.bins[item.novelty_type].append(item)
        if self.val_invalid:
            log.write(f'Skipping {len(self.val_invalid)} invalid val items:\n')
            for item in self.val_invalid:
                log.write(f'Skipping invalid item: ')
                item.debug_print(log)
            log.write('\n')
        if self.val_ignored:
            log.write(f'Skipping {len(self.val_ignored)} valid val items that we do not want to use:\n')
            for item in self.val_ignored:
                log.write(f'Skipping ignored item: ')
                item.debug_print(log)
            log.write('\n')
        # if self.val_bad_combo:
        #     ## LAR HACK
        #     log.write(f'Found but not skipping {len(self.val_bad_combo)} apparently but not actually NOVEL_COMB:\n')
        #     #log.write(f'Skipping {len(self.val_bad_combo)} apparently but not actually NOVEL_COMB:\n')
        #     for item in self.val_bad_combo:
        #         log.write(f'Skipping non NOVEL_COMB: ')
        #         item.debug_print(log)
        #     log.write('\n')

    def print_val_counts(self, log):
        log.write(f'\nCount of all valid val items: {self.val_count}\n')
        bins_total = 0
        for novelty_type in NoveltyType:
            bins_total += len(self.bins[novelty_type])
        log.write(f'Total assigned to novelty bins: {bins_total}\n')
        log.write(f'\nCounts by novelty type\n')
        log.write(f'  Train Known: {len(self.train_known)}\n')
        for novelty_type in NoveltyType:
            log.write(f'  Type {novelty_type.value} {novelty_type.name}: {len(self.bins[novelty_type])}\n')


    def debug_list(self, log):
        for novelty_type in NoveltyType:
            log.write(f'\n*** Type {novelty_type.value} {novelty_type.name} ***\n')
            for item in self.bins[novelty_type]:
                item.debug_print(log)

    def print_known_combos_buggy(self, combos, log):
        first_combo = combos[0]
        s_val = first_combo[0]
        for combo in combos:
            if combo[0] > s_val:
                log.write('\n')
                s_val = combos[0]
            log.write(f' {combo}')
        log.write('\n')

    def print_known_combos(self, combos, log):
        log.write('\n')
        for combo in combos:
            log.write(f'  {combo}\n')

    def debug_print_known(self, log):
        log.write(f'Known SVO combinations:')
        triples = sorted(self.known_svo)
        if triples:
            self.print_known_combos(triples, log)
        log.write(f'Known SV combinations:')
        duples = sorted(self.known_sv)
        if duples:
            self.print_known_combos(duples, log)
