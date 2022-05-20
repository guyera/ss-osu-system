"""
Class to parse classification.csv lines into top-3 triples
"""
#########################################################################
# Copyright 2021-2022 by Raytheon BBN Technologies.  All Rights Reserved
#########################################################################

import itertools
import numpy as np

class ClassFileReader():
    def __init__(self):
        self.num_subj = 5
        self.num_verb = 8
        self.num_obj = 12
        self.triple_list = list(itertools.product(np.arange(-1, self.num_subj),
                                                  np.arange(0, self.num_verb),
                                                  np.arange(-1, self.num_obj)))
        self.get_triple = {}
        for i, triple in enumerate(self.triple_list):
            self.get_triple[i] = triple

    def get_answers(self, class_line):
        chunks = class_line.split(',')
        prob_strs = chunks[1:]
        probs = np.array([float(prob_str) for prob_str in prob_strs])
        ind_1, ind_2, ind_3 = np.argsort(-1.0 * probs)[:3]
        ans_1_s, ans_1_v, ans_1_o = self.get_triple[ind_1]
        ans_2_s, ans_2_v, ans_2_o = self.get_triple[ind_2]
        ans_3_s, ans_3_v, ans_3_o = self.get_triple[ind_3]
        return ans_1_s, ans_1_v, ans_1_o, ans_2_s, ans_2_v, ans_2_o, ans_3_s, ans_3_v, ans_3_o
