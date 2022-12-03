import csv
import os



import pandas as pd
import itertools
import numpy as np

class ClassFileReader():
    def __init__(self):
        self.num_subj = 5
        self.num_verb = 8
        self.num_obj = 12
        self.triple_list = list(itertools.product(np.arange(0, self.num_subj),
                                                  np.arange(0, self.num_verb),
                                                  np.arange(0, self.num_obj)))
    
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



def match(corr_s, corr_o, corr_v, ans_s, ans_o, ans_v):
    'This will need to get fancier'
    if corr_s == -1:
        return ans_o == corr_o
    elif corr_o == -1:
        if corr_v == -1:
            return ans_s == corr_s
        else:
            return ans_s == corr_s and ans_v == corr_v
    else:
        return ans_s == corr_s and ans_o == corr_o and ans_v == corr_v




def percent_string(num, denom):
    return f'{100 * num / denom:6.2f}%'


def score_test(test_df, class_lines, class_file_reader):
    # import ipdb; ipdb.set_trace()

    # The metadata file is not currently used.
    # import ipdb; ipdb.set_trace()
    total_pre_red = total_post_red = total_novel = total = 0
    pre_red_top_1_hits = post_red_top_1_hits = pre_red_top_3_hits = post_red_top_3_hits = 0
    novel_top_1_hits = novel_top_3_hits = 0
    test_tuples = test_df.itertuples()
    red_button = False
    red_button_pos = -1
    sys_declare_pos = -1
    novel = True

    for pos, (test_tuple, class_line) in enumerate(zip(test_tuples, class_lines[1:])):
        # print(test_tuple.new_image_path, class_line.split(',')[0])
        if novel:
            red_button = True
            if red_button_pos == -1:
                red_button_pos = pos 
        corr_s = test_tuple.subject_id
        corr_o = test_tuple.object_id
        corr_v = test_tuple.verb_id
        ans_1_s, ans_1_v, ans_1_o, ans_2_s, ans_2_v, ans_2_o, ans_3_s, ans_3_v, ans_3_o = (
            class_file_reader.get_answers(class_line)
        )
        top_1 = match(corr_s, corr_o, corr_v, ans_1_s, ans_1_o, ans_1_v)
        top_3 = (match(corr_s, corr_o, corr_v, ans_1_s, ans_1_o, ans_1_v) or
                 match(corr_s, corr_o, corr_v, ans_2_s, ans_2_o, ans_2_v) or
                 match(corr_s, corr_o, corr_v, ans_3_s, ans_3_o, ans_3_v))

        # print(ans_1_s, ans_1_v, ans_1_o, ans_2_s, ans_2_v, ans_2_o, ans_3_s, ans_3_v, ans_3_o)
        # print(top_1, top_3)
        total += 1
        if red_button:
            total_post_red += 1
            if top_1:
                post_red_top_1_hits += 1
            if top_3:
                post_red_top_3_hits += 1
        else:
            total_pre_red += 1
            if top_1:
                pre_red_top_1_hits += 1
            if top_3:
                pre_red_top_3_hits += 1
        if novel:
            total_novel += 1
            if top_1:
                novel_top_1_hits += 1
            if top_3:
                novel_top_3_hits += 1
    total_top_1_hits = pre_red_top_1_hits + post_red_top_1_hits
    total_top_3_hits = pre_red_top_3_hits + post_red_top_3_hits

    if post_red_top_1_hits > 0:
        post_top_1_score = percent_string(post_red_top_1_hits, total_post_red)
        post_top_3_score = percent_string(post_red_top_3_hits, total_post_red)
    else:
        post_top_1_score = "    NA"
        post_top_3_score = "    NA"
    total_top_1_score = percent_string(total_top_1_hits, total)
    total_top_3_score = percent_string(total_top_3_hits, total)
    if total_novel > 0:
        novel_top_1_score = percent_string(novel_top_1_hits, total_novel)
        novel_top_3_score = percent_string(novel_top_3_hits, total_novel)
    else:
        novel_top_1_score = "    NA"
        novel_top_3_score = "    NA"
    print(f'   Test           Top-1     Top-3')
    print(f' Results    {total_top_1_score}    {total_top_3_score}')
    



test_df = pd.read_csv('./val_files/dataset_v4_2_cal_corruption.csv')
class_lines = open('./results/events_baseline0.csv').read().splitlines()
class_file_reader = ClassFileReader()
score_test(test_df, class_lines, class_file_reader)