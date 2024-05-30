"""
Takes a tests directory and a results directory, and prints scores
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

from pathlib import Path
from argparse import ArgumentParser
import json
import pandas as pd
from class_file_reader import ClassFileReader

# num_subj = 5
# num_verb = 8
# num_obj = 13
# triple_list = list(itertools.product(np.arange(-1, num_subj+1),
#                                      np.arange(0, num_verb+1),
#                                      np.arange(-1, num_obj+1)))
# get_triple = {}
# for i, triple in enumerate(triple_list):
#     get_triple[i] = triple

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

# def get_from_tail(tail):
#     first_comma = tail.find(',')
#     second_comma = tail[first_comma+1:].find(',') + first_comma + 1
#     third_comma = tail[second_comma+1:].find(',') + second_comma + 1
#     s_span = tail[0:first_comma]
#     if s_span == 'None':
#         s = None
#     else:
#         s = int(s_span)
#     v_span = tail[first_comma+2:second_comma]
#     if v_span == 'None':
#         v = None
#     else:
#         v = int(v_span)
#     o_span = tail[second_comma+2:third_comma]
#     if o_span == 'None':
#         o = None
#     else:
#         o = int(o_span)
#     return s, v, o
#
# def get_answers(class_line):
#     first_paren = class_line.find('(')
#     ans_1_s, ans_1_v, ans_1_o = get_from_tail(class_line[first_paren+1:])
#     second_paren = class_line[first_paren+1:].find('(') + first_paren + 1
#     ans_2_s, ans_2_v, ans_2_o = get_from_tail(class_line[second_paren+1:])
#     third_paren = class_line[second_paren+1:].find('(') + second_paren + 1
#     ans_3_s, ans_3_v, ans_3_o = get_from_tail(class_line[third_paren+1:])
#     return ans_1_s, ans_1_v, ans_1_o, ans_2_s, ans_2_v, ans_2_o, ans_3_s, ans_3_v, ans_3_o

# def get_answers(class_line):
#     chunks = class_line.split(',')
#     prob_strs = chunks[1:]
#     probs = np.array([float(prob_str) for prob_str in prob_strs])
#     ind_1, ind_2, ind_3 = np.argsort(-1.0 * probs)[:3]
#     ans_1_s, ans_1_v, ans_1_o = get_triple[ind_1]
#     ans_2_s, ans_2_v, ans_2_o = get_triple[ind_2]
#     ans_3_s, ans_3_v, ans_3_o = get_triple[ind_3]
#     return ans_1_s, ans_1_v, ans_1_o, ans_2_s, ans_2_v, ans_2_o, ans_3_s, ans_3_v, ans_3_o

def code(ans_val):
    if ans_val == None:
        return -9
    else:
        return ans_val

def minus_one_p(answers):
    for triple in answers:
        for val in triple:
            if val == -1 or val == 'None':
                return True
    return False

def process_test(test_id, metadata, test_df, detect_lines, class_lines, class_file_reader, log):
    pre_red_top_1_hits = pre_red_top_1_misses = post_red_top_1_hits = post_res_top_1_misses = 0
    pre_red_top_3_hits = pre_red_top_3_misses = post_red_top_3_hits = post_res_top_3_misses = 0
    test_tuples = test_df.itertuples()
    # log.write(f' - - - - - Path - - - - -        Key Sys    Key        Sys1        Sys2       Sys3     Top-1  Top-3\n')
    # log.write(f'                                 Nov Nov   S  O  V    S  O  V     S  O  V    S  O  V\n')
    print(f'Test {test_id}')
    log.write(f'**** Test {test_id} ****\n')
    missing_s_count = missing_o_count = 0
    missing_s_corr = []
    missing_s_answers = []
    missing_o_corr = []
    missing_o_answers = []
    non_miss_minus_one_corr = []
    non_miss_minus_ones = []
    for test_tuple, detect_line, class_line in zip(test_tuples, detect_lines[1:], class_lines[1:]):
        ans_nov = float(detect_line.split(',')[1]) > 0.5
        corr_s = test_tuple.subject_id
        corr_o = test_tuple.object_id
        corr_v = test_tuple.verb_id
        corr = (corr_s, corr_v, corr_o)
        ans_1_s, ans_1_v, ans_1_o, ans_2_s, ans_2_v, ans_2_o, ans_3_s, ans_3_v, ans_3_o = (
            class_file_reader.get_answers(class_line)
        )
        missing_s = test_tuple.subject_ymin == -1
        if test_tuple.subject_ymin == -1:
            assert test_tuple.subject_xmin == -1
            assert test_tuple.subject_ymax == -1
            assert test_tuple.subject_xmax == -1
            assert test_tuple.object_ymin != -1
            missing_s_count += 1
        missing_o = test_tuple.object_ymin == -1
        if test_tuple.object_ymin == -1:
            assert test_tuple.object_xmin == -1
            assert test_tuple.object_ymax == -1
            assert test_tuple.object_xmax == -1
            assert test_tuple.subject_ymin != -1
            missing_o_count += 1
        answers = ((ans_1_s, ans_1_v, ans_1_o),
                   (ans_2_s, ans_2_v, ans_2_o),
                   (ans_3_s, ans_3_v, ans_3_o))
        if missing_s:
            missing_s_answers.append(answers)
            missing_s_corr.append(corr)
        if missing_o:
            missing_o_answers.append(answers)
            missing_o_corr.append(corr)
        if not missing_s and not missing_o:
            if minus_one_p(answers):
                non_miss_minus_ones.append(answers)
                non_miss_minus_one_corr.append(corr)
        # top_1 = match(corr_s, corr_o, corr_v, ans_1_s, ans_1_o, ans_1_v)
        # top_3 = (match(corr_s, corr_o, corr_v, ans_1_s, ans_1_o, ans_1_v) or
        #     match(corr_s, corr_o, corr_v, ans_2_s, ans_2_o, ans_2_v) or
        #     match(corr_s, corr_o, corr_v, ans_3_s, ans_3_o, ans_3_v))
        # if top_1:
        #     pre_red_top_1_hits += 1
        # else:
        #     pre_red_top_1_misses += 1
        # if top_3:
        #     pre_red_top_3_hits += 1
        # else:
        #     pre_red_top_3_misses += 1
        # log.write(f'{test_tuple.new_image_path:33} {test_tuple.novel:1}   {ans_nov:1}   '
        #           f'{corr_s:2d} {corr_o:2d} {corr_v:2d}   '
        #           f'{code(ans_1_s):2d} {code(ans_1_o):2d} {code(ans_1_v):2d}    '
        #           f'{code(ans_2_s):2d} {code(ans_2_o):2d} {code(ans_2_v):2d}   '
        #           f'{code(ans_3_s):2d} {code(ans_3_o):2d} {code(ans_3_v):2d} '
        #           f'{top_1:6} {top_3:6} \n')
    # top_1_score = f'{100 * pre_red_top_1_hits / (pre_red_top_1_hits + pre_red_top_1_misses):0.2f}%'
    # top_3_score = f'{100 * pre_red_top_3_hits / (pre_red_top_3_hits + pre_red_top_3_misses):0.2f}%'
    # print(f' {int(test_id):2d}    {top_1_score}    {top_3_score}')
    # log.write(f'{" "*86} {top_1_score}  {top_3_score}\n')
    print(f'{missing_s_count=}')
    log.write(f'{missing_s_count=}\n')
    for corr, ans in zip(missing_s_corr, missing_s_answers):
        print(f'  {corr}  {ans}')
        log.write(f'  {corr}  {ans}\n')
    print(f'{missing_o_count=}')
    log.write(f'{missing_o_count=}\n')
    for corr, ans in zip(missing_o_corr, missing_o_answers):
        print(f'  {corr}  {ans}')
        log.write(f'  {corr}  {ans}\n')
    if non_miss_minus_ones:
        print(f'Answers that included -1 or None when neither S nor O were missing:')
        log.write(f'Answers that included -1 or None when neither S nor O were missing:\n')
        for corr, ans in zip(non_miss_minus_one_corr, non_miss_minus_ones):
            print(f'  {corr}  {ans}')
            log.write(f'  {corr}  {ans}\n')
    log.write(f'\n')

def process_tests(test_dir, results_dir, session_id, class_file_reader, log_file):
    test_ids = open(test_dir/'test_ids.csv', 'r').read().splitlines()
    # print(f'Found {len(test_ids)} tests...')
    # print(f'Test   Top-1     Top-3')
    with open(log_file, 'w') as log:
        log.write(f'Showing the key and our three answer candidates for missing box cases.\n\n')
        for test_id in test_ids:
            metadata = json.load(open(test_dir / f'{test_id}_metadata.json', 'r'))
            test_df = pd.read_csv(test_dir / f'{test_id}_single_df.csv')
            if (results_dir / f'{session_id}.{test_id}_detection.csv').exists():
                detect_lines = open(results_dir / f'{session_id}.{test_id}_detection.csv').read().splitlines()
                class_lines = open(results_dir / f'{session_id}.{test_id}_classification.csv').read().splitlines()
                process_test(test_id, metadata, test_df, detect_lines, class_lines, class_file_reader, log)
            else:
                print(f'No results found for test {test_id}.')
                log.write(f'*** No results found for test {test_id}. ***\n')

def main():
    p = ArgumentParser()
    p.add_argument('test_root')
    p.add_argument('results_root')
    p.add_argument('log_file')
    args = p.parse_args()
    test_dir = Path(args.test_root)/"OND"/'svo_classification'
    results_dir = Path(args.results_root)/'OND'/'svo_classification'
    session_ids = set()
    for file in results_dir.iterdir():
        session_id = file.name.split('.')[0]
        session_ids.add(session_id)
    if len(session_ids) > 1:
        raise Exception('More than one session id in results dir')
    session_id = list(session_ids)[0]
    log_file = Path(args.log_file)
    process_tests(test_dir, results_dir, session_id, ClassFileReader(), log_file)

if __name__ == '__main__':
    main()
