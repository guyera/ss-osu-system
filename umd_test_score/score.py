"""
Takes a tests directory and a results directory.
Writes log files for each test and a summary scores file to an output logs directory.
"""
#########################################################################
# Copyright 2021-2022 by Raytheon BBN Technologies.  All Rights Reserved
#########################################################################


from pathlib import Path
from argparse import ArgumentParser
import json
import pandas as pd
from class_file_reader import ClassFileReader
from stats import Stats, Instance
import shutil
import numpy as np

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

def code(ans_val):
    """
    Supply a dummy numeric value for printing any Nones to the log file.
    But I don't think we actually get any Nones.
    """
    if ans_val is None:
        return -9
    else:
        return ans_val

def percent_string(num, denom):
    return f'{100 * num / denom:6.2f}%'

def score_test(test_id, metadata, test_df, detect_lines, class_lines, class_file_reader, log, summary, stats):
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
    log.write(f' - - - - - Path - - - - -        Key Sys    Key        Sys1        Sys2       Sys3     Top-1  Top-3\n')
    log.write(f'                                 Nov Nov   S  O  V    S  O  V     S  O  V    S  O  V\n')
    for pos, (test_tuple, detect_line, class_line) in enumerate(zip(test_tuples, detect_lines[1:], class_lines[1:])):
        if test_tuple.novel:
            red_button = True
            if red_button_pos == -1:
                red_button_pos = pos
        ans_nov = float(detect_line.split(',')[1]) > 0.5
        if sys_declare_pos == -1 and ans_nov:
            sys_declare_pos = pos        
        corr_s = test_tuple.subject_id
        corr_o = test_tuple.object_id
        corr_v = test_tuple.verb_id
        ans_1_s, ans_1_v, ans_1_o, ans_2_s, ans_2_v, ans_2_o, ans_3_s, ans_3_v, ans_3_o = (
            class_file_reader.get_answers(class_line)
        )
        stats.add(Instance(corr_s, corr_o, corr_v,
                           ans_1_s, ans_1_o, ans_1_v,
                           ans_2_s, ans_2_o, ans_2_v,
                           ans_3_s, ans_3_o, ans_3_v,
                           test_tuple.new_image_path))
        top_1 = match(corr_s, corr_o, corr_v, ans_1_s, ans_1_o, ans_1_v)
        top_3 = (match(corr_s, corr_o, corr_v, ans_1_s, ans_1_o, ans_1_v) or
                 match(corr_s, corr_o, corr_v, ans_2_s, ans_2_o, ans_2_v) or
                 match(corr_s, corr_o, corr_v, ans_3_s, ans_3_o, ans_3_v))
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
        if test_tuple.novel:
            total_novel += 1
            if top_1:
                novel_top_1_hits += 1
            if top_3:
                novel_top_3_hits += 1
        log.write(f'{test_tuple.new_image_path:33} {test_tuple.novel:1}   {ans_nov:1}   '
                  f'{corr_s:2f} {corr_o:2f} {corr_v:2f}   '
                  f'{code(ans_1_s):2f} {code(ans_1_o):2f} {code(ans_1_v):2f}    '
                  f'{code(ans_2_s):2f} {code(ans_2_o):2f} {code(ans_2_v):2f}   '
                  f'{code(ans_3_s):2f} {code(ans_3_o):2f} {code(ans_3_v):2f} '
                  f'{top_1:6} {top_3:6} \n')
    total_top_1_hits = pre_red_top_1_hits + post_red_top_1_hits
    total_top_3_hits = pre_red_top_3_hits + post_red_top_3_hits

    pre_top_1_score = percent_string(pre_red_top_1_hits, total_pre_red)
    pre_top_3_score = percent_string(pre_red_top_3_hits, total_pre_red)
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
    print(f' {test_id}    {total_top_1_score}    {total_top_3_score}')
    log.write(f'{" "*86} {total_top_1_score}  {total_top_3_score}\n')
    if sys_declare_pos == -1:
        detect = 'Miss'
        delay = ' NA'
    elif sys_declare_pos < red_button_pos:
        detect = ' FA '
        delay = ' NA'
    else:
        detect = 'HIT '
        delay = f'{sys_declare_pos - red_button_pos:3}'
    summary.write(f'{test_id}: '
                  f'{red_button_pos:3} {sys_declare_pos:3} {detect} {delay}    ' 
                  f'{total_top_1_score:7} {pre_top_1_score:7} {post_top_1_score:7} {novel_top_1_score:7}     '
                  f'{total_top_3_score:7} {pre_top_3_score:7} {post_top_3_score:7} {novel_top_3_score:7}\n')

def score_tests(test_dir, sys_output_dir, session_id, class_file_reader, log_dir,
                save_symlinks, dataset_root):
    
    stats = Stats()
    # test_ids = open(test_dir/'test_ids.csv', 'r').read().splitlines()
    import pathlib 
    test_ids = []
    for p in pathlib.Path(test_dir).glob('*'):
        if p.suffix != '.csv':
            continue
        test_name = p.name.split('_')[0]
        test_ids.append(test_name)
    
    # print(f'Found {len(test_ids)} tests...')
    print(f'   Test           Top-1     Top-3')
    with open(log_dir / f'summary.log', 'w') as summary:
        summary.write(f'              Red Decl Res Delay     -------- TOP 1 --------             -------- TOP 3 ---------   \n')
        summary.write(f'                                  Total    Pre      Post    Novel     Total    Pre     Post    Novel\n')
        for test_id in test_ids:
            if 'OND' in test_id and '100.000' not in test_id:

                metadata = json.load(open(test_dir / f'{test_id}_metadata.json', 'r'))
                test_df = pd.read_csv(test_dir / f'{test_id}_single_df.csv')
                test_id = test_id[4:]

                detect_lines = []
                class_lines = []
                for round_ in range(10):
                    if (sys_output_dir / f'{session_id}.{test_id}_{round_}_detection.csv').exists():
                        detect_lines.append(open(sys_output_dir / f'{session_id}.{test_id}_{round_}_detection.csv').read().splitlines())
                        class_lines.append(open(sys_output_dir / f'{session_id}.{test_id}_{round_}_classification.csv').read().splitlines())
                        
                    else:
                        print(f'No results found for Test {session_id}.{test_id}_{round_}.')
                detect_lines = np.concatenate(detect_lines)

                class_lines = np.concatenate(class_lines)
                
                # if (sys_output_dir / f'{session_id}.{test_id}_detection.csv').exists():
                #     detect_lines = open(sys_output_dir / f'{session_id}.{test_id}_detection.csv').read().splitlines()
                #     class_lines = open(sys_output_dir / f'{session_id}.{test_id}_classification.csv').read().splitlines()
                    
                # else:
                #     print(f'No results found for Test {session_id}.{test_id}_.')
                

                with open(log_dir / f'{test_id}.log', 'w') as log:
                            score_test(test_id, metadata, test_df, detect_lines, class_lines, class_file_reader,
                                    log, summary, stats)
            
            # for round_ in range(19):
            #     if (sys_output_dir / f'{session_id}.{test_id}_{round_}_detection.csv').exists():
            #         detect_lines = open(sys_output_dir / f'{session_id}.{test_id}_{round_}_detection.csv').read().splitlines()
            #         class_lines = open(sys_output_dir / f'{session_id}.{test_id}_{round_}_classification.csv').read().splitlines()
            #         with open(log_dir / f'{test_id}.log', 'w') as log:
            #             score_test(test_id, metadata, test_df, detect_lines, class_lines, class_file_reader,
            #                     log, summary, stats)
            #     else:
            #         print(f'No results found for Test {session_id}.{test_id}_{round_}.')
    stats.print_confusion_matrices_1(log_dir / "confusion.pdf")
    if save_symlinks:
        stats.save_image_paths(log_dir / "images", dataset_root)

def main():
    p = ArgumentParser()
    p.add_argument("test_root", help="test specification directory, "
                   "named 'api_tests' for tests generated by umd_test_generate")
    p.add_argument("sys_output_root", help="system output directory, parent of 'OND'")
    p.add_argument("log_dir", help="where to write scoring output directory")
    p.add_argument("--save_symlinks", action="store_true",
                   help="flag to request saving a classified tree of image symlinks")
    p.add_argument("--dataset_root", help="path to UMD image dataset, "
                   "required for save_symlinks")
    args = p.parse_args()
    if args.save_symlinks and not args.dataset_root:
        raise Exception("dataset_root must be specified if you want to save image symlinks")
    dataset_root = Path(args.dataset_root) if args.dataset_root else None
    test_dir = Path(args.test_root)#/"OND"/'svo_classification'
    sys_output_dir = Path(args.sys_output_root) #/'OND'/'svo_classification'
    session_ids = set()
    for file in sys_output_dir.iterdir():
        session_id = file.name.split('.')[0]
        session_ids.add(session_id)
    
    if len(session_ids) > 1:
        raise Exception('More than one session id in results dir')
    session_id = list(session_ids)[0]
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)
    for file in log_dir.iterdir():
        if file.is_dir():
            shutil.rmtree(file)
        else:
            file.unlink()
    
    score_tests(test_dir, sys_output_dir, session_id, ClassFileReader(), log_dir,
                args.save_symlinks, dataset_root)

if __name__ == '__main__':
    main()