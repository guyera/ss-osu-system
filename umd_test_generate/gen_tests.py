#########################################################################
# Copyright 2008-2022 by Raytheon BBN Technologies.  All Rights Reserved
#########################################################################

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

random.seed(42)

test_header = 'new_image_path,subject_name,subject_id,master_subject_id,' \
              'object_name,object_id,master_object_id,verb_name,verb_id,master_verb_id,'\
              'image_width,image_height,subject_ymin,subject_xmin,subject_ymax,subject_xmax,' \
              'object_ymin,object_xmin,object_ymax,object_xmax,novel'

def gen_no_novel_api_test(test_name,
                          known_feed,
                          test_dir, test_len, round_size):
    metadata_file = test_dir / f'{test_name}_metadata.json'
    test_file = test_dir / f'{test_name}_single_df.csv'
    with open(test_file, 'w') as handle:
        handle.write(f'{test_header}\n')
        for i in range(test_len):
            item = known_feed.get_next()
            item_str = item.test_str
            handle.write(f'{item_str},0\n')
    feedback_budget = round(round_size * 0.5)
    metadata = {
        'domain' : 'svo_classification',
        'protocol' : 'OND',
        'known_classes' : 8,
        'round_size' : round_size,
        'threshold' : 0.1,
        'red_light' : 'none',
        'feedback_max_ids' : feedback_budget,
        'max_datection_feedback_ids' : feedback_budget
    }
    with open(metadata_file, 'w') as handle:
        handle.write(json.dumps(metadata))

def gen_api_test(test_name,
                 known_feed, novel_feed,
                 test_dir, test_len, red_button, alpha, round_size):
    metadata_file = test_dir / f'{test_name}_metadata.json'
    test_file = test_dir / f'{test_name}_single_df.csv'
    first_novel = None
    with open(test_file, 'w') as handle:
        handle.write(f'{test_header}\n')
        for i in range(test_len):
            if i < red_button:
                item = known_feed.get_next()
                item_str = item.test_str
                handle.write(f'{item_str},0\n')
            elif random.random() < alpha:
                item = novel_feed.get_next()
                if first_novel is None:
                    first_novel = item.new_image_path
                item_str = item.test_str
                handle.write(f'{item_str},1\n')
            else:
                item = known_feed.get_next()
                item_str = item.test_str
                handle.write(f'{item_str},0\n')
    feedback_budget = round(round_size * 0.5)
    metadata = {
        'domain' : 'svo_classification',
        'protocol' : 'OND',
        'known_classes' : 8,
        'round_size' : round_size,
        'threshold' : 0.1,
        'red_light' : first_novel,
        'feedback_max_ids' : feedback_budget,
        'max_datection_feedback_ids' : feedback_budget
    }
    with open(metadata_file, 'w') as handle:
        handle.write(json.dumps(metadata))

def add_topk_files(test_dir):
    source_dir = Path("./topk_files")
    for f in ["subject_topk.csv", "verb_topk.csv", "object_topk.csv"]:
        shutil.copy(source_dir/f, test_dir/f)

def gen_api_tests(test_source_data, out_dir, test_configs):
    if not out_dir.exists():
        out_dir.mkdir()
    api_dir = out_dir / 'api_tests'
    if api_dir.exists():
        shutil.rmtree(api_dir)
    api_dir.mkdir()
    ond_dir = api_dir / 'OND'
    ond_dir.mkdir()
    test_dir = ond_dir / 'svo_classification'
    test_dir.mkdir()
    test_names = []
    known_feed = Feed(test_source_data.known)
    novel_feeds = {}
    for novelty_type in NoveltyType:
        novel_feeds[novelty_type] = Feed(test_source_data.novelty_bins[novelty_type])
    for test_config in test_configs:
        for count in range(test_config.count):
            # maybe generate an initial no_novelty test
            if test_config.no_novel_test_len:
                test_name = f'OND.{test_config.batch_num}00.{count:03d}'
                test_names.append(test_name)
                gen_no_novel_api_test(test_name,
                                      known_feed,
                                      test_dir, test_config.no_novel_test_len,
                                      test_config.no_novel_round_size)
            for novelty_type in NoveltyType:
                if novelty_type == NoveltyType.NO_NOVEL:
                    continue
                # For now, just one test per type
                test_name = f'OND.{test_config.batch_num}0{novelty_type.value}.{count:03d}'
                test_names.append(test_name)
                gen_api_test(test_name,
                             known_feed, novel_feeds[novelty_type],
                             test_dir, test_config.test_len, test_config.red_button,
                             test_config.alpha, test_config.round_size)
    with open(test_dir / 'test_ids.csv', 'w') as handle:
        for test_name in test_names:
            handle.write(f'{test_name}\n')
    add_topk_files(test_dir)


def main():
    p = ArgumentParser()
    p.add_argument('--test_source_data', required=True)
    p.add_argument('--test_config_file', required=True)
    # p.add_argument('--test_len', type=int, default=400)
    # p.add_argument('--red_button', type=int, default=70)
    # p.add_argument('--alpha', type=float, default='0.25')
    # p.add_argument('--round_size', type=int, default=20)
    p.add_argument('--out_dir', required=True)
    args = p.parse_args()
    pickle_file = Path(args.test_source_data)
    config_file = Path(args.test_config_file)
    out_dir = Path(args.out_dir)
    with open(pickle_file, 'rb') as handle:
        test_source_data = pickle.load(handle)
    with open(config_file, 'r') as handle:
        df = pd.read_csv(handle)
    test_configs = list(df.itertuples())
    gen_api_tests(test_source_data, out_dir, test_configs)


if __name__ == '__main__':
    main()