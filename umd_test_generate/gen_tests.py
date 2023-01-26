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
from enums import NoveltyType, EnvironmentType, TestSubNoveltyType, Id2noveltyType, get_subnovelty_varname 
from data import Data
from feed import Feed

random.seed(42)


test_header = 'image_path,filename,width,height,master_id,novel'

# keys_to_take = ('image_path', 'capture_id', 'height', 'width', 'master_id')
# env_subnovelties = ('dawn-dusk', 'night', 'day-fog', 'day-snow')

def gen_no_novel_api_test(test_name, known_feed, test_dir, test_len, round_size):
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
        'domain' : 'image_classification',
        'protocol' : 'OND',
        'known_classes' : 11,
        "max_novel_classes": 16,
        'round_size' : round_size,
        'threshold' : 0.1,
        'red_light' : 'none',
        'feedback_max_ids' : feedback_budget,
        'max_datection_feedback_ids' : feedback_budget
    }
    with open(metadata_file, 'w') as handle:
        handle.write(json.dumps(metadata))


def gen_api_test(test_name, known_feed, novel_feed, test_dir, test_len, red_button, alpha, round_size):
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
                    first_novel = item.image_path
                item_str = item.test_str
                handle.write(f'{item_str},1\n')
            else:
                item = known_feed.get_next()
                item_str = item.test_str
                handle.write(f'{item_str},0\n')
    feedback_budget = round(round_size * 0.5)
    metadata = {
        'domain' : 'image_classification',
        'protocol' : 'OND',
        'known_classes' : 11,
        "max_novel_classes": 16,
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
    test_dir = ond_dir / 'image_classification'
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

def gen_api_single_novelty_tests(test_source_data, out_dir, test_configs, novelty_type=None, subnovelty=None):
    if not out_dir.exists():
        out_dir.mkdir()
    api_dir = out_dir / 'api_tests'
    if api_dir.exists():
        shutil.rmtree(api_dir)
    api_dir.mkdir()
    ond_dir = api_dir / 'OND'
    ond_dir.mkdir()
    test_dir = ond_dir / 'image_classification'
    test_dir.mkdir()
    test_names = []

    # check if the prenovelty validation set is available otherwise select prenovelty images for the trial in the training set
    trial_known_size = (test_configs[0].test_len - test_configs[0].red_button)*(1 - test_configs[0].alpha)
    if Feed(test_source_data.novelty_bins[NoveltyType.NO_NOVEL]).count >= trial_known_size:
        print('>>> Test being generated with known examples not seen during training!')
        known_feed = Feed(test_source_data.novelty_bins[NoveltyType.NO_NOVEL])
    else:
        known_feed = Feed(test_source_data.known)
    
    novel_feeds = {}
    if novelty_type is not None:  # generate test for a single novelty
        if novelty_type == NoveltyType.NOVEL_ENVIRONMENT and subnovelty is not None:
            novel_feeds = Feed(test_source_data.novelty_bins[subnovelty])
        elif novelty_type == NoveltyType.NOVEL_ENVIRONMENT:
            assert subnov_type is None
            # the feed from which the novel examples are drawn is the union of all subnovelty of type 7 (environment)
            print('>|>|>>> Test being generate with all subnovelties in novelty type 7!')
            for type6_subnov in EnvironmentType:
                novel_feeds[type6_subnov] = Feed(test_source_data.novelty_bins[type6_subnov])
        else:
            novel_feeds = Feed(test_source_data.novelty_bins[novelty_type])
    else:
        # generate test for all novelty and subnovelty
        for nov_type in NoveltyType:
            if nov_type.value == 6:  # generate based on subnovelty if novelty type is 6
                continue
            novel_feeds[nov_type] = Feed(test_source_data.novelty_bins[nov_type])

        val_test_subnov = [x.value for x in TestSubNoveltyType]
        for subnov_type in EnvironmentType:
            if subnov_type.value not in val_test_subnov:
                # subnovelty to in the set of subnovelties for testing
                continue
            if len(test_source_data.novelty_bins[subnov_type]) == 0:
                continue
            
            try:
                novel_feeds[subnov_type] = Feed(test_source_data.novelty_bins[subnov_type])
                print('subnovelty:', subnov_type, 'size of feed:', novel_feeds[subnov_type].count)
            except Exception as ex:
                print(ex)
                continue

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
            
            # generate test for a single novelty
            if novelty_type is not None:  
                test_name = f'OND.{test_config.batch_num}0{novelty_type.value}.{count:03d}'
                test_names.append(test_name)
                gen_api_test(
                    test_name, known_feed, novel_feeds, test_dir, test_config.test_len, 
                    test_config.red_button, test_config.alpha, test_config.round_size
                    )
            else: 
                # generate test for all novelty and subnovelty type
                test_subnovelty_values = [subnov.value for subnov in TestSubNoveltyType]
                novelty_types = list(NoveltyType) + [subnov for subnov in EnvironmentType if subnov.value in test_subnovelty_values]
                for novelty_type in novelty_types:
                    if novelty_type == NoveltyType.NO_NOVEL or novelty_type not in novel_feeds.keys():
                        print('>>> Skipping novelty or subnovelty:', novelty_type)
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
    # add_topk_files(test_dir)


def main():
    p = ArgumentParser()
    p.add_argument('--test_source_data', required=True)
    p.add_argument('--test_config_file', required=True)
    p.add_argument('--novelty_type', default=None)
    # p.add_argument('--subnovelty', nargs='?', default='environmnet', const='environmnet')
    p.add_argument('--subnovelty', default=None)
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
    # gen_api_tests(test_source_data, out_dir, test_configs)
    nov_type = args.novelty_type
    
    # sub_novelty = None
    sub_novelty = args.subnovelty
    if nov_type is not None:
        assert nov_type in ['type0', 'type2', 'type3', 'type4', 'type5', 'type6']
        
        if nov_type == 'type0':
            novel_type = NoveltyType.NO_NOVEL
            # raise TypeError(f'Novelty type {nov_type[-1]} is not a novelty!')
        elif nov_type == 'type2':
            # images containing at least one novel agent
            novel_type = NoveltyType.NOVEL_AGENT
        elif nov_type == 'type3':
            # images containing at least one novel activity
            novel_type = NoveltyType.NOVEL_ACTIVITY
        elif nov_type == 'type4':
            # images containing a combination of at least 2 known agents
            novel_type = NoveltyType.NOVEL_COMB_KNOWN_AGENTS
        elif nov_type == 'type5':
            # images containing a combination of at least 2 knonw activities
            novel_type = NoveltyType.NOVEL_COMB_KNOWN_ACTIVITIES
        else:
            # images containing a single known agents in a novel environment
            novel_type = NoveltyType.NOVEL_ENVIRONMENT
            
            # If subnovelty is provided, then all the novel entries will be of that subnovelty
            if sub_novelty is not None:
                assert isinstance(sub_novelty, int)
                sub_novelty = get_subnovelty_varname(sub_novelty)
    else:
        novel_type = None
    
    gen_api_single_novelty_tests(test_source_data, out_dir, test_configs, novelty_type=novel_type, subnovelty=sub_novelty)


if __name__ == '__main__':
    """
    python gen_tests.py --test_config_file '/nfs/hpc/share/sail_on3/final/osu_train_cal_val/test_trials/test_config_file.csv' --test_source_data '/nfs/hpc/share/sail_on3/final/osu_train_cal_val/test_trials/osu_test_sandbox/test_source_data_valid.pkl' --out_dir '/nfs/hpc/share/sail_on3/final/osu_train_cal_val/test_trials/'
    """
    main()