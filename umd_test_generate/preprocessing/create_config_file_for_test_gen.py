"""
Create config (csv) file containing information for test generation
"""
import pandas as pd

test_param_dict = {
    'batch_num': [1],
    'count': [1],  # number of tests of each type to generate
    'no_novel_test_len': [1000],  # length of the initial no-novel tests
    'no_novel_round_size': [10],  # round size for the initial no-novel tests
    'test_len': [1000],  # length of the tests with novelty
    'round_size': [10],  # round size for the tests with novelty
    'red_button': [500],  # index value for the switch to the post-novel distribution
    'alpha': [0.4]  # percentage of post-red-button images that will be novel
}

test_param_df = pd.DataFrame.from_dict(test_param_dict)

# test_param_df.to_csv('output_dir/test_config_file.csv')
test_param_df.to_csv('/nfs/hpc/share/sail_on3/final/osu_train_cal_val/test_trials/test_config_file.csv')