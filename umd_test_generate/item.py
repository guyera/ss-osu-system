"""
Parse a tuple from a val_data CSV to type it
and output it in either api or stubs versions.
"""
#########################################################################
# Copyright 2008-2022 by Raytheon BBN Technologies.  All Rights Reserved
#########################################################################

import ast
import math
import pandas as pd
from enums import NoveltyType, get_subnovelty_varname, Id2noveltyType
import json


# all_subnovelties = [sub.value for sub in SubNoveltyType]
all_subnovelties = [sub for sub in Id2noveltyType.keys()]

class Item:
    def __init__(self, _tuple, val=False):
        # Don't save the tuple, since it has Pandas stuff attached that can't be pickled
        
        self.width = _tuple.width
        self.height = _tuple.height
        self.agent1_name = _tuple.agent1_name
        self.agent1_id = _tuple.agent1_id
        self.agent1_count = _tuple.agent1_count
        self.agent2_name = _tuple.agent2_name
        self.agent2_id = _tuple.agent2_id
        self.agent2_count = _tuple.agent2_count
        self.agent3_name = _tuple.agent3_name
        self.agent3_id = _tuple.agent3_id
        self.agent3_count = _tuple.agent3_count
        self.activities = ast.literal_eval(_tuple.activities)  # converts the string representation of a list to a list
        self.activities_id = ast.literal_eval(_tuple.activities_id)
        self.environment_id = _tuple.environment_id
        self.filename = _tuple.filename
        self.image_path = _tuple.image_path
        self.test_str = self.test_str(_tuple, val)
        self.novelty_type_id = _tuple.novelty_type
        self.subnovelty = get_subnovelty_varname(env_id=self.environment_id, novelty_type_id=self.novelty_type_id)
        self.novelty_type = self.find_novelty_type()
        self.valid = self.check_valid()
        if not self.valid:
            print(f'\n>>> This is invalid: {_tuple}')
        # if self.valid:
        #     self.novelty_type = self.find_novelty_type()
        # else:
        #     self.novelty_type = None

    def find_novelty_type(self):
        if self.subnovelty is None:
            # *************** This corresponds to novelty type 2 to 5 ************************
            return Id2noveltyType[self.novelty_type_id]
        # *************** This corresponds to novelty type 6 ************************
        # return Id2noveltyType[self.subnovelty_id]
        return self.subnovelty

    def check_valid(self):
        # Check if no novelty criterion is violated
        if self.novelty_type_id == 0:
            # this is known example (no novelty):
            if (len(self.activities) > 1) | (self.environment_id > 0) | (not math.isnan(self.agent2_id)) | (not  math.isnan(self.agent3_id)):
                return False
            return True
        elif self.novelty_type_id == 2:
            # this is an example with novel agent but single known activity
            if (len(self.activities) > 1) | (math.isnan(self.agent1_id)) | (self.environment_id > 0):
                return False
            return True
        elif self.novelty_type_id == 3:
            # this is an example with novel activity but single agent
            if (self.environment_id > 0) | (math.isnan(self.agent1_id)) | (not math.isnan(self.agent2_id)) | (not math.isnan(self.agent3_id)):
                return False
            return True
        elif self.novelty_type_id == 4:
            # this is an example with a combination of at least 2 known agents all engaged in the same known activity
            if (len(self.activities) > 1) | (self.environment_id > 0) | (math.isnan(self.agent1_id)) | (math.isnan(self.agent2_id)):
                return False
            return True
        elif self.novelty_type_id == 5:
            # this is an example with agents from the same species but engaged in at least 2 known activities
            if (len(self.activities) < 2) | (self.environment_id > 0) | (math.isnan(self.agent1_id)) | (not math.isnan(self.agent2_id)) | (not math.isnan(self.agent3_id)):
                return False
            return True
        else:
            # this an example with a single known activity and a single known agent but in a novel environment
            assert self.novelty_type_id == 6
            if (len(self.activities) > 1) | (self.environment_id <= 0) | (math.isnan(self.agent1_id)) | (not math.isnan(self.agent2_id)) | (not math.isnan(self.agent3_id)):
                return False
        return True

    def test_str(self, _tuple, val):

        # If the input validation data does not have master_id columns
        # print('type of _tuple:', type(_tuple), '\n', _tuple, '\n')

        return ','.join([str(value) for value in [
            _tuple.image_path,
            _tuple.filename,
            _tuple.width,
            _tuple.height,
            _tuple.agent1_name,
            _tuple.agent1_id if pd.isnull(_tuple.agent1_id) else int(_tuple.agent1_id),
            _tuple.agent1_count if pd.isnull(_tuple.agent1_count) else int(_tuple.agent1_count),
            _tuple.agent2_name,
            _tuple.agent2_id if pd.isnull(_tuple.agent2_id) else int(_tuple.agent2_id),
            _tuple.agent2_count if pd.isnull(_tuple.agent2_count) else int(_tuple.agent2_count),
            _tuple.agent3_name,
            _tuple.agent3_id if pd.isnull(_tuple.agent3_id) else int(_tuple.agent3_id),
            _tuple.agent3_count if pd.isnull(_tuple.agent3_count) else int(_tuple.agent3_count),
            json.dumps(_tuple.activities),
            json.dumps(_tuple.activities_id),
            _tuple.environment_id,
            _tuple.novelty_type if pd.isnull(_tuple.novelty_type) else int(_tuple.novelty_type),
            _tuple.master_id if pd.isnull(_tuple.master_id) else int(_tuple.master_id),
        ]]) if val else None

    
    def debug_print(self, log):
        def int_else_nan(x):
            if math.isnan(x):
                return x
            return int(x)
        # log.write(f'{self.image_path:32} {self.agent1_name:13} {self.agent2_name:13} {self.agent3_name:13} {self.activities} {self.environment_id} \n')
        log.write(f'{self.filename:30} {self.agent1_name:10}:{int_else_nan(self.agent1_count)} {self.agent2_name:10}:{int_else_nan(self.agent2_count)} {self.agent3_name:10}:{int_else_nan(self.agent3_count)} {self.activities} {self.environment_id} \n')
    

