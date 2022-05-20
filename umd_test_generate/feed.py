"""
Iterate randomly and repeatedly through a list
"""
#########################################################################
# Copyright 2008-2022 by Raytheon BBN Technologies.  All Rights Reserved
#########################################################################

import random

random.seed(42)

class Feed:
    def __init__(self, items):
        self.items = items
        self.count = len(items)
        self.cur_list = random.sample(self.items, len(self.items))
        self.pos = -1

    def get_next(self):
        self.pos += 1
        if self.pos == self.count:
            self.cur_list = random.sample(self.items, len(self.items))
            self.pos = 0
        item = self.cur_list[self.pos]
        return item