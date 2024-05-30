"""
Iterate randomly and repeatedly through a list
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
