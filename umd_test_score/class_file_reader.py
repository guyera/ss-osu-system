"""
Class to parse classification.csv lines into top-3 triples
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

import itertools
import numpy as np

class ClassFileReader():
    def __init__(self):
        self.max_species = 31
        self.max_activities = 5
        

    def get_answers(self, class_line, num_species=None, num_activities=None):
        chunks = class_line.split(',')
        if num_species is None:
            num_species = self.max_species
        if num_activities is None:
            num_activities = self.max_activities
        assert len(chunks) == 2 * num_species + num_activities + 1

        species_counts = chunks[1:num_species+1]
        species_presence = chunks[num_species+1:2*num_species+1]
        activity_presence = chunks[2*num_species+1:]
        return [float(x) for x in species_counts], [float(x) for x in species_presence], [float(x) for x in activity_presence]
