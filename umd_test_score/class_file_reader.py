"""
Class to parse classification.csv lines into top-3 triples
"""
#########################################################################
# Copyright 2021-2022 by Raytheon BBN Technologies.  All Rights Reserved
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
