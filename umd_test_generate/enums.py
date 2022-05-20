#########################################################################
# Copyright 2008-2022 by Raytheon BBN Technologies.  All Rights Reserved
#########################################################################

from enum import Enum

# class NoveltyType(Enum):
#     NOVEL_S = 1
#     NOVEL_V_WITH_O = 2
#     NOVEL_O = 3
#     NOVEL_COMB = 4
#     NOVEL_V_MISS_O = 5

class NoveltyType(Enum):
    NOVEL_S = 1
    NOVEL_V = 2
    NOVEL_O = 3
    NO_NOVEL = 4

class InstanceType(Enum):
    BOTH_PRESENT = 1
    S_MISSING = 2
    O_MISSING= 3
