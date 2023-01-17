#########################################################################
# Copyright 2008-2022 by Raytheon BBN Technologies.  All Rights Reserved
#########################################################################

from enums import NoveltyType
from item import Item

class TestSourceData:
    def __init__(self, novelty_bins):
        self.known = novelty_bins[NoveltyType.NO_NOVEL]
        self.novelty_bins = novelty_bins
