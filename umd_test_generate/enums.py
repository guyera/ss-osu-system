#########################################################################
# Copyright 2008-2022 by Raytheon BBN Technologies.  All Rights Reserved
#########################################################################

from enum import Enum


class NoveltyType(Enum):
    NO_NOVEL = 0  # No novelty
    NOVEL_AGENT = 2  # Novel agent
    NOVEL_ACTIVITY = 3  # Novel activity
    NOVEL_COMB_KNOWN_AGENTS = 4  # Novel combination of known agents
    NOVEL_COMB_KNOWN_ACTIVITIES = 5  # Novel combination of known activities
    NOVEL_ENVIRONMENT = 6  # Novel environment


# Enumeration of the types of environemtn
class EnvironmentType(Enum):
    NO_NOVEL_ENV = 60  # Day
    NOVEL_ENV_DADU = 61  # Dawn and dusk
    NOVEL_ENV_NIGH = 62  # Night
    NOVEL_ENV_DFOG = 63  # Day time and fog corruption
    NOVEL_ENV_DSNO = 64  # Day time and snow corruption

# Enumeration of the subnovelties of novelty type 6 used for testing
class TestSubNoveltyType(Enum):
    NOVEL_ENV_DADU = 61  # Dawn and dusk
    NOVEL_ENV_NIGH = 62  # Night
    NOVEL_ENV_DFOG = 63  # Day time and fog corruption
    NOVEL_ENV_DSNO = 64  # Day time and snow corruption

def get_subnovelty_varname(env_id, novelty_type_id):
    if novelty_type_id != 6:
        # env_id 0 is the only known environment
        return None

    id2name_mapping = {
        0: EnvironmentType.NO_NOVEL_ENV, # Day
        1: EnvironmentType.NOVEL_ENV_DADU, # Dawn and dusk
        2: EnvironmentType.NOVEL_ENV_NIGH,  # Night
        3: EnvironmentType.NOVEL_ENV_DFOG,  # Day time and fog corruption
        4: EnvironmentType.NOVEL_ENV_DSNO,  # Day time and snow corruption
    }

    assert env_id in id2name_mapping.keys()

    return id2name_mapping[env_id]


# Get Mapping from novelty index to novelty name
Id2noveltyType = {}
for nov in NoveltyType:
    Id2noveltyType[nov.value] = nov

# for nov in SubNoveltyType:
for nov in EnvironmentType:
    Id2noveltyType[nov.value] = nov
