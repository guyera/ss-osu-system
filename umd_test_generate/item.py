"""
Parse a tuple from a val_data CSV to type it
and output it in either api or stubs versions.
"""
#########################################################################
# Copyright 2008-2022 by Raytheon BBN Technologies.  All Rights Reserved
#########################################################################

from enums import InstanceType, NoveltyType

# fields = [
#     'new_image_path',
#     'subject_name',
#     'subject_id',
#     'original_subject_id',
#     'object_name',
#     'object_id',
#     'original_object_id',
#     'verb_name',
#     'verb_id',
#     'original_verb_id',
#     'image_width',
#     'image_height',
#     'subject_ymin',
#     'subject_xmin',
#     'subject_ymax',
#     'subject_xmax',
#     'object_ymin',
#     'object_xmin',
#     'object_ymax',
#     'object_xmax',
#     'master_subject_id',
#     'master_verb_id',
#     'master_object_id',
# ]

class Item:
    def __init__(self, _tuple, val=False):
        # Don't save the tuple, since it has Pandas stuff attached that can't be pickled
        #self._tuple = _tuple
        self.s = _tuple.subject_id
        self.v = _tuple.verb_id
        self.o = _tuple.object_id
        self.subject_ymin = _tuple.subject_ymin
        self.object_ymin = _tuple.object_ymin
        self.new_image_path = _tuple.new_image_path
        self.test_str = self.test_str(_tuple, val)
        self.instance_type = self.find_instance_type()
        self.valid = self.check_valid()
        if self.valid:
            self.novelty_type = self.find_novelty_type()
        else:
            self.novelty_type = None
        # if self.novelty_type == NoveltyType.NOVEL_COMB and not self.is_valid_novel_comb():
        #     log.arite(f'')

    def find_instance_type(self):
        if self.subject_ymin == -1:
            return InstanceType.S_MISSING
        elif self.object_ymin == -1:
            return InstanceType.O_MISSING
        else:
            return InstanceType.BOTH_PRESENT

    def find_novelty_type(self):
        assert self.instance_type is not None
        novel_s = self.s == 0
        novel_v = self.v == 0
        novel_o = self.o == 0
        ## making Tom's change to remove Type 2 from Type 1
        if novel_s:
            # assert not novel_v, 'Cases with two novelties should have been ruled invalid.'
            return NoveltyType.NOVEL_S
        elif novel_o:
            # assert not novel_v, 'Cases with two novelties should have been ruled invalid.'
            return NoveltyType.NOVEL_O
        elif novel_v:
            # assert not novel_s, 'Cases with two novelties should have been ruled invalid.'
            if self.instance_type == InstanceType.S_MISSING:
                return NoveltyType.NO_NOVEL
            else:
                return NoveltyType.NOVEL_V
        else:
            return NoveltyType.NO_NOVEL

    # def is_valid_novel_comb(self, data):
    #     if (self.instance_type == InstanceType.BOTH_PRESENT and
    #         self.s != 0 and
    #         self.v != 0 and
    #         self.o != 0 and
    #         (self.s, self.v, self.o ) not in data.

    def is_non_comb_novel(self):
        if self.s == 0:
            return True
        if self.o == 0:
            return True
        if self.s != -1 and self.v == 0:
            return True
        return False

    def check_valid(self):
        if self.instance_type == InstanceType.S_MISSING and self.v != 0:
            return False
        elif self.s == 0 and self.o == 0:
            return False
        return True
        # # Cases with >1 novelty are invalid
        # elif self.s == 0 and self.v == 0:
        #     return False
        # elif self.s == 0 and self.o == 0:
        #     return False
        # elif self.v == 0 and self.o == 0:
        #     return False
        # return True

    def test_str(self, _tuple, val):
        return ','.join([str(val) for val in [
            _tuple.new_image_path,
            _tuple.subject_name,
            _tuple.subject_id,
            _tuple.master_subject_id if val else "0",
            _tuple.object_name,
            _tuple.object_id,
            _tuple.master_object_id if val else "0",
            _tuple.verb_name,
            _tuple.verb_id,
            _tuple.master_verb_id if val else "0",
            _tuple.image_width,
            _tuple.image_height,
            _tuple.subject_ymin,
            _tuple.subject_xmin,
            _tuple.subject_ymax,
            _tuple.subject_xmax,
            _tuple.object_ymin,
            _tuple.object_xmin,
            _tuple.object_ymax,
            _tuple.object_xmax
        ]])

    def known_triple(self, known_triples):
        triple = (self.s, self.v, self.o)
        return triple in known_triples

    def known_duple(self, known_duples):
        duple = (self.s, self.v)
        return duple in known_duples

    def debug_print(self, log):
        log.write(f'{self.new_image_path:32} {self.instance_type.name:13} '
                  f'({self.s}, {self.v}, {self.o}) \n')

