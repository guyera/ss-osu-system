import unittest

import torch

import noveltydetection.utils

class TestConfidenceCalibrationMethods(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda:0'
        subject_nominal_scores = torch.randn(1000) * 2 + 0
        subject_novel_scores = torch.randn(100) * 1 + 30
        self.subject_score_ctx = noveltydetection.utils.ScoreContext(noveltydetection.utils.ScoreContext.Source.UNSUPERVISED, subject_nominal_scores, subject_novel_scores).to(self.device)
        object_nominal_scores = torch.randn(1000) * 2 + 10
        object_novel_scores = torch.randn(100) * 1 + 40
        self.object_score_ctx = noveltydetection.utils.ScoreContext(noveltydetection.utils.ScoreContext.Source.UNSUPERVISED, object_nominal_scores, object_novel_scores).to(self.device)
        verb_nominal_scores = torch.randn(1000) * 2 + 20
        verb_novel_scores = torch.randn(100) * 1 + 30
        self.verb_score_ctx = noveltydetection.utils.ScoreContext(noveltydetection.utils.ScoreContext.Source.UNSUPERVISED, verb_nominal_scores, verb_novel_scores).to(self.device)
        
        subject_scores =                [1.0,  29.0, -1.0, 0.0,  2.0,  1.0,  30.0, 0.0,  1.0,  None, None]
        verb_scores =                   [20.0, 19.5, 30.5, 21.0, 21.0, 20.0, 18.0, 32.0, 20.0, None, None]
        object_scores =                 [9.0,  11.0, 8.0,  37.0, 10.0, None, None, None, None, 10.0, 41.0]
        self.p_known_svo = torch.tensor([0.99, 0.99, 0.99, 0.99, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], device = self.device)
        self.p_known_sv =  torch.tensor([0.99, 0.99, 0.99, 0.99, 0.01, 0.99, 0.99, 0.99, 0.01, 0.00, 0.00], device = self.device)
        self.p_known_so =  torch.tensor([0.99, 0.99, 0.99, 0.99, 0.99, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], device = self.device)
        self.p_known_vo =  torch.tensor([0.99, 0.99, 0.99, 0.99, 0.99, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], device = self.device)
        self.subject_scores = [torch.tensor(x, device = self.device) if x is not None else None for x in subject_scores]
        self.object_scores = [torch.tensor(x, device = self.device) if x is not None else None for x in object_scores]
        self.verb_scores = [torch.tensor(x, device = self.device) if x is not None else None for x in verb_scores]

    def test_all(self):
        p_type, p_n = noveltydetection.utils.compute_probability_novelty(self.subject_scores, self.verb_scores, self.object_scores, self.p_known_svo, self.p_known_sv, self.p_known_so, self.p_known_vo, self.subject_score_ctx, self.verb_score_ctx, self.object_score_ctx)
        
        self.assertTrue(p_n[0] < 0.5)
        self.assertTrue(p_n[1] > 0.5)
        self.assertTrue(p_n[2] > 0.5)
        self.assertTrue(p_n[3] > 0.5)
        self.assertTrue(p_n[4] > 0.5)
        self.assertTrue(p_n[5] < 0.5)
        self.assertTrue(p_n[6] > 0.5)
        self.assertTrue(p_n[7] > 0.5)
        self.assertTrue(p_n[8] > 0.5)
        self.assertTrue(p_n[9] < 0.5)
        self.assertTrue(p_n[10] > 0.5)

        p_type_sums = p_type.sum(dim = 1)
        self.assertTrue(torch.all(p_type_sums == 1.0))
        self.assertFalse(torch.any(p_type < 0.0))
        self.assertFalse(torch.any(p_type > 1.0))

        self.assertTrue(p_type[0][0] > 0.5)
        self.assertTrue(p_type[1][1] > 0.5)
        self.assertTrue(p_type[2][2] > 0.5)
        self.assertTrue(p_type[3][3] > 0.5)
        self.assertTrue(p_type[4][4] > 0.5)
        self.assertTrue(p_type[5][0] > 0.5)
        self.assertTrue(p_type[6][1] > 0.5)
        self.assertTrue(p_type[7][2] > 0.5)
        self.assertTrue(p_type[8][4] > 0.5)
        self.assertTrue(p_type[9][0] > 0.5)
        self.assertTrue(p_type[10][3] > 0.5)

if __name__ == '__main__':
    unittest.main()
