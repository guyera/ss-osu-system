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
        
        subject_scores =           [1.0,  29.0, -1.0, 0.0,  2.0,  1.0,  30.0, 0.0,  None, None]
        object_scores =            [9.0,  11.0, 37.0, 8.0,  10.0, None, None, None, 10.0, 41.0]
        verb_scores =              [20.0, 19.5, 21.0, 30.5, 21.0, 20.0, 18.0, 32.0, None, None]
        self.p_n_t4 = torch.tensor([0.01, 0.02, 0.01, 0.01, 0.98, 0.00, 0.0,  0.0,  0.0,  0.0 ], device = self.device)
        self.subject_scores = [torch.tensor(x, device = self.device) if x is not None else None for x in subject_scores]
        self.object_scores = [torch.tensor(x, device = self.device) if x is not None else None for x in object_scores]
        self.verb_scores = [torch.tensor(x, device = self.device) if x is not None else None for x in verb_scores]

    def test_uniform_p_type(self):
        p_type = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2], device = self.device)
        probs = noveltydetection.utils.compute_probability_novelty(self.subject_scores, self.verb_scores, self.object_scores, self.p_n_t4, self.subject_score_ctx, self.verb_score_ctx, self.object_score_ctx, p_type)
        
        self.assertTrue(probs[0] < 0.5)
        self.assertTrue(probs[1] > 0.5)
        self.assertTrue(probs[2] > 0.5)
        self.assertTrue(probs[3] > 0.5)
        self.assertTrue(probs[4] > 0.5)
        self.assertTrue(probs[5] < 0.5)
        self.assertTrue(probs[6] > 0.5)
        self.assertTrue(probs[7] > 0.5)
        self.assertTrue(probs[8] < 0.5)
        self.assertTrue(probs[9] > 0.5)

if __name__ == '__main__':
    unittest.main()
