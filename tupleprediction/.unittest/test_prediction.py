import unittest

import torch

from tupleprediction import TuplePredictor

class TestPrediction(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda:0'
        self.epsilon = 1e-11

    def t(self, i):
        return torch.tensor(i, dtype=torch.float, device=self.device)

    def _generic_sub_tests(self, tuple_predictor, probs, p_type, sub_dict):
        n = len(probs)
        species_count, species_presence, activity_count, activity_presence =\
            tuple_predictor.predict([probs], [probs], [p_type])[0]

        s_dict = sub_dict['species']
        sc_dict = s_dict['count']

        self.assertTrue(species_count.sum().isclose(self.t(n)))
        for k in range(len(species_count)):
            self.assertTrue(species_count[k].isclose(self.t(sc_dict[k] * n)))

        sp_dict = s_dict['presence']
        for k in range(len(species_presence)):
            self.assertTrue(species_presence[k].isclose(self.t(sp_dict[k])))

        a_dict = sub_dict['activity']
        ac_dict = a_dict['count']
        self.assertTrue(activity_count.sum().isclose(self.t(n)))
        for k in range(len(activity_count)):
            self.assertTrue(activity_count[k].isclose(self.t(ac_dict[k] * n)))

        ap_dict = a_dict['presence']
        for k in range(len(activity_presence)):
            self.assertTrue(activity_presence[k].isclose(self.t(ap_dict[k])))

    def _generic_tests(self, answer_dict, p_type):
        n_species_cls = 4
        n_activity_cls = 4
        n_known_species_cls = 2
        n_known_activity_cls = 2
        tuple_predictor = TuplePredictor(
            n_species_cls,
            n_activity_cls,
            n_known_species_cls,
            n_known_activity_cls
        )

        # Case: Two boxes, only known classes, total confidence about one
        # class
        probs = torch.tensor(
            [
                [
                    1.0 - 3 * self.epsilon,
                    self.epsilon,
                    self.epsilon,
                    self.epsilon
                ],
                [
                    1.0 - 3 * self.epsilon,
                    self.epsilon,
                    self.epsilon,
                    self.epsilon
                ]
            ],
            device=self.device
        )
        sub_dict = answer_dict['known-confident-1']
        self._generic_sub_tests(tuple_predictor, probs, p_type, sub_dict)

        # Case: Two boxes, only known classes, total confidence about two
        # different classes
        probs = torch.tensor(
            [
                [
                    1.0 - 3 * self.epsilon,
                    self.epsilon,
                    self.epsilon,
                    self.epsilon
                ],
                [
                    self.epsilon,
                    1.0 - 3 * self.epsilon,
                    self.epsilon,
                    self.epsilon
                ]
            ],
            device=self.device
        )
        sub_dict = answer_dict['known-confident-2']
        self._generic_sub_tests(tuple_predictor, probs, p_type, sub_dict)

        # Case: Five boxes, only known classes, unconfident
        probs = torch.tensor(
            [
                [
                    0.5 - self.epsilon,
                    0.5 - self.epsilon,
                    self.epsilon,
                    self.epsilon
                ],
                [
                    0.5 - self.epsilon,
                    0.5 - self.epsilon,
                    self.epsilon,
                    self.epsilon
                ],
                [
                    0.5 - self.epsilon,
                    0.5 - self.epsilon,
                    self.epsilon,
                    self.epsilon
                ],
                [
                    0.5 - self.epsilon,
                    0.5 - self.epsilon,
                    self.epsilon,
                    self.epsilon
                ],
                [
                    0.5 - self.epsilon,
                    0.5 - self.epsilon,
                    self.epsilon,
                    self.epsilon
                ]
            ],
            device=self.device
        )
        sub_dict = answer_dict['known-unconfident']
        self._generic_sub_tests(tuple_predictor, probs, p_type, sub_dict)

        # Case: Two boxes, only novel classes, confident in one class
        probs = torch.tensor(
            [
                [
                    self.epsilon,
                    self.epsilon,
                    1.0 - 3 * self.epsilon,
                    self.epsilon
                ],
                [
                    self.epsilon,
                    self.epsilon,
                    1.0 - 3 * self.epsilon,
                    self.epsilon
                ]
            ],
            device=self.device
        )
        sub_dict = answer_dict['novel-confident-1']
        self._generic_sub_tests(tuple_predictor, probs, p_type, sub_dict)

        # Case: Two boxes, only novel classes, total confidence about two
        # different classes
        probs = torch.tensor(
            [
                [
                    self.epsilon,
                    self.epsilon,
                    1.0 - 3 * self.epsilon,
                    self.epsilon
                ],
                [
                    self.epsilon,
                    self.epsilon,
                    self.epsilon,
                    1.0 - 3 * self.epsilon
                ]
            ],
            device=self.device
        )
        sub_dict = answer_dict['novel-confident-2']
        self._generic_sub_tests(tuple_predictor, probs, p_type, sub_dict)

        # Case: Five boxes, only novel classes, unconfident
        probs = torch.tensor(
            [
                [
                    self.epsilon,
                    self.epsilon,
                    0.5 - self.epsilon,
                    0.5 - self.epsilon
                ],
                [
                    self.epsilon,
                    self.epsilon,
                    0.5 - self.epsilon,
                    0.5 - self.epsilon
                ],
                [
                    self.epsilon,
                    self.epsilon,
                    0.5 - self.epsilon,
                    0.5 - self.epsilon
                ],
                [
                    self.epsilon,
                    self.epsilon,
                    0.5 - self.epsilon,
                    0.5 - self.epsilon
                ],
                [
                    self.epsilon,
                    self.epsilon,
                    0.5 - self.epsilon,
                    0.5 - self.epsilon
                ]
            ],
            device=self.device
        )
        sub_dict = answer_dict['novel-unconfident']
        self._generic_sub_tests(tuple_predictor, probs, p_type, sub_dict)

        # Case: Two boxes, uniform prediction
        probs = torch.tensor(
            [
                [
                    0.25,
                    0.25,
                    0.25,
                    0.25
                ],
                [
                    0.25,
                    0.25,
                    0.25,
                    0.25
                ]
            ],
            device=self.device
        )
        sub_dict = answer_dict['uniform']
        self._generic_sub_tests(tuple_predictor, probs, p_type, sub_dict)

    def test_t06(self):
        answer_dict = {
            'known-confident-1': {
                'species': {
                    'count': [1.0, 0.0, 0.0, 0.0],
                    'presence': [1.0, 0.0, 0.0, 0.0]
                },
                'activity': {
                    'count': [1.0, 0.0, 0.0, 0.0],
                    'presence': [1.0, 0.0, 0.0, 0.0]
                }
            },
            'known-confident-2': {
                'species': {
                    'count': [0.5, 0.5, 0.0, 0.0],
                    'presence': [0.5, 0.5, 0.0, 0.0]
                },
                'activity': {
                    'count': [0.5, 0.5, 0.0, 0.0],
                    'presence': [0.5, 0.5, 0.0, 0.0]
                }
            },
            'known-unconfident': {
                'species': {
                    'count': [0.5, 0.5, 0.0, 0.0],
                    'presence': [0.5, 0.5, 0.0, 0.0]
                },
                'activity': {
                    'count': [0.5, 0.5, 0.0, 0.0],
                    'presence': [0.5, 0.5, 0.0, 0.0]
                }
            },
            'novel-confident-1': {
                'species': {
                    'count': [0.5, 0.5, 0.0, 0.0],
                    'presence': [0.5, 0.5, 0.0, 0.0]
                },
                'activity': {
                    'count': [0.5, 0.5, 0.0, 0.0],
                    'presence': [0.5, 0.5, 0.0, 0.0]
                }
            },
            'novel-confident-2': {
                'species': {
                    'count': [0.5, 0.5, 0.0, 0.0],
                    'presence': [0.5, 0.5, 0.0, 0.0]
                },
                'activity': {
                    'count': [0.5, 0.5, 0.0, 0.0],
                    'presence': [0.5, 0.5, 0.0, 0.0]
                }
            },
            'novel-unconfident': {
                'species': {
                    'count': [0.5, 0.5, 0.0, 0.0],
                    'presence': [0.5, 0.5, 0.0, 0.0]
                },
                'activity': {
                    'count': [0.5, 0.5, 0.0, 0.0],
                    'presence': [0.5, 0.5, 0.0, 0.0]
                }
            },
            'uniform': {
                'species': {
                    'count': [0.5, 0.5, 0.0, 0.0],
                    'presence': [0.5, 0.5, 0.0, 0.0]
                },
                'activity': {
                    'count': [0.5, 0.5, 0.0, 0.0],
                    'presence': [0.5, 0.5, 0.0, 0.0]
                }
            }
        }

        p_type = torch.tensor(
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0
            ],
            device=self.device
        )
        self._generic_tests(answer_dict, p_type)

        p_type = torch.tensor(
            [
                1.0 - 5 * self.epsilon,
                self.epsilon,
                self.epsilon,
                self.epsilon,
                self.epsilon,
                self.epsilon
            ],
            device=self.device
        )
        self._generic_tests(answer_dict, p_type)

        p_type = torch.tensor(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0
            ],
            device=self.device
        )
        self._generic_tests(answer_dict, p_type)

        p_type = torch.tensor(
            [
                self.epsilon,
                self.epsilon,
                self.epsilon,
                self.epsilon,
                self.epsilon,
                1.0 - 5 * self.epsilon
            ],
            device=self.device
        )
        self._generic_tests(answer_dict, p_type)

    def test_t4(self):
        answer_dict = {
            'known-confident-1': {
                'species': {
                    'count': [0.5, 0.5, 0.0, 0.0],
                    'presence': [1.0, 1.0, 0.0, 0.0]
                },
                'activity': {
                    'count': [1.0, 0.0, 0.0, 0.0],
                    'presence': [1.0, 0.0, 0.0, 0.0]
                }
            },
            'known-confident-2': {
                'species': {
                    'count': [0.5, 0.5, 0.0, 0.0],
                    'presence': [1.0, 1.0, 0.0, 0.0]
                },
                'activity': {
                    'count': [0.5, 0.5, 0.0, 0.0],
                    'presence': [0.5, 0.5, 0.0, 0.0]
                }
            },
            'known-unconfident': {
                'species': {
                    'count': [0.5, 0.5, 0.0, 0.0],
                    'presence': [1.0, 1.0, 0.0, 0.0]
                },
                'activity': {
                    'count': [0.5, 0.5, 0.0, 0.0],
                    'presence': [0.5, 0.5, 0.0, 0.0]
                }
            },
            'novel-confident-1': {
                'species': {
                    'count': [0.5, 0.5, 0.0, 0.0],
                    'presence': [1.0, 1.0, 0.0, 0.0]
                },
                'activity': {
                    'count': [0.5, 0.5, 0.0, 0.0],
                    'presence': [0.5, 0.5, 0.0, 0.0]
                }
            },
            'novel-confident-2': {
                'species': {
                    'count': [0.5, 0.5, 0.0, 0.0],
                    'presence': [1.0, 1.0, 0.0, 0.0]
                },
                'activity': {
                    'count': [0.5, 0.5, 0.0, 0.0],
                    'presence': [0.5, 0.5, 0.0, 0.0]
                }
            },
            'novel-unconfident': {
                'species': {
                    'count': [0.5, 0.5, 0.0, 0.0],
                    'presence': [1.0, 1.0, 0.0, 0.0]
                },
                'activity': {
                    'count': [0.5, 0.5, 0.0, 0.0],
                    'presence': [0.5, 0.5, 0.0, 0.0]
                }
            },
            'uniform': {
                'species': {
                    'count': [0.5, 0.5, 0.0, 0.0],
                    'presence': [1.0, 1.0, 0.0, 0.0]
                },
                'activity': {
                    'count': [0.5, 0.5, 0.0, 0.0],
                    'presence': [0.5, 0.5, 0.0, 0.0]
                }
            }
        }

        p_type = torch.tensor(
            [
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0
            ],
            device=self.device
        )
        self._generic_tests(answer_dict, p_type)

        p_type = torch.tensor(
            [
                self.epsilon,
                self.epsilon,
                self.epsilon,
                1.0 - 5 * self.epsilon,
                self.epsilon,
                self.epsilon
            ],
            device=self.device
        )
        self._generic_tests(answer_dict, p_type)

if __name__ == '__main__':
    unittest.main()
