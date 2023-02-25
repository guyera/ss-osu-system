import unittest

import torch

from tupleprediction import TuplePredictor

class TestPrediction(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda:0'
        self.epsilon = 1e-11
        self.high_epsilon = 1e-6

    def t(self, i):
        return torch.tensor(i, dtype=torch.float, device=self.device)

    def _generic_sub_tests(self, tuple_predictor, probs, p_type, sub_dict, atol):
        n = len(probs)
        species_count, species_presence, activity_count, activity_presence =\
            tuple_predictor.predict([probs], [probs], [p_type])[0]

        s_dict = sub_dict['species']
        sc_dict = s_dict['count']

        for k in range(len(species_count)):
            if sc_dict is not None:
                self.assertTrue(species_count[k].isclose(self.t(sc_dict[k] * n), atol=atol))
            else:
                self.assertTrue(species_count.sum().isclose(self.t(n), atol=atol))
                self.assertFalse(species_count[k].isnan().any())

        sp_dict = s_dict['presence']
        for k in range(len(species_presence)):
            if sp_dict is not None:
                self.assertTrue(species_presence[k].isclose(self.t(sp_dict[k]), atol=atol))
            else:
                self.assertFalse(species_presence[k].isnan().any())
                self.assertFalse(species_presence[k].isinf().any())
                self.assertTrue(
                    torch.logical_and(
                        species_presence >= 0,
                        species_presence <= 1
                    ).all()
                )

        a_dict = sub_dict['activity']
        ac_dict = a_dict['count']
        for k in range(len(activity_count)):
            if ac_dict is not None:
                self.assertTrue(activity_count[k].isclose(self.t(ac_dict[k] * n), atol=atol))
            else:
                self.assertTrue(activity_count.sum().isclose(self.t(n), atol=atol))
                self.assertFalse(activity_count[k].isnan().any())

        ap_dict = a_dict['presence']
        for k in range(len(activity_presence)):
            if ap_dict is not None:
                self.assertTrue(activity_presence[k].isclose(self.t(ap_dict[k]), atol=atol))
            else:
                self.assertFalse(activity_presence[k].isnan().any())
                self.assertFalse(activity_presence[k].isinf().any())
                self.assertTrue(
                    torch.logical_and(
                        activity_presence >= 0,
                        activity_presence <= 1
                    ).all()
                )

    def _generic_tests(self, answer_dict, p_type, epsilon, atol=1e-8):
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
                    1.0 - 3 * epsilon,
                    epsilon,
                    epsilon,
                    epsilon
                ],
                [
                    1.0 - 3 * epsilon,
                    epsilon,
                    epsilon,
                    epsilon
                ]
            ],
            device=self.device
        )
        sub_dict = answer_dict['known-confident-1']
        self._generic_sub_tests(tuple_predictor, probs, p_type, sub_dict, atol)

        # Case: Two boxes, only known classes, total confidence about two
        # different classes
        probs = torch.tensor(
            [
                [
                    1.0 - 3 * epsilon,
                    epsilon,
                    epsilon,
                    epsilon
                ],
                [
                    epsilon,
                    1.0 - 3 * epsilon,
                    epsilon,
                    epsilon
                ]
            ],
            device=self.device
        )
        sub_dict = answer_dict['known-confident-2']
        self._generic_sub_tests(tuple_predictor, probs, p_type, sub_dict, atol)

        # Case: Five boxes, only known classes, unconfident
        probs = torch.tensor(
            [
                [
                    0.5 - epsilon,
                    0.5 - epsilon,
                    epsilon,
                    epsilon
                ],
                [
                    0.5 - epsilon,
                    0.5 - epsilon,
                    epsilon,
                    epsilon
                ],
                [
                    0.5 - epsilon,
                    0.5 - epsilon,
                    epsilon,
                    epsilon
                ],
                [
                    0.5 - epsilon,
                    0.5 - epsilon,
                    epsilon,
                    epsilon
                ],
                [
                    0.5 - epsilon,
                    0.5 - epsilon,
                    epsilon,
                    epsilon
                ]
            ],
            device=self.device
        )
        sub_dict = answer_dict['known-unconfident']
        self._generic_sub_tests(tuple_predictor, probs, p_type, sub_dict, atol)

        # Case: Two boxes, only novel classes, confident in one class
        probs = torch.tensor(
            [
                [
                    epsilon,
                    epsilon,
                    1.0 - 3 * epsilon,
                    epsilon
                ],
                [
                    epsilon,
                    epsilon,
                    1.0 - 3 * epsilon,
                    epsilon
                ]
            ],
            device=self.device
        )
        sub_dict = answer_dict['novel-confident-1']
        self._generic_sub_tests(tuple_predictor, probs, p_type, sub_dict, atol)

        # Case: Two boxes, only novel classes, total confidence about two
        # different classes
        probs = torch.tensor(
            [
                [
                    epsilon,
                    epsilon,
                    1.0 - 3 * epsilon,
                    epsilon
                ],
                [
                    epsilon,
                    epsilon,
                    epsilon,
                    1.0 - 3 * epsilon
                ]
            ],
            device=self.device
        )
        sub_dict = answer_dict['novel-confident-2']
        self._generic_sub_tests(tuple_predictor, probs, p_type, sub_dict, atol)

        # Case: Five boxes, only novel classes, unconfident
        probs = torch.tensor(
            [
                [
                    epsilon,
                    epsilon,
                    0.5 - epsilon,
                    0.5 - epsilon
                ],
                [
                    epsilon,
                    epsilon,
                    0.5 - epsilon,
                    0.5 - epsilon
                ],
                [
                    epsilon,
                    epsilon,
                    0.5 - epsilon,
                    0.5 - epsilon
                ],
                [
                    epsilon,
                    epsilon,
                    0.5 - epsilon,
                    0.5 - epsilon
                ],
                [
                    epsilon,
                    epsilon,
                    0.5 - epsilon,
                    0.5 - epsilon
                ]
            ],
            device=self.device
        )
        sub_dict = answer_dict['novel-unconfident']
        self._generic_sub_tests(tuple_predictor, probs, p_type, sub_dict, atol)

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
        self._generic_sub_tests(tuple_predictor, probs, p_type, sub_dict, atol)

        # Case: One box, uniform prediction (for catching one-box edge cases,
        # like with combination novelties)
        sub_dict = answer_dict['one-box']
        if sub_dict is not None:
            probs = torch.tensor(
                [
                    [
                        0.25,
                        0.25,
                        0.25,
                        0.25
                    ]
                ],
                device=self.device
            )
            self._generic_sub_tests(tuple_predictor, probs, p_type, sub_dict, atol)

    def test_t06(self):
        epsilon = self.epsilon
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
            },
            'one-box': {
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
        self._generic_tests(answer_dict, p_type, epsilon)

        p_type = torch.tensor(
            [
                1.0 - 5 * epsilon,
                epsilon,
                epsilon,
                epsilon,
                epsilon,
                epsilon
            ],
            device=self.device
        )
        self._generic_tests(answer_dict, p_type, epsilon)

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
        self._generic_tests(answer_dict, p_type, epsilon)

        p_type = torch.tensor(
            [
                epsilon,
                epsilon,
                epsilon,
                epsilon,
                epsilon,
                1.0 - 5 * epsilon
            ],
            device=self.device
        )
        self._generic_tests(answer_dict, p_type, epsilon)

    def test_t4(self):
        epsilon = self.epsilon
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
            },
            'one-box': {
                'species': {
                    'count': [0.0, 0.0, 0.0, 0.0],
                    'presence': [0.0, 0.0, 0.0, 0.0]
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
        self._generic_tests(answer_dict, p_type, epsilon)

        p_type = torch.tensor(
            [
                epsilon,
                epsilon,
                epsilon,
                1.0 - 5 * epsilon,
                epsilon,
                epsilon
            ],
            device=self.device
        )
        self._generic_tests(answer_dict, p_type, epsilon)

    def test_t2(self):
        epsilon = self.epsilon
        answer_dict = {
            'known-confident-1': {
                'species': {
                    'count': None,
                    'presence': None
                },
                'activity': {
                    'count': [1.0, 0.0, 0.0, 0.0],
                    'presence': [1.0, 0.0, 0.0, 0.0]
                }
            },
            'known-confident-2': {
                'species': {
                    'count': None,
                    'presence': None
                },
                'activity': {
                    'count': [0.5, 0.5, 0.0, 0.0],
                    'presence': [0.5, 0.5, 0.0, 0.0]
                }
            },
            'known-unconfident': {
                'species': {
                    'count': None,
                    'presence': None
                },
                'activity': {
                    'count': [0.5, 0.5, 0.0, 0.0],
                    'presence': [0.5, 0.5, 0.0, 0.0]
                }
            },
            'novel-confident-1': {
                'species': {
                    'count': [0.0, 0.0, 1.0, 0.0],
                    'presence': [0.0, 0.0, 1.0, 0.0]
                },
                'activity': {
                    'count': [0.5, 0.5, 0.0, 0.0],
                    'presence': [0.5, 0.5, 0.0, 0.0]
                }
            },
            'novel-confident-2': {
                'species': {
                    'count': [0.0, 0.0, 0.5, 0.5],
                    'presence': [0.0, 0.0, 1.0, 1.0]
                },
                'activity': {
                    'count': [0.5, 0.5, 0.0, 0.0],
                    'presence': [0.5, 0.5, 0.0, 0.0]
                }
            },
            'novel-unconfident': {
                'species': {
                    'count': [0.0, 0.0, 0.5, 0.5],
                    # There are 5 boxes, and they each have a 50% chance to
                    # belong to one novel class and 50% chance to belong
                    # to the other (uniform). They're conditionally
                    # independent. So a novel class will be absent if and
                    # only if all five boxes belong to the other novel class,
                    # with probability 0.5^5. Then P(novel class k present)
                    # = 1 - 0.5^5 = 0.96875
                    'presence': [0.0, 0.0, 0.96875, 0.96875]
                },
                'activity': {
                    'count': [0.5, 0.5, 0.0, 0.0],
                    'presence': [0.5, 0.5, 0.0, 0.0]
                }
            },
            'uniform': {
                'species': {
                    'count': None,
                    'presence': None
                },
                'activity': {
                    'count': [0.5, 0.5, 0.0, 0.0],
                    'presence': [0.5, 0.5, 0.0, 0.0]
                }
            },
            'one-box': {
                'species': {
                    'count': [0.0, 0.0, 0.5, 0.5],
                    'presence': [0.0, 0.0, 0.5, 0.5]
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
                1.0,
                0.0,
                0.0,
                0.0,
                0.0
            ],
            device=self.device
        )
        self._generic_tests(answer_dict, p_type, epsilon, atol=1e-2)

        p_type = torch.tensor(
            [
                epsilon,
                1.0 - 5 * epsilon,
                epsilon,
                epsilon,
                epsilon,
                epsilon
            ],
            device=self.device
        )
        self._generic_tests(answer_dict, p_type, epsilon, atol=1e-2)

    def test_tuniform(self):
        epsilon = self.epsilon
        answer_dict = {
            'known-confident-1': {
                'species': {
                    'count': None,
                    'presence': None
                },
                'activity': {
                    'count': None,
                    'presence': None
                }
            },
            'known-confident-2': {
                'species': {
                    'count': None,
                    'presence': None
                },
                'activity': {
                    'count': None,
                    'presence': None
                }
            },
            'known-unconfident': {
                'species': {
                    'count': None,
                    'presence': None
                },
                'activity': {
                    'count': None,
                    'presence': None
                }
            },
            'novel-confident-1': {
                'species': {
                    'count': None,
                    'presence': None
                },
                'activity': {
                    'count': None,
                    'presence': None
                }
            },
            'novel-confident-2': {
                'species': {
                    'count': None,
                    'presence': None
                },
                'activity': {
                    'count': None,
                    'presence': None
                }
            },
            'novel-unconfident': {
                'species': {
                    'count': None,
                    'presence': None
                },
                'activity': {
                    'count': None,
                    'presence': None
                }
            },
            'uniform': {
                'species': {
                    'count': None,
                    'presence': None
                },
                'activity': {
                    'count': None,
                    'presence': None
                }
            },
            'one-box': None
        }

        p_type = torch.tensor(
            [
                1.0 / 6.0,
                1.0 / 6.0,
                1.0 / 6.0,
                1.0 / 6.0,
                1.0 / 6.0,
                1.0 / 6.0
            ],
            device=self.device
        )
        self._generic_tests(answer_dict, p_type, epsilon)


if __name__ == '__main__':
    unittest.main()
