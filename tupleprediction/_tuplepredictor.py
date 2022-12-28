import torch

'''
Notation:
    S: int
        The number of species classes
    A: int
        The number of activity classes
'''
class TuplePredictor:
    '''
    Parameters:
        n_species_cls: int
            The number of known species classes
        n_activity_cls: int
            The number of known activity classes
    '''
    def __init__(self, n_species_cls, n_activity_cls):
        self._n_species_cls = n_species_cls
        self._n_activity_cls = n_activity_cls

    def _species(self, species_probs, p_type):
        # NOTE: Every probability here is implicitly conditioned on the box
        # image. And we're able to multiply probabilities across boxes to
        # compute intersection probabilities because we're making a conditional
        # independence assumption: box class probabilities are conditionally
        # independent given the box images themselves because a box should
        # generally be a sufficient statistic for the label.
        n = species_probs.shape[0]
        known_species_probs = species_probs[:self._n_species_cls]

        # If the novelty type is 0, 3, 5, or 6, then we know the species vector
        # vector only contains known species, and the labels are the same
        # across boxes. Compute a single count vector for all three of these
        # types.
        
        # For each species s, compute
        # P(all S_j=s | all S_j are the same and known)
        same_species_probs = torch.prod(known_species_probs, dim=0)
        t0356_species_probs = same_species_probs / same_species_probs.sum()
        
        # Compute the species count vector given that all the species are the
        # same and known (this can be done by just multiplying
        # same_species_cond_prob by the number of boxes)
        t0356_species_count = t0356_species_probs * n

        # Type 0/3/5/6 species presence / absence. In this particular case,
        # it's just equal to the species probs
        t0356_species_presence = t0356_species_probs

        # If the novelty type is 4, then we know the species vector only
        # contains known species, and they cannot all be the same. We'll
        # compute this in two stages. First, we'll ignore the novel species
        # and renormalize to condition out type 2. Then, we'll subtract off
        # the probability vector for all the species being the same---a box
        # belongs to species j if all the boxes belong to species j, OR it
        # belongs to species j AND one or more other boxes belong to another
        # species. We want to compute the latter, but it's difficult to compute
        # directly due to a combinatoric problem. So instead, we just subtract
        # off the former value from the sum of the two disjoint event
        # probabilities.
        
        # Stage 1: Compute label predictions conditioned on the absence of any
        # novel labels
        known_species_cond_probs = \
            known_species_probs / known_species_probs.sum(dim=1, keepdim=True)

        # known_species_cond_probs represents P(S_j=s | c), where c is the
        # condition: there are no novel species.
        # Stage 2:
        # P(S_j=s, >=2 unique species | c) = P(S_j=s, NOT ALL S_j=s | c)
        # = P(S_j=s | c) - P(all S_j=s | c)
        same_species_cond_probs = torch.prod(known_species_cond_probs, dim=0)
        diff_species_cond_probs = \
            known_species_cond_probs - same_species_cond_probs

        # Normalize to convert from P(S_j=s, >=2 unique species | c) to
        # P(S_j=s | type=4)
        t4_species_probs = diff_species_cond_probs / \
            diff_species_cond_probs.sum(dim=1, keepdim=True)

        # Compute species count vector for type 4
        t4_species_count = t4_species_probs.sum(dim=0)

        # Type 4 species presence / absence
        t4_species_presence = 1 - torch.prod(1 - t4_species_probs, dim=0)

        # Given c, a species is absent and it's a type 4 image if and only if
        # none of the boxes match the species and the boxes do not belong to
        # exactly one known species.

        # To compute type 2 species probabilities, we consider the following:
        # A type 2 image's box labels cannot belong to two or more known species
        # classes, and there must be at least one novel label. These two
        # criteria form a necessary and sufficient definition for type 2.
        # We'll condition on them one at a time, starting with the former:
        # For known s:
        # P(S_j=s, <2 known species) = P(S_j=s, all other S_j in {s, novel})
        novel_species_probs = species_probs[self._n_species_cls:]
        combined_novel_species_probs = novel_species_probs.sum(dim=1)
        known_or_novel_species_probs = \
            known_species_probs + combined_novel_species_probs[:, None]
        unique_known_species_probs = \
            torch.prod(known_or_novel_species_probs, dim=0) / \
                known_or_novel_species_probs
        known_unique_species_probs = \
            known_species_probs * unique_known_species_probs

        # For novel s:
        # P(S_j=s, <2 known species)
        # = P(S_j=s) * \sum_{known k} P(All other S_j in {k, novel})
        novel_unique_species_probs = \
            novel_species_probs * \
                known_or_novel_species_probs.sum(dim=1, keepdim=True)

        # Concatenate the results for known and novel species
        unique_species_probs = torch.cat(
            (known_unique_species_probs, novel_unique_species_probs),
            dim=1
        )

        # Normalize to convert from joint to conditional probability
        unique_species_cond_probs = \
            unique_species_probs / unique_species_probs.sum(dim=1, keepdim=True)

        # Extract known and novel parts of conditional probabilities
        known_unique_species_cond_probs = \
            unique_species_cond_probs[:self._n_species_cls]
        novel_unique_species_cond_probs = \
            unique_species_cond_probs[self._n_species_cls:]

        # unique_species_cond_probs represents P(S_j=s | c), where c is the
        # condition: there are 0 or 1 unique species among the boxes.
        # The next step is to additionally condition on the event that there
        # is at least one novel label. That's easy.
        # For known s:
        # P(S_j=s, >=1 novel label | c)
        # = P(S_j=s | c) - P(S_j=s, no novel labels | c)
        # = P(S_j=s | c) - P(all S_j=s | c)
        exactly_one_species_cond_probs = \
            torch.prod(known_unique_species_cond_probs, dim=0)
        known_t2_joint_probs = \
            known_unique_species_cond_probs - exactly_one_species_cond_probs

        # And for novel s, it's trivial:
        # P(S_j=s, >=1 novel label | c) = P(S_j=s | c)
        # = novel_unique_species_cond_probs
        
        # Concatenate known and novel parts to get P(S_j=s | type=2)
        t2_species_probs = torch.cat(
            (known_t2_joint_probs, novel_unique_species_cond_probs),
            dim=1
        )

        # Compute species count vector for type 2
        t2_species_count = t2_species_probs.sum(dim=0)

        # Type 2 species presence / absence
        t2_species_presence = 1 - torch.prod(1 - t2_species_probs, dim=0)

        # Compute expectation of species counts and presence / absence 
        # probabilities by mixing the per-type predictions weighted by the
        # corresponding P(Type) probabilities
        t0356 = p_type[0] + p_type[2] + p_type[4] + p_type[5]
        t2 = p_type[1]
        t4 = p_type[3]
        species_count = t0356_species_count * t0356 \
            + t2_species_count * t2 \
            + t4_species_count * t4

        return species_count, species_presence

    def _activity(self, activity_probs, p_type):
        # NOTE: Every probability here is implicitly conditioned on the box
        # image. And we're able to multiply probabilities across boxes to
        # compute intersection probabilities because we're making a conditional
        # independence assumption: box class probabilities are conditionally
        # independent given the box images themselves because a box should
        # generally be a sufficient statistic for the label.
        n = activity_probs.shape[0]
        known_activity_probs = activity_probs[:self._n_activity_cls]

        # If the novelty type is 0, 2, 4, or 6, then we know the activity vector
        # vector only contains known activities, and the labels are the same
        # across boxes. Compute a single count vector for all three of these
        # types.
        
        # For each activity s, compute
        # P(all S_j=s | all S_j are the same and known)
        same_activity_probs = torch.prod(known_activity_probs, dim=0)
        t0246_activity_probs = same_activity_probs / same_activity_probs.sum()
        
        # Compute the activity count vector given that all the activity are the
        # same and known (this can be done by just multiplying
        # same_activity_cond_prob by the number of boxes)
        t0246_activity_count = t0246_activity_probs * n

        # Type 0/2/4/6 activity presence / absence
        t0246_activity_presence = 1 - torch.prod(1 - t0246_activity_probs, dim=0)

        # If the novelty type is 5, then we know the activity vector only
        # contains known activity, and they cannot all be the same. We'll
        # compute this in two stages. First, we'll ignore the novel activity
        # and renormalize to condition out type 2. Then, we'll subtract off
        # the probability vector for all the activity being the same---a box
        # belongs to activity j if all the boxes belong to activity j, OR it
        # belongs to activity j AND one or more other boxes belong to another
        # activity. We want to compute the latter, but it's difficult to compute
        # directly due to a combinatoric problem. So instead, we just subtract
        # off the former value from the sum of the two disjoint event
        # probabilities.
        
        # Stage 1: Compute label predictions conditioned on the absence of any
        # novel labels
        known_activity_cond_probs = \
            known_activity_probs / known_activity_probs.sum(dim=1, keepdim=True)

        # known_activity_cond_probs represents P(S_j=s | c), where c is the
        # condition: there are no novel activity.
        # Stage 2:
        # P(S_j=s, >=2 unique activity | c) = P(S_j=s, NOT ALL S_j=s | c)
        # = P(S_j=s | c) - P(all S_j=s | c)
        same_activity_cond_probs = torch.prod(known_activity_cond_probs, dim=0)
        diff_activity_cond_probs = \
            known_activity_cond_probs - same_activity_cond_probs

        # Normalize to convert from P(S_j=s, >=2 unique activity | c) to
        # P(S_j=s | type=5)
        t5_activity_probs = diff_activity_cond_probs / \
            diff_activity_cond_probs.sum(dim=1, keepdim=True)

        # Compute activity count vector for type 5
        t5_activity_count = t5_activity_probs.sum(dim=0)

        # Type 5 activity presence / absence
        t5_activity_presence = 1 - torch.prod(1 - t5_activity_probs, dim=0)

        # To compute type 3 activity probabilities, we consider the following:
        # A type 3 image's box labels cannot belong to two or more known activity
        # classes, and there must be at least one novel label. These two
        # criteria form a necessary and sufficient definition for type 3.
        # We'll condition on them one at a time, starting with the former:
        # For known s:
        # P(S_j=s, <3 known activity) = P(S_j=s, all other S_j in {s, novel})
        novel_activity_probs = activity_probs[self._n_activity_cls:]
        combined_novel_activity_probs = novel_activity_probs.sum(dim=1)
        known_or_novel_activity_probs = \
            known_activity_probs + combined_novel_activity_probs[:, None]
        unique_known_activity_probs = \
            torch.prod(known_or_novel_activity_probs, dim=0) / \
                known_or_novel_activity_probs
        known_unique_activity_probs = \
            known_activity_probs * unique_known_activity_probs

        # For novel s:
        # P(S_j=s, <3 known activity)
        # = P(S_j=s) * \sum_{known k} P(All other S_j in {k, novel})
        novel_unique_activity_probs = \
            novel_activity_probs * \
                known_or_novel_activity_probs.sum(dim=1, keepdim=True)

        # Concatenate the results for known and novel activity
        unique_activity_probs = torch.cat(
            (known_unique_activity_probs, novel_unique_activity_probs),
            dim=1
        )

        # Normalize to convert from joint to conditional probability
        unique_activity_cond_probs = \
            unique_activity_probs / unique_activity_probs.sum(dim=1, keepdim=True)

        # Extract known and novel parts of conditional probabilities
        known_unique_activity_cond_probs = \
            unique_activity_cond_probs[:self._n_activity_cls]
        novel_unique_activity_cond_probs = \
            unique_activity_cond_probs[self._n_activity_cls:]

        # unique_activity_cond_probs represents P(S_j=s | c), where c is the
        # condition: there are 0 or 1 unique activity among the boxes.
        # The next step is to additionally condition on the event that there
        # is at least one novel label. That's easy.
        # For known s:
        # P(S_j=s, >=1 novel label | c)
        # = P(S_j=s | c) - P(S_j=s, no novel labels | c)
        # = P(S_j=s | c) - P(all S_j=s | c)
        exactly_one_activity_cond_probs = \
            torch.prod(known_unique_activity_cond_probs, dim=0)
        known_t3_joint_probs = \
            known_unique_activity_cond_probs - exactly_one_activity_cond_probs

        # And for novel s, it's trivial:
        # P(S_j=s, >=1 novel label | c) = P(S_j=s | c)
        # = novel_unique_activity_cond_probs
        
        # Concatenate known and novel parts to get P(S_j=s | type=3)
        t3_activity_probs = torch.cat(
            (known_t3_joint_probs, novel_unique_activity_cond_probs),
            dim=1
        )

        # Compute activity count vector for type 3
        t3_activity_count = t3_activity_probs.sum(dim=0)

        # Type 3 activity presence / absence
        t3_activity_presence = 1 - torch.prod(1 - t3_activity_probs, dim=0)

        # Compute expectation of activity counts and presence / absence 
        # probabilities by mixing the per-type predictions weighted by the
        # corresponding P(Type) probabilities
        t0246 = p_type[0] + p_type[2] + p_type[4] + p_type[5]
        t2 = p_type[1]
        t4 = p_type[3]
        activity_count = t0246_activity_count * t0246 \
            + t2_activity_count * t2 \
            + t4_activity_count * t4

        activity_presence = t0246_activity_presence * t0246 \
            + t2_activity_presence * t2 \
            + t4_activity_presence * t4

        return activity_count, activity_presence

    '''
    Parameters:
        species_probs: List of tensors. Each tensor is of shape [N, S], where
                N is the number of boxes for the associated image.
            Calibrated probabilities output by the ConfidenceCalibrator
        activity_probs: List of tensors. Each tensor is of shape [N, A], where
                N is the number of boxes for the associated image.
            Calibrated probabilities output by the ConfidenceCalibrator
        p_type: Tensor of shape [N, 5]
            Per-image P(Type_i) predictions. In order, the six probabilities
            correspond to type 0 (no novelty), type 2 (novel species), type
            3 (novel activity), type 4 (>=2 known species), type 5 (>=2 known
            activities), and type 6 (novel environment)

    Returns:
    '''
    def predict(self, species_probs, activity_probs, p_type):
        predictions = []
        for idx in range(len(species_probs)):
            cur_species_probs = species_probs[idx]
            cur_activity_probs = activity_probs[idx]
            cur_p_type = p_type[idx]

            species_count, species_presence = self._species(
                cur_species_probs,
                cur_p_type
            )
            activity_count, activity_presence = self._activity(
                cur_activity_probs,
                cur_p_type
            )
            predictions.append((
                species_count,
                species_presence,
                activity_count,
                activity_presence
            ))

        return predictions
