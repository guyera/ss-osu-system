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
        n_known_species_cls: int
            The number of known species classes
        n_known_activity_cls: int
            The number of known activity classes
    '''
    def __init__(self, n_known_species_cls, n_known_activity_cls):
        self._n_known_species_cls = n_known_species_cls
        self._n_known_activity_cls = n_known_activity_cls

    def _counts_presence(self,
            class_probs,
            p_type,
            n_known_cls,
            unique_types,
            novel_box_type,
            combination_type):
        # If there are no box images, then return a bunch of zeros for
        # both presence and counts
        if len(class_probs) == 0:
            zero_count = torch.zeros(
                class_probs.shape[1],
                device=class_probs.device
            )
            zero_presence = torch.zeros(
                class_probs.shape[1],
                device=class_probs.device
            )
            return zero_count, zero_presence

        # NOTE: Comments draw out math from the perspective of species.
        # Translating to activities, type 2 <-> type 3, and type 4 <-> type 5.
        # In the comments, type 2 means "novel box", type 4 means "combination",
        # and the remaining types mean "unique"
        n_max_novel_cls = class_probs.shape[1] - n_known_cls

        n = class_probs.shape[0]
        known_class_probs = class_probs[:, :n_known_cls]

        # If the novelty type is 0, 3, 5, or 6, then we know the class vector
        # vector only contains known class, and the labels are the same
        # across boxes. Compute a single count vector for all three of these
        # types.
        
        # For each class s, compute
        # P(all S_j=k | all S_j are the same and known)
        same_class_probs = torch.prod(known_class_probs, dim=0)
        unique_types_class_probs = same_class_probs / same_class_probs.sum()

        # Compute the class count vector given that all the class are the
        # same and known (this can be done by just multiplying
        # same_class_cond_prob by the number of boxes)
        unique_types_class_count = unique_types_class_probs * n

        # Append zeros for the novel classes
        unique_types_class_count = torch.cat(
            (
                unique_types_class_count,
                torch.zeros(
                    n_max_novel_cls,
                    device=unique_types_class_count.device
                )
            ),
            dim=0
        )

        # Type 0/3/5/6 class presence / absence. In this particular case,
        # it's just equal to the class probs
        unique_types_class_presence = unique_types_class_probs

        # Append zeros for the novel classes
        unique_types_class_presence = torch.cat(
            (
                unique_types_class_presence,
                torch.zeros(
                    n_max_novel_cls,
                    device=unique_types_class_presence.device
                )
            ),
            dim=0
        )

        # To compute type 2/4 count predictions, here is the general strategy:
        # To compute a count, we want a weighted average of counts across
        # possible valid predictions, with the weights being the conditional
        # probabilities of those predictions and the counts being the counts
        # of the class per-prediction. That is, for some class k, we
        # want to compute the count n_k as
        # \sum_{valid p}(# boxes containing k in p * P(Tuple=p|valid)).
        # Suppose p is a matrix with each row representing a box and each
        # column representing a class. Each row is a one-hot vector.
        # Then n_k = \sum_{valid p}(\sum_j(p_jk) * P(Tuple=p|valid))
        # = \sum_j(\sum_{valid p}(p_jk * P(Tuple=p|valid)))
        # = \sum_j(\sum_{valid p s.t. p_jk = 1}(p_jk * P(Tuple=p|valid)))
        # = \sum_j(\sum_{valid p s.t. p_jk = 1}(P(Tuple=p|valid)))
        # = \sum_j(P(S_j=k | valid)).
        # That is, we start by computing P(S_j=k | valid prediction), i.e.
        # P(S_j=k | Type=t), for each box. Then we just sum over the boxes.

        # If the novelty type is 4, then we know the class vector only
        # contains known class, and they cannot all be the same. We'll
        # compute this in two stages. First, we'll ignore the novel class
        # and renormalize to condition out type 2. Then, we'll subtract off
        # the probability vector for all the class being the same---a box
        # belongs to class j if all the boxes belong to class j, OR it
        # belongs to class j AND one or more other boxes belong to another
        # class. We want to compute the latter, but it's difficult to compute
        # directly due to a combinatoric problem. So instead, we just subtract
        # off the former value from the sum of the two disjoint event
        # probabilities.

        ### Type 4 counts:

        # Stage 1: Compute label predictions conditioned on the absence of any
        # novel labels
        known_class_cond_probs = \
            known_class_probs / known_class_probs.sum(dim=1, keepdim=True)

        # known_class_cond_probs represents P(S_j=k | c), where c is the
        # condition: there are no novel class.

        # Stage 2:
        # P(S_j=k, >=2 unique class | c) = P(S_j=k, NOT ALL S_j=k | c)
        # = P(S_j=k | c) - P(all S_j=k | c)
        same_class_cond_probs = torch.prod(known_class_cond_probs, dim=0)
        diff_class_cond_probs = \
            known_class_cond_probs - same_class_cond_probs

        # Normalize to convert from P(S_j=k, >=2 unique class | c) to
        # P(S_j=k | type=4)
        combination_type_class_probs = diff_class_cond_probs / \
            diff_class_cond_probs.sum(dim=1, keepdim=True)

        # Compute class count vector for type 4
        combination_type_class_count = combination_type_class_probs.sum(dim=0)

        # Append zeros for the novel classes
        combination_type_class_count = torch.cat(
            (
                combination_type_class_count,
                torch.zeros(
                    n_max_novel_cls,
                    device=combination_type_class_count.device
                )
            ),
            dim=0
        )

        ### Type 4 presence:

        # First, compute P(k present, valid t4 | no novel boxes)
        # = P(k present | no novel boxes) - P(all k | no novel boxes)
        # = P(k present | c) - P(all k | c)
        # = [1 - \prod_j(1 - P(S_j=k | c))] - \prod_j(P(S_j=k | c))
        # = [1 - \prod_j(1 - P(S_j=k | c))] - same_class_cond_probs
        p_cond_present = 1 - torch.prod(1 - known_class_cond_probs, dim=0)
        p_cond_present_combination_type = p_cond_present - same_class_cond_probs

        # Next, compute P(valid t4 | no novel boxes)
        # = P(valid t4 | c)
        # = 1 - P(one unique class | c)
        # = 1 - \sum_k(P(all k | c))
        # = 1 - \sum_k(\prod_j(P(S_j=k | c)))
        p_cond_combination_type = 1 - torch.prod(known_class_cond_probs, dim=0).sum(dim=0)

        # Lastly, compute P(k present | valid t4)
        # = P(k present | valid t4, no novel boxes)
        # = P(k present, valid t4 | no novel boxes) / P(t4 | no novel boxes).
        combination_type_class_presence = p_cond_present_combination_type / p_cond_combination_type

        # Append zeros for the novel classes
        combination_type_class_presence = torch.cat(
            (
                combination_type_class_presence,
                torch.zeros(
                    n_max_novel_cls,
                    device=combination_type_class_presence.device
                )
            ),
            dim=0
        )

        ### Type 2 count:

        # A type 2 image's box labels cannot belong to two or more known class
        # classes, and there must be at least one novel label. These two
        # criteria form a necessary and sufficient definition for type 2.
        # We'll condition on them one at a time, starting with the former:
        # For known k:
        # P(S_j=k, <2 known class) = P(S_j=k, all other S_j in {k, novel})
        novel_class_probs = class_probs[:, n_known_cls:]
        combined_novel_class_probs = novel_class_probs.sum(dim=1)
        known_or_novel_class_probs = \
            known_class_probs + combined_novel_class_probs[:, None]
        unique_known_class_probs = \
            torch.prod(known_or_novel_class_probs, dim=0) / \
                known_or_novel_class_probs
        known_unique_class_probs = \
            known_class_probs * unique_known_class_probs

        # For novel s:
        # P(S_j=k, <2 known class)
        # = P(S_j=k, \union_{known k}(remaining S_j in {k, novel}))
        # = P(S_j=k, \union_{known k}(A_k)),
        #       where A_k = "remaining S_j in {k, novel}"
        # = P(S_j=k) * P(\union_{known k}(A_k))
        # = P(S_j=k) * [
        #       \sum_{known k}(P(A_k))
        #       - \sum_{known k1, k2}(P(A_k1, A_k2))
        #       + \sum_{known k1, k2, k3}(P(A_k1, A_k2, A_k3))
        #       - ... + ... .....
        #   ]
        # ... NOTE: This is just the formula for the union of K events
        # ... NOTE: (A_k1 AND A_k2) = N = "remaining S_j all novel"
        # ... NOTE: (N AND A_k) = N, for any k
        # = P(S_j=k) * [
        #       \sum_{known k}(P(A_k))
        #       - \sum_{known k1, k2}(P(N))
        #       + \sum_{known k1, k2, k3}(P(N))
        #       - ... + ... .....
        #   ]
        # = P(S_j=k) * [
        #       \sum_{known k}(P(A_k))
        #       + P(N) * (-(K choose 2) + (K choose 3) - ... (K choose K))
        #   ], where K is the number of known classes
        # = P(S_j=k) * [
        #       \sum_{known k}(P(A_k))
        #       + P(N) * (1 - K)
        #   ].

        # P(A_k) = \prod_{remaining j'}(P(S_j' in {k, novel}))
        # = \prod_j'(P(S_j' in {k, novel})) / P(S_j in {k, novel})
        # = unique_known_class_probs. Already computed.
 
        # P(N) = \prod_{remaining j'}(P(S_j' in novel))
        # = \prod_j'(P(S_j' in novel)) / P(S_j in novel)
        prob_remaining_novel =\
            torch.prod(combined_novel_class_probs, dim=0) /\
                combined_novel_class_probs

        # Now, P(S_j=k, <2 known class) for novel s:
        novel_unique_class_probs = novel_class_probs *\
            (
                unique_known_class_probs.sum(dim=1)\
                    + prob_remaining_novel * (1 - known_class_probs.shape[1])
            )[:, None]

        # Concatenate the results for known and novel class
        unique_class_probs = torch.cat(
            (known_unique_class_probs, novel_unique_class_probs),
            dim=1
        )

        # Normalize to convert from joint to conditional probability,
        # P(S_j=k | <2 known class)
        unique_class_cond_probs = \
            unique_class_probs / unique_class_probs.sum(dim=1, keepdim=True)

        # Extract known and novel parts of conditional probabilities
        known_unique_class_cond_probs = \
            unique_class_cond_probs[:, :n_known_cls]
        novel_unique_class_cond_probs = \
            unique_class_cond_probs[:, n_known_cls:]

        # unique_class_cond_probs represents P(S_j=k | c), where c is the
        # condition: there are 0 or 1 unique class among the boxes.
        # The next step is to additionally condition on the event that there
        # is at least one novel label. That's easy.
        # For known k:
        # P(S_j=k, >=1 novel label | c)
        # = P(S_j=k | c) - P(S_j=k, no novel labels | c)
        # = P(S_j=k | c) - P(all S_j=k | c)
        exactly_one_class_cond_probs = \
            torch.prod(known_unique_class_cond_probs, dim=0)
        known_novel_box_type_joint_probs = \
            known_unique_class_cond_probs - exactly_one_class_cond_probs

        # And for novel s, it's trivial:
        # P(S_j=k, >=1 novel label | c) = P(S_j=k | c)
        # = novel_unique_class_cond_probs
        
        # Concatenate known and novel parts to get P(S_j=k | type=2)
        novel_box_type_class_joint_probs = torch.cat(
            (known_novel_box_type_joint_probs, novel_unique_class_cond_probs),
            dim=1
        )

        # Again, normalize to get conditional probability
        # P(S_j=k | >=1 novel label, c) = P(S_j=k | valid type 2)
        novel_box_type_class_probs = novel_box_type_class_joint_probs /\
            novel_box_type_class_joint_probs.sum(dim=1, keepdim=True)

        # Compute class count vector for type 2
        novel_box_type_class_count = novel_box_type_class_probs.sum(dim=0)

        ### Type 2 presence:

        # First, compute P(valid t2)
        # P(valid t2) = P(\union_{known k}(A_k)),
        #   where A_k = "all in {k, novel} but not all k"
        # = \sum_{known k}(P(A_k))
        #       - \sum_{known k1, k2}(P(A_k1, A_k2))
        #       + \sum_{known k1, k2, k3}(P(A_k1, A_k2, A_k3))
        #       - ... 
        # ... NOTE: This is just the formula for the union of N events
        # ... NOTE: (A_k1 AND A_k2) = N = "all novel"
        # ... NOTE: (N AND A_k) = N, for any k
        # = \sum_{known k}(P(A_k))
        #       - \sum_{known k1, k2}(P(N))
        #       + \sum_{known k1, k2, k3}(P(N))
        #       - ...
        # = \sum_{known k}(P(A_k))
        #       + P(N) * (-(K choose 2) + (K choose 3) - ... (K choose K))
        # = \sum_{known k}(P(A_k))
        #       + P(N) * (1 - K)
        
        # P(A_k) = P(all in {k, novel}) - P(all k)
        prob_a_k =\
            torch.prod(known_or_novel_class_probs, dim=0) -\
                torch.prod(known_class_probs, dim=0)

        # P(N) = P(all novel)
        prob_n = torch.prod(combined_novel_class_probs, dim=0)

        # P(valid t2)
        prob_novel_box_type =\
            prob_a_k.sum(dim=0) +\
                prob_n * (1 - known_class_probs.shape[1])

        # Next, compute each P(k present, valid t2).
        
        # For known k:
        # P(k present, valid t2)
        # = P(all boxes s or novel) - P(all boxes s) - P(all boxes novel).
        known_prob_present_novel_box_type =\
            torch.prod(known_or_novel_class_probs, dim=0) -\
                torch.prod(known_class_probs, dim=0) -\
                torch.prod(combined_novel_class_probs, dim=0)

        # For novel s:
        # P(k present, valid t2) = P(valid t2) - P(k absent, valid t2).

        # P(k absent, valid t2)
        # = \union_{known k}(A_k),
        #       where A_k = "all in {k, novel / k} and not all k"
        # = \sum_{known k}(P(A_k))
        #       - \sum_{known k1, k2}(P(A_k1, A_k2))
        #       + \sum_{known k1, k2, k3}(P(A_k1, A_k2, A_k3))
        #       - ... 
        # ... NOTE: This is just the formula for the union of N events
        # ... NOTE: (A_k1 AND A_k2) = N = "all in novel / k"
        # ... NOTE: (N AND A_k) = N, for any k
        # = \sum_{known k}(P(A_k))
        #       - \sum_{known k1, k2}(P(N))
        #       + \sum_{known k1, k2, k3}(P(N))
        #       - ...
        # = \sum_{known k}(P(A_k))
        #       + P(N) * (-(K choose 2) + (K choose 3) - ... (K choose K))
        # = \sum_{known k}(P(A_k))
        #       + P(N) * (1 - K)

        # P(A_k) = P(all in {k, novel / k}) - P(all k)
        novel_not_s_prob =\
            combined_novel_class_probs[:, None] - novel_class_probs
        prob_not_s_maybe_k =\
            known_class_probs[:, None] + novel_not_s_prob[:, :, None]
        prob_absent_s_unique_k = torch.prod(prob_not_s_maybe_k, dim=0)
        prob_a_k =\
            prob_absent_s_unique_k - torch.prod(known_class_probs, dim=0)

        # P(N) = P(all in {novel / k})
        prob_n = torch.prod(novel_not_s_prob, dim=0)

        # P(k absent, valid t2)
        prob_s_absent_novel_box_type = prob_a_k.sum(dim=1) +\
            prob_n * (1 - known_class_probs.shape[1])

        # P(k present, valid t2)
        novel_prob_present_novel_box_type = prob_novel_box_type - prob_s_absent_novel_box_type

        # Concatenate known and novel components for P(k present, valid t2)
        prob_present_novel_box_type = torch.cat(
            (known_prob_present_novel_box_type, novel_prob_present_novel_box_type),
            dim=0
        )

        # Finally, P(k present | valid t2)
        # = P(k present, valid t2) / P(valid t2)
        novel_box_type_class_presence = prob_present_novel_box_type / prob_novel_box_type

        # Compute expectation of class counts and presence / absence 
        # probabilities by mixing the per-type predictions weighted by the
        # corresponding P(Type) probabilities
        p_unique_type = p_type[unique_types].sum()
        p_novel_box = p_type[novel_box_type]
        p_combination = p_type[combination_type]
        class_count = unique_types_class_count * p_unique_type \
            + novel_box_type_class_count * p_novel_box \
            + combination_type_class_count * p_combination
        class_presence = unique_types_class_presence * p_unique_type \
            + novel_box_type_class_presence * p_novel_box \
            + combination_type_class_presence * p_combination

        return class_count, class_presence

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

            species_count, species_presence = self._counts_presence(
                cur_species_probs,
                cur_p_type,
                self._n_known_species_cls,
                [0, 2, 4, 5],
                1,
                3
            )
            activity_count, activity_presence = self._counts_presence(
                cur_activity_probs,
                cur_p_type,
                self._n_known_activity_cls,
                [0, 1, 3, 5],
                2,
                4
            )
            predictions.append((
                species_count,
                species_presence,
                activity_count,
                activity_presence
            ))

        return predictions
