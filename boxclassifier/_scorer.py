from abc import ABC, abstractmethod

class Scorer(ABC):
    '''
    Produces novelty signals in the form of scalar scores for each image,
    given the box classifier's logit predictions.

    Parameters:
        species_logits: Tensor of shape MxS, where M is the total number of
            boxes across the entire batch and S is the number of species
            classes.
        activity_logits: Tensor of shape MxA, where A is the number of activity
            classes.
        box_counts: Iterable[Int] of length N, where N is the number of
            images in the batch and box_counts[i] specifies the number of boxes
            for image i. Logits are presented in image-order; that is, the
            logits for the boxes of an image are all consecutive.

    Returns:
        Tensor of shape MxL, where L is the number of scores produced for
            each image.
    '''
    @abstractmethod
    def score(self, species_logits, activity_logits, box_counts):
        return NotImplemented

    '''
    Returns L, the number of scores produced by the scorer. Necessary for
    constructing the novelty type logistic regressions (novelty
    characterization + detection)
    '''
    @abstractmethod
    def n_scores(self):
        return NotImplemented

class MaxAverageLogitScorer(Scorer):
    def score(self, species_logits, activity_logits, box_counts):
        split_species_logits = torch.split(species_logits, box_counts, dim=0)
        split_activity_logits = torch.split(activity_logits, box_counts, dim=0)
        avg_max_species_logits = []
        max_avg_species_logits = []
        avg_max_activity_logits = []
        max_avg_activity_logits = []
        for img_species_logits, img_activity_logits in\
                zip(split_species_logits, split_activity_logits):
            img_max_species_logits, _ = torch.max(img_species_logits, dim=1)
            img_avg_species_logits = img_species_logits.mean(dim=0)
            img_avg_max_species_logit = img_max_species_logits.mean()
            img_max_avg_species_logit, _ =\
                torch.max(img_avg_species_logits, dim=0)
            avg_max_species_logits.append(img_avg_max_species_logit)
            max_avg_species_logits.append(img_max_avg_species_logit)

            img_max_activity_logits, _ = torch.max(img_activity_logits, dim=1)
            img_avg_activity_logits = img_activity_logits.mean(dim=0)
            img_avg_max_activity_logit = img_max_activity_logits.mean()
            img_max_avg_activity_logit, _ =\
                torch.max(img_avg_activity_logits, dim=0)
            avg_max_activity_logits.append(img_avg_max_activity_logit)
            max_avg_activity_logits.append(img_max_avg_activity_logit)

        avg_max_species_logits = torch.stack(avg_max_species_logits, dim=0)
        max_avg_species_logits = torch.stack(max_avg_species_logits, dim=0)
        avg_max_activity_logits = torch.stack(avg_max_activity_logits, dim=0)
        max_avg_activity_logits = torch.stack(max_avg_activity_logits, dim=0)
        
        scores = torch.stack(
            (
                avg_max_species_logits,
                max_avg_species_logits,
                avg_max_activity_logits,
                max_avg_activity_logits
            ),
            dim=1
        )
        return scores

    def n_scores(self):
        return 4
