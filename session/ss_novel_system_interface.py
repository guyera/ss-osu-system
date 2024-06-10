# This software was created by Oregon State University under Army Research
# Office (ARO) Award Number W911NF-22-2-0149. ARO, as the Federal awarding
# agency, reserves a royalty-free, nonexclusive and irrevocable right to
# reproduce, publish, or otherwise use this software for Federal purposes, and
# to authorize others to do so in accordance with 2 CFR 200.315(b).

from abc import ABC, abstractmethod

class SSNovelSystemInterface(ABC):
    @abstractmethod
    def start_session(self, session_id):
        """
        Called by bbn_session to signal to the system that a session of test
        trials has started.
        :param session_id: Session ID for connection to SS API Server (likely
            unused by the system)
        :return: None
        """
        pass

    def start_test(self, test_id):
        """
        Called by bbn_session to signal to the system that a test trial has
        started.
        :param test_id: ID of the current test trial (likely unused by the
            system except for logging purposes)
        :return: None
        """
        pass

    def given_detect_red_light(self, red_light_image_path):
        """
        Called by bbn_session during given_detection trials at the beginning
        of the round in which the red light image occurs, and passing the path
        of that red light image. (Given detection mode is a task mode wherein
        the system is given the point in time at which the first novel image
        appears in the trial; this mode is unconventional and is only used to
        run oracle studies on methods that employ change point detection)
        :param red_light_image_path: Path of the first novel image in the trial
            (also in the round)
        :return: None
        """
        pass

    def process_round(self, test_id, round_id, image_paths, bbox_dict, hint_type_a_data, hint_type_b_data):
        """
        :param test_id: ID of the current test trial (likely unused by the
            system except for logging purposes)
        :param round_id: ID of the current round within the trial
        :param image_paths: list of strings representing image paths for the
            current round
        :param bbox_dict: dictionary mapping image filenames to bounding boxes
        :param hint_type_a_data: None, unless hint A is enabled in the current
            trial, in which case this parameter is an Integer in
            {0, 2, 3, 4, 5, 6} representing the novelty type of the trial.
            Used primarily for oracle studies.
            0 corresponds to a no-novelty trial.
            2 corresponds to a trial with novel species classes.
            3 corresponds to a trial with novel activity classes.
            4 corresponds to a trial with novel combinations of known species.
            5 corresponds to a trial with novel combinations of known
                activities.
            6 corresponds to a trial with novel environment conditions.
        :param hint_type_b_data: None, unless hint B is enabled in the current
            trial, in which case this parameter is a list of N booleans, where
            N is the number of images in the current round, indicating whether
            each image contains novelty (True to indicate that some form of
            novelty is present, False otherwise). Used primarily for oracle
            studies.
        :return: tuple of the form (novelty_preds, svo_preds).
            - novelty_preds: A list of N tuples of the form
                (image_path, red_light_score, p_novel), where N is the number of
                images in the round.
                - red_light_score: Predicted probability that the trial's
                    post-novelty phase has begun as of the ith image of the
                    round
                - p_novel: Predicted probability that the ith image of the
                    round contains any sort of novelty
            - svo_preds: A list of N tuples of the form (species_count,
                species_presence, activity_count, activity_presence), where N
                is th enumber of images in the round.
                - species_count: Torch tensor of shape [S], where S is the
                    maximum number of species classes (including all known
                    classes and the maximum number of novel species classes),
                    containing the predicted counts of each corresponding
                    species in image i.
                - species_presence: Torch tensor of shape [S], where S is the
                    maximum number of species classes (including all known
                    classes and the maximum number of novel species classes),
                    containing the predicted probability of each corresponding
                    species being present in image i.
                - activity_count: Torch tensor of shape [S], where S is the
                    maximum number of activity classes (including all known
                    classes and the maximum number of novel activity classes),
                    containing the predicted counts of each corresponding
                    activity in image i.
                - activity_presence: Torch tensor of shape [S], where S is the
                    maximum number of activity classes (including all known
                    classes and the maximum number of novel activity classes),
                    containing the predicted probability of each corresponding
                    activity being present in image i.
        TODO should do characterize_round? Or is that even part of the
        benchmark? I don't think so.
        """
        pass


    def choose_detection_feedback_ids(self, test_id, round_id, image_paths, feedback_max_ids):
        """
        Called by bbn_session to ask the system which images from the round
        about which to query feedback
        :param test_id: ID of the current test trial (likely unused by the
            system except for logging purposes)
        :param round_id: ID of the just-completed round within the trial
        :param image_paths: list of image_paths from the just-completed round
        :param feedback_max_ids: number of image paths to pick from image_paths
        :return: list of selected image paths from image_paths
        """
        pass


    def record_detection_feedback(self, test_id, round_id, feedback_csv_content, bboxes):
        """
        :param test_id: ID of the current test trial (likely unused by the
            system except for logging purposes)
        :param round_id: ID of the just-completed round within the trial
        :param feedback_csv_content: A string containing the subset of the
            ground truth data CSV file pertaining to the images queried for
            feedback. This CSV content can be handled in the same way as the
            CSV content for the rest of the trial data. For example, this string
            can be dumped to a csv file, and that file's path can be passed to
            the constructor of a BoxImageDataset.
        :param bboxes: Dict mapping image paths to bounding box data for
            the images queried for feedback. This dictionary can be dumped
            to a json file with the same basename as a csv file containing the
            feedback_csv_content and subsequently loaded by a BoxImageDataset
            by passing the csv file path to the BoxImageDataset constructor.
        :return: None
        """
        pass


    def end_test(self, test_id):
        """
        Called by bbn_session to signal to the system that the current test
        has just ended (e.g., to initiate cleanup or reset models, perhaps in
        preparation for another test within the same session)
        :param test_id: ID of the just-completed test trial (likely unused by
            the system except for logging purposes)
        """
        pass


    def end_session(self, session_id):
        """
        Called by bbn_session to signal to the system that the current session
        has just ended.
        :param session_id: ID of the just-completed session (likely unused by
            the system except for logging purposes)
        """
        pass
