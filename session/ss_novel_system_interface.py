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
        :param hint_type_a_data: TODO
        :param hint_type_b_data: TODO
        :return: tuple of (novelty_preds, svo_preds).
            novelty_preds: TODO
            svo_preds: TODO
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
        :param feedback_csv_content: TODO
        :param bboxes: TODO
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
