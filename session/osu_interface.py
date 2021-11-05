import pathlib
import numpy as np
import os
import pandas as pd
from toplevel import TopLevelApp


class OSUInterface:
    def __init__(self, scg_ensemble, num_subject_classes, num_object_classes, num_verb_classes, 
        data_root, pretrained_unsupervised_novelty_path, cusum_thresh):

        self.app = TopLevelApp(ensemble_path=scg_ensemble, 
            num_subject_classes=num_subject_classes, 
            num_object_classes=num_object_classes, 
            num_verb_classes=num_verb_classes, 
            data_root=data_root, 
            pretrained_unsupervised_module_path=pretrained_unsupervised_novelty_path,
            th=cusum_thresh)

        self.declare_novelty_round = 3
        self.declare_novelty_image = 15

        self.temp_path = pathlib.Path('./session/temp/')
        self.batch_csv_path = self.temp_path.joinpath('batch.csv')

    def start_session(self, session_id, detection_feedback, classification_feedback, given_detection):
        """
        :param session_id:
        :param detection_feedback:  We assume this weill be True
        :param classification_feedback:  We assume this weill be False
        :param given_detection: I haven't figure out yet how this will work
        :return: None
        """
        print(f'==> OSU got start session {session_id}')

    def start_test(self, test_id):
        """
        :param test_id:
        :return: None
        """
        print(f'==> OSU got start test {test_id}')

    def process_round(self, test_id, round_id, contents):
        """
        :param test_id:
        :param round_id:
        :param contents: string of csv lines with image_path and bounding box data
        :return: tuple of (novelty_preds, svo_preds)
        novelty_preds: string of csv lines with image_path, red_light_score, and per_image novelty score
        svo_preds: string of csv lines with image_path, and top 3 (S_id, V_id, O_id, prob) values
        """

        df = pd.DataFrame(columns=['new_image_path', 'subject_name', 'subject_id', 'original_subject_id',
            'object_name', 'object_id', 'original_object_id', 'verb_name',
            'verb_id', 'original_verb_id', 'image_width', 'image_height',
            'subject_ymin', 'subject_xmin', 'subject_ymax', 'subject_xmax',
            'object_ymin', 'object_xmin', 'object_ymax', 'object_xmax'])

        lines = [line.split(',') for line in contents.splitlines()]

        create_row = lambda idx, line: [line[0], None, None, None, None, None, None, None, None, None,
            line[1], line[2], 
            line[3], line[4], line[5], line[6], 
            line[7], line[8], line[9], line[10]]

        idx = 0
        for l in lines:
            row = create_row(idx, l)
            df.loc[idx] = row
            idx += 1

        df.to_csv(self.batch_csv_path, index=True)
        
        ret = self.app.run(self.batch_csv_path)
        p_ni = ret['p_ni']
        red_light_score = ret['red_light_score']
        top_1 = ret['svo']
        top_1_probs = ret['svo_probs']

        novelty_lines = [f'{img}, {red_light_score:e}, {p:e}' for img, p in zip(df['new_image_path'].to_list(), p_ni)]
        novelty_preds = '\n'.join(novelty_lines)
        svo_preds = '\n'.join([f'{img}, {(svo[0], svo[1], svo[2], p)}' for img, svo, p in zip(df['new_image_path'].to_list(), top_1, top_1_probs)])

        # temp_features_path = self.temp_path.joinpath('batch_novelty_features.pth')
        # if temp_features_path.exists():
        #     os.remove(temp_features_path)

        print(f'  ==> OSU processed round {round_id}')
        
        return (novelty_preds, svo_preds)

    def choose_detection_feedback_ids(self, test_id, round_id, image_paths, feedback_max_ids):
        """
        :param test_id:
        :param round_id:
        :param image_paths: list of image_paths from the just_completed round
        :param feedback_max_ids: number of image_paths to pick
        :return: list of selected image_paths
        """
        queries = self.app.select_queries(feedback_max_ids)
        return queries

    def record_detection_feedback(self, test_id, round_id, feedback_results):
        """
        :param test_id:
        :param round_id:
        :param feedback_results: dict with boolean value for each image_path
        :return: Null
        """
        print(f'  ==> OSU got detection feedback results for '
            f'{len(feedback_results)} images for round {round_id}')

        d = {}
        for e in feedback_results:
            d[e[0]] = e[1]

        self.app.feedback_callback(d)

    def choose_classification_feedback_ids(self, filenames, feedback_max_ids):
        raise NotImplementedError

    def record_classification_feedback(self, feedback_results):
        raise NotImplementedError

    def end_test(self, test_id):
        print(f'==> OSU got end test {test_id}')

    def end_session(self, session_id):
        print(f'==> OSU got end session {session_id}')

    # def _red_light(self, round_id):
    #     return round_id >= self.declare_novelty_round

    # def _red_light_score(self, round_id, image_id):
    #     if round_id < self.declare_novelty_round:
    #         return random.uniform(0.01, 0.48)
    #     elif round_id == self.declare_novelty_round:
    #         if image_id < self.declare_novelty_image:
    #             return random.uniform(0.01, 0.48)
    #         elif image_id == self.declare_novelty_image:
    #             return 0.75
    #         else:
    #             return random.random()
    #     else:
    #         return random.random()

    # def _image_novelty_score(self):
    #     return random.random()

    # def _random_svo_preds(self, top_k = 3):
    #     return ','.join(self._random_svo_pred() for i in range(top_k))

    # def _random_svo_pred(self):
    #     return ','.join([str(random.randrange(0,5)),
    #                     str(random.randrange(0,5)),
    #                     str(random.randrange(0,5)),
    #                     f'{random.random():3}'])

    # def _novel_p(self, image_path):
    #     substrings = ['novel_val', 'imgs', 'verb_dataset', 'obj_dataset', 'sub_dataset']
    #     return any([substring in image_path for substring in substrings])

