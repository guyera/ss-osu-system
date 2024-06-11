# This software was created by Oregon State University under Army Research
# Office (ARO) Award Number W911NF-22-2-0149. ARO, as the Federal awarding
# agency, reserves a royalty-free, nonexclusive and irrevocable right to
# reproduce, publish, or otherwise use this software for Federal purposes, and
# to authorize others to do so in accordance with 2 CFR 200.315(b).

import json
import uuid
import pathlib
import os
import pandas as pd
from toplevel import TopLevelApp
import csv

class OSUInterface:
    def __init__(self, scg_ensemble, data_root, pretrained_models_dir, backbone_architecture,
            feedback_enabled, given_detection, log, log_dir, ignore_verb_novelty, train_csv_path, val_csv_path,
            trial_batch_size, trial_size, disable_retraining,
            root_cache_dir, n_known_val, classifier_trainer, precomputed_feature_dir, retraining_augmentation, retraining_lr, retraining_batch_size, retraining_val_interval, retraining_patience,
            retraining_min_epochs, retraining_max_epochs, retraining_label_smoothing, retraining_scheduler_type, feedback_loss_weight, retraining_loss_fn, balance_class_frequencies, gan_augment, device, retrain_fn, val_reduce_fn, model_unwrap_fn, feedback_sampling_configuration, oracle_training, ewc_lambda):

        self.app = TopLevelApp( 
            data_root=data_root,
            pretrained_models_dir=pretrained_models_dir,
            backbone_architecture=backbone_architecture,
            feedback_enabled=feedback_enabled,
            given_detection=given_detection, 
            log=log,
            log_dir=log_dir,
            ignore_verb_novelty=ignore_verb_novelty,
            train_csv_path=train_csv_path,
            val_csv_path=val_csv_path,
            trial_size=trial_size,
            trial_batch_size=trial_batch_size,
            disable_retraining=disable_retraining,
            root_cache_dir=root_cache_dir,
            n_known_val=n_known_val,
            classifier_trainer=classifier_trainer,
            precomputed_feature_dir=precomputed_feature_dir,
            retraining_augmentation=retraining_augmentation,
            retraining_lr=retraining_lr,
            retraining_batch_size=retraining_batch_size,
            retraining_val_interval=retraining_val_interval,
            retraining_patience=retraining_patience,
            retraining_min_epochs=retraining_min_epochs,
            retraining_max_epochs=retraining_max_epochs,
            retraining_label_smoothing=retraining_label_smoothing,
            retraining_scheduler_type=retraining_scheduler_type,
            feedback_loss_weight=feedback_loss_weight,
            retraining_loss_fn=retraining_loss_fn,
            balance_class_frequencies=balance_class_frequencies,
            gan_augment=gan_augment,
            device=device,
            retrain_fn=retrain_fn,
            val_reduce_fn=val_reduce_fn,
            model_unwrap_fn=model_unwrap_fn,
            feedback_sampling_configuration=feedback_sampling_configuration,
            oracle_training = oracle_training,
            ewc_lambda = ewc_lambda
        )
        self.num_queries = 0
        self.num_of_queries = []
        self.log_dir =log_dir
        self.temp_path = pathlib.Path('./session/temp/')
        self.temp_path.mkdir(parents=True, exist_ok=True)

    def start_session(self, session_id):
        """
        :param session_id:
        :return: None
        """
        print(f'==> OSU got start session {session_id}')

    def start_test(self, test_id):
        """
        :param test_id:
        :return: None
        """
        self.app.reset()
        self.num_queries = 0


        print(f'==> OSU got start test {test_id}')

    def given_detect_red_light(self, red_light_image_path):
        """
        Called by bbn_session during given_detectinon trials at the beginning
        of the round in which the red light image occurs,
        and passing the path of that red light image.
        :param red_light_image_path
        :return: None
        """
        self.app.red_light_hint_callback(red_light_image_path)

    def process_round(self, test_id, round_id, image_paths, bbox_dict, hint_typeA_data, hint_typeB_data):
        """
        :param test_id:
        :param round_id:
        :param image_paths: list of strings of representing image paths
        :param bbox_dict: dictionary mapping image filenames to bounding boxes
        :return: tuple of (novelty_preds, svo_preds)
        novelty_preds: string of csv lines with image_path, red_light_score, and per_image novelty score
        predictions: predictions as returned by TuplePredictor
        """

        df = pd.DataFrame(columns=['image_path', 'filename', 'capture_id', 'width', 'height', 'agent1_name', 'agent1_id', 'agent1_count', 'agent2_name', 'agent2_id', 'agent2_count', 'agent3_name', 'agent3_id', 'agent3_count', 'activities', 'activities_id', 'environment', 'novelty_type', 'master_id'])

        assert len(image_paths) > 0, "content was empty"

        create_row = lambda path, name: [path, name, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]

        for idx, image_path in enumerate(image_paths):
            image_name = os.path.basename(image_path)
            row = create_row(image_path, image_name)
            df.loc[idx] = row

        csv_path = self.temp_path.joinpath(f'{os.getpid()}_batch_{round_id}.csv')
        json_path = self.temp_path.joinpath(f'{os.getpid()}_batch_{round_id}.json')
        df.to_csv(csv_path, index=True)
        with open(json_path, 'w') as f:
            json.dump(bbox_dict, f)

        ret = self.app.process_batch(csv_path, test_id, round_id, df['image_path'].to_list(), hint_typeA_data, hint_typeB_data)
        p_ni = ret['p_ni']
        red_light_scores = ret['red_light_score']
        predictions = ret['predictions']

        novelty_preds = [(float(rs), float(p)) for rs, p in zip(red_light_scores, p_ni)]

        if csv_path.exists():
            os.remove(csv_path)
        if json_path.exists():
            os.remove(json_path)

        print(f'  ==> OSU processed round {round_id}')
        # import ipdb; ipdb.set_trace()

        return (novelty_preds, predictions)

    # def select_oracle(test_id, round_id, image_paths, feedback_max_ids):
    #     api_test_folder = '/nfs/hpc/share/sail_on3/final/test_trials/api_tests/OND/image_classification/'
    #     csv_path = api_test_folder +'/'+ test_id+'_single_df.csv'
    #     with open(csv_path, "r") as f:
    #         csv_reader = csv.reader(f, delimiter=",", quotechar='"', skipinitialspace=True)
    #         rows = [x for x in csv_reader][round_id*10:round_id+10]
    #     sort these rows according to novelty column
    #     select top 5



    def choose_detection_feedback_ids(self, test_id, round_id, image_paths, feedback_max_ids):
        """
        :param test_id:
        :param round_id:
        :param image_paths: list of image_paths from the just_completed round
        :param feedback_max_ids: number of image_paths to pick
        :return: list of selected image_paths
        """
        queries, bboxes = self.app.select_queries(feedback_max_ids)
        # queries, bboxes = select_oracle(test_id, round_id, image_paths, feedback_max_ids)
        return queries, bboxes

    def record_detection_feedback(self, test_id, round_id, feedback_csv_content, bboxes):
        """
        :param test_id:
        :param round_id:
        :param feedback_results: dict with boolean value for each image_path
        :return: Null
        """
        print(f'  ==> OSU got detection feedback results for round {round_id}')

        feedback_uuid = uuid.uuid4()
        
        # Count the number of rows in feedback_csv_content
        # Each line in CSV is separated by a newline character
        num_rows = feedback_csv_content.count('\n')  # Counts the number of newlines
        if round_id < 200:
            self.num_queries += num_rows
        csv_path = self.temp_path.joinpath(f'{os.getpid()}_batch_{round_id}_feedback_{feedback_uuid}.csv')
        with open(csv_path, 'w') as f:
            f.write(feedback_csv_content)
        json_path = self.temp_path.joinpath(f'{os.getpid()}_batch_{round_id}_feedback_{feedback_uuid}.json')
        with open(json_path, 'w') as f:
            json.dump(bboxes, f)
        self.app.feedback_callback(csv_path)

    def end_test(self, test_id):
        print(f'==> OSU got end test {test_id}')
        self.num_of_queries.append([test_id, self.num_queries])
        returned_pni = self.app.test_completed_callback(test_id)
        # import numpy as np
        # np.savetxt(test_id+'.csv', returned_pni, delimiter=',', header='p_ni')


        

    def end_session(self, session_id):
        # Write to CSV

        csv_pathh = self.log_dir+ '/' +session_id +'_Number_of_Feedback_Queries.csv'
        with open(csv_pathh, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Test ID', 'Number of Queries'])  # Writing header
            writer.writerows(self.num_of_queries)

        print(f'==> OSU got end session {session_id}')

