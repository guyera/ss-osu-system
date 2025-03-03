#########################################################################
# Copyright 2008-2022 by Raytheon BBN Technologies.  
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this code and associated documentation files (the "Code"), to deal
# in the Code without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Code, and to permit persons to whom the Code is
# furnished to do so, subject to the following conditions:
#
# This copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Code.
#
# BBN makes no warranties regarding the fitness of the code for any use or
# purpose.
#########################################################################

import ast
import io
import json
import os
from argparse import ArgumentParser
import requests
from seaborn.distributions import ecdfplot
from session.api_stubs import APIStubs
import itertools
import numpy as np

class BBNSession:
    def __init__(self, protocol, domain, class_count, detection_fb, given_detection,
                 image_directory, results_directory, api_url,
                 batch_size, version, detection_threshold,
                 api_stubs, osu_interface, hintA, hintB, feedback_budget_override=None):

        # self.agent = Agent(config['detectormodelpath'],
        #                    config['classifiermodelpath'],
        #                    config['redlightdistributionpath'],
        #                    config['redlightscalefactor'],
        #                    config['redlightthreshold'])
        self.version = version
        self.protocol = protocol
        self.domain = domain
        self.class_count = class_count
        self.detection_fb = detection_fb
        self.given_detection = given_detection
        self.results_directory = results_directory
        self.url = api_url
        self.version = version
        self.detection_threshold = detection_threshold
        self.image_directory = image_directory
        self.results_directory = results_directory
        self.given_detect_red_light_round = None
        # self.characterization = characterization
        # self.bbn_max_clusters = config['bbnmaxclusters']
        # self.bbn_min_novel_frac = config["bbnminnovelfrac"]
        # self.red_light_threshold = config['redlightthreshold']
        # self.test_count = test_count
        self.batch_size = batch_size
        self.history = None  # TestHistory for current test
        self.api_stubs = api_stubs
        self.osu_stubs = osu_interface
        # These are also defined in toplevel
        self.n_species_cls = 31 # TODO pass this in from main
        self.n_activity_cls = 7 # TODO pass this in from main
        self.hintA = hintA
        self.hintB = hintB
        self.feedback_budget_override = feedback_budget_override

        # Turn this to True to get readable (S,O,V) triples for our top-3 in classification output.
        # Default is to get the 945 floats that UMD is requesting. 
        self.probs_debug_format = False

        if not os.path.exists(self.results_directory):
            os.makedirs(self.results_directory)

        if self.api_stubs:
            self.api_stubs.clear_results_dirs()

    def request_feedback(self, session_id, test_id, round_id, feedback_ids):
        # print(f'====> Requesting detection feedback on round {round_id} for {len(feedback_ids)} images.')
        if self.api_stubs:
            return self.api_stubs.detection_feedback(test_id, round_id, feedback_ids)
        else:
            response = requests.get(
                f'{self.url}/session/feedback',
                {'session_id': session_id,
                 'test_id': test_id,
                 'round_id': round_id,
                 'feedback_type': 'classification',
                 'feedback_ids': '|'.join(feedback_ids),
                 }
            )

            if response.status_code == 500:
                raise Exception(f'Got error response from feedback request: {response}')
            
            feedback_csv_content = response.content.decode('utf-8')

            return feedback_csv_content

    def request_hint_typeA(self, session_id, test_id):
        print(f'====> Asking for hint on test_id {test_id}')

        response = requests.get(
            f'{self.url}/session/hint',
            {'session_id': session_id,
                'test_id': test_id,
                'hint_type': 'typeA',
                }
        )

        data = response.content.decode('utf-8').split('\n')
        # import ipdb; ipdb.set_trace()

        print("TYPE A HINT ==> ")
        print(data)

        return data

    def request_hint_typeB(self, session_id, test_id, round_id):
        print(f'====> Asking for hint on test_id {test_id}')

        response = requests.get(
            f'{self.url}/session/hint',
            {'session_id': session_id,
                'test_id': test_id,
                'round_id': round_id,
                'hint_type': 'typeB',
                }
        )

        data = response.content.decode('utf-8').split('\n')
        # print("TYPE B HINT ==> ")
        # print(data)

        return data


    # def force_to_missing(self, s, v, o, missing_s, missing_o, free_s, free_v, free_o):
    #     """Force our answer to be consistent with missing_s and missing_o"""
    #     ## Use the free_s, _v, _o vars to ensure that our three new forced answers are distinct.
    #     if missing_s:
    #         if s != -1 or v != 0 or o == -1:
    #             self.missing_box_problems += 1
    #             print(f' ======> Missing box s: hypo has s or v or does not have o')
    #         # S and V must be missing
    #         s = -1
    #         v = 0
    #         # O must not be missing
    #         if o == -1:
    #             o = list(free_o)[0]
    #             free_o.remove(o)
    #     elif missing_o:
    #         if o != -1 or s == -1:
    #             self.missing_box_problems += 1
    #             print(f' =======> Missing box o: hypo has o or does not have s')
    #         # O must be missing
    #         o = -1
    #         # S must not be missing
    #         if s == -1:
    #             s = list(free_s)[0]
    #             free_s.remove(s)
    #     else:
    #         if s == -1 or o == -1:
    #             self.missing_box_problems += 1
    #             print(f' ======> Missing box: missing hypo when nothing missing')
    #         # neither S nor O can be missing
    #         if s == -1:
    #             s = list(free_s)[0]
    #             free_s.remove(s)
    #         if o == -1:
    #             o = list(free_o)[0]
    #             free_o.remove(o)
    #     return s, v, o, free_s, free_v, free_o
        

    def write_file_entries(self, filename, detection_file, classification_file,
                           species_counts, species_presence, activity_presence,
                           # red_light,
                           red_light_score, image_novelty_score,
                           round_id, red_light_declared):  ##, top_layer):
        # UMD lists the K+1 probability in position 0, ahead of the K known classes.
        # classification_probs = np.insert(
        #     predicted_probs, 0, 0.0)

        # # If novelty has been declared (and test_type not predicted as T2 or T3),
        # # insert K+1 with top probability, rescaling the others.
        # if red_light and self.history.test_type != TestType.PROB_2_3:
        #     classification_probs[0] = np.amax(classification_probs) + .0001
        #     classification_probs /= np.sum(classification_probs, keepdims=1)

        # self.history.add(filename, round_id, red_light, image_novelty_score, red_light_score,
        #                  classification_probs, top_layer)

        if self.probs_debug_format:
            output_probs = predicted_probs
        else:
            species_counts_str =\
                ','.join([str(x) for x in species_counts.tolist()])
            species_presence_str =\
                ','.join([str(x) for x in species_presence.tolist()])
            activity_presence_str =\
                ','.join([str(x) for x in activity_presence.tolist()])
            # ## LAR
            # try:
            #     predicted_probs = predicted_probs.replace('inf', '"inf"')
            #     answers = ast.literal_eval(f'({predicted_probs})')
            # except ValueError:
            #     print(f'Hit error on filename: {filename}')
            #     print(f'predicted_probs: {predicted_probs}')
            #     breakpoint()
            # ## LAR
            # assert len(answers) == 3
            # free_s = set(range(1, self.num_subject_classes+1))
            # free_v = set(range(1, self.num_verb_classes+1))
            # free_o = set(range(1, self.num_object_classes+1))
            # for answer in answers:
            #     s, v, o, prob = answer
            #     free_s.discard(s)
            #     free_v.discard(v)
            #     free_o.discard(o)
            # output_triples = []
            # for answer in answers:
            #     s, v, o, prob = answer
            #     ## LAR
            #     if prob == 'inf':
            #         prob = 0.0001
            #     ## LAR
            #     # for V, id 0 covers both missing and novel
            #     if v == -1:
            #         v = 0
            #     s, v, o, free_s, free_v, free_o = self.force_to_missing(s, v, o, missing_s, missing_o,
            #                                                             free_s, free_v, free_o)
            #     triple = (s, v, o)
            #     loop_count = 0
            #     while triple in output_triples:
            #         # change one value randomly to avoid duplication
            #         loop_count += 1
            #         if loop_count > 3:
            #             raise Exception('Excessive looping to avoid duplicates')
            #         if s != -1:
            #             s = list(free_s)[0]
            #             free_s.remove(s)
            #         elif o != -1:
            #             o = list(free_o)[0]
            #             free_o.remove(o)
            #         else:
            #             raise Exception('One of S or O should not be missing.')
            #         triple = (s, v, o)
            #     output_triples.append(triple)
            #     output_vals[self.triple_dict[triple]] = prob
            # output_probs = ','.join([f'{val:e}' for val in output_vals])            
    
        if self.given_detection:
            if red_light_declared:
                red_light_score = 1.0
            else:
                red_light_score = 0.0
        detection_file.write("%s,%e,%e\n" %
                             (filename, red_light_score, image_novelty_score))
        classification_file.write("%s,%s,%s,%s\n" %
                                  (filename, species_counts_str, species_presence_str, activity_presence_str))

    def run(self, detector_seed, test_ids=None):
        print('Running')
        print('test_ids:', test_ids)
        if not test_ids:
            # test_ids = requests.get(
            #     f"{self.url}/test/ids?protocol={self.protocol}&detector_seed={detector_seed}").content.decode('utf-8').split('\n')
            ##result = requests.get(f"{self.url}/test/ids?protocol={self.protocol}&detector_seed={detector_seed}")
            ### To test if UMD eval would work.
            ### self.domain = 'image_classification'
            if self.api_stubs:
                test_ids = self.api_stubs.test_ids()
            else:
                result = requests.get(
                    f"{self.url}/test/ids?protocol={self.protocol}&detector_seed={detector_seed}&domain={self.domain}")

                test_ids = result.content.decode('utf-8').split('\n')

                # Need to remove possible final empty strings
                test_ids = [test_id for test_id in test_ids if test_id.strip("\n\t\"',.") != ""]
        
        # import ipdb; ipdb.set_trace()
        self.hints = []
        if self.detection_fb:
            self.hints.append('novelty_instance_detection')
        if self.given_detection:
            self.hints.append('red_light')
        if self.api_stubs:
            session_id = self.api_stubs.session_id()
        else:
            response = requests.post(f"{self.url}/session", json={
                'configuration': {
                    'protocol': self.protocol,
                    'novelty_detector_version': self.version,
                    'test_ids': test_ids,
                    'domain': self.domain,
                    # 'hints': self.hints,
                    'detection_threshold': self.detection_threshold, #self.detection_threshold,
                    'hints': ['red_light' if self.given_detection else '']
                }
            })
            # print(response.content.decode('utf-8'))
            session_id = ast.literal_eval(response.content.decode('utf-8'))['session_id']
        if self.osu_stubs:
            self.osu_stubs.start_session(session_id)


        print(f"=> initialized session: {session_id}")

        for test_id in test_ids:
            ## GDD
            # if test_id == 'OND.100.000':
            #     print(f'==> skipping test: {test_id}')
            #     continue
            print(f"==> starting test: {test_id}")
            self.run_test(session_id, test_id)
            # self.agent.reset()

        print("=> terminating session")

        if self.api_stubs:
            pass
        else:
            requests.delete(f"{self.url}/session?session_id={session_id}")
        if self.osu_stubs:
            self.osu_stubs.end_session(session_id)

    def run_test(self, session_id, test_id):

        # self.history = TestHistory()
        # Request Hint Type A for given session Id and Test Id --> [test_id, kind_of_novelty]  
        if self.hintA:      
            hint_typeA = self.request_hint_typeA(session_id, test_id)[0].split(',')[-1]
            if hint_typeA == 'object':
                hint_typeA_data = 3
            elif hint_typeA == 'event':
                hint_typeA_data = 6
            elif hint_typeA == 'environment':
                hint_typeA_data = 7
            elif hint_typeA == 'type0':
                hint_typeA_data = 0
            else: 
                # hint_typeA_data = None
                hint_typeA_data = 0
        else:
            hint_typeA_data = None
        print('Hint A is : ',hint_typeA_data)
        if self.api_stubs:
            metadata = self.api_stubs.get_metadata(test_id)
        else:
            metadata = ast.literal_eval(requests.get(
                f"{self.url}/test/metadata?session_id={session_id}&test_id={test_id}")
                                        .content.decode('utf-8'))
        print(metadata)
        if self.osu_stubs:
            self.osu_stubs.start_test(test_id)

        round_size = metadata['round_size']
        feedback_max_ids = metadata['feedback_max_ids']
        if self.feedback_budget_override is not None:
            feedback_max_ids = self.feedback_budget_override
        trial_feedback_budget = 0

        if self.given_detection:
            red_light_image = metadata['red_light']
        else:
            red_light_image = None
        # max_novel_classes = metadata['max_novel_classes']
        # print(f'===> Max_novel_classes: {max_novel_classes}')
        red_light_declared = False

        round_id = 0

        # Request Hint Type A for given session Id, Test Id and round_id --> ['fname, 0/1', ... ]
        
        hintsBList = []
        while True:
            image_data = None
            filenames = None
            if self.given_detection and self.hintB:
                hint_typeB_data = self.request_hint_typeB(session_id, test_id, round_id)
                hint_typeB_data = [bool(int(hint.split(',')[-1])) for hint in hint_typeB_data[:-1]]
                print(hint_typeB_data)
            else:
                hint_typeB_data = None

            hintsBList.append(hint_typeB_data)
            
            if self.api_stubs:
                raise NotImplementedError
            else:
                dataset_response = requests.get(
                    f"{self.url}/session/dataset?session_id={session_id}&test_id={test_id}&round_id={round_id}")

                if dataset_response.status_code == 204:
                    print("==> No more rounds")
                    break

                print(f"===> Round {round_id}")

                response_dict = dataset_response.json()
                image_paths = response_dict['images']
                bbox_dict = response_dict['bboxes']

                ## GD
                if self.given_detect_red_light_round is None and self.given_detection and (red_light_image in image_paths):
                    self.given_detect_red_light_round = round_id
                    
                ## GDD
                # if self.given_detect_red_light_round == round_id:
                #     print(f' **** Turned on given_detect_red_light_round on round {round_id}')


            detection_filename = os.path.join(
                self.results_directory, "%s_%s_%s_detection.csv" % (session_id, test_id, round_id))
            detection_file = open(detection_filename, "w")
            if round_id == 0:
                detection_file.write('image_path,red_light_prob,per_image_prob\n')
            classification_filename = os.path.join(
                self.results_directory, "%s_%s_%s_classification.csv" % (session_id, test_id, round_id))
            classification_file = open(classification_filename, "w")
            if round_id == 0:
                species_count_cols = [f'species_{x}_count' for x in range(self.n_species_cls)]
                species_presence_cols = [f'species_{x}_presence' for x in range(self.n_species_cls)]
                activity_presence_cols = [f'activity_{x}_presence' for x in range(self.n_activity_cls)]
                species_count_str = ','.join(species_count_cols)
                species_presence_str = ','.join(species_presence_cols)
                activity_presence_str = ','.join(activity_presence_cols)
                classification_file.write(f'image_path,{species_count_str},{species_presence_str},{activity_presence_str}\n')

            if self.osu_stubs:
                ## GD
                # If we're in a given detection trial and this round contains the red light,
                # inform the OSU code of that, passing them the red light image path.
                if self.given_detect_red_light_round == round_id:
                    self.osu_stubs.given_detect_red_light(red_light_image)
                novelty_lines, predictions = self.osu_stubs.process_round(test_id, round_id, image_paths, bbox_dict, hint_typeA_data, hint_typeB_data)

                filenames = []
                for novelty_image_idx, (novelty_line, cur_preds) in enumerate(zip(novelty_lines, predictions)):
                    novelty_image_path = image_paths[novelty_image_idx]
                    if isinstance(novelty_line, tuple):
                        (red_light_score, per_image_nov) = novelty_line
                    else:
                        red_light_score = 0
                        per_image_nov = novelty_line
                    filenames.append(novelty_image_path)
                    species_counts,\
                        species_presence,\
                        _,\
                        activity_presence = cur_preds

                    if self.given_detection:
                        if self.given_detect_red_light_round == round_id:
                            if novelty_image_path == red_light_image:
                                red_light_declared = True
                    else:
                        if red_light_score > 0.5:
                            red_light_declared = True
                    
                    
                        
                    self.write_file_entries(novelty_image_path, detection_file, classification_file,
                                            species_counts, species_presence, activity_presence, red_light_score, per_image_nov,
                                            round_id, red_light_declared)
            else:
                pass

            detection_file.close()
            classification_file.close()

            ### LAR useful trace
            # print(f'===> Submitting results with test_type: {self.history.test_type}')
            # import ipdb; ipdb.set_trace()
            if self.api_stubs:
                self.api_stubs.record_results(test_id, round_id,
                                         open(detection_filename, 'r').read(),
                                         open(classification_filename, 'r').read())
            else:
                requests.post(f"{self.url}/session/results", files={
                    'test_identification': io.StringIO(json.dumps({
                        'session_id': session_id,
                        'test_id': test_id,
                        'round_id': round_id,
                        'result_types': 'classification|detection'
                    })),
                    'detection_file': open(detection_filename, "r"),
                    'classification_file': open(classification_filename, "r")
                })
             ## Handle classification feedback
            if self.detection_fb:
                ## Handle detection feedback
                if (self.detection_fb or self.given_detection) and red_light_declared:
                    # Lance: It should be OK to always call choose_detection_feedback_ids
                    # if self.given_detection:
                    #     feedback_ids = filenames
                    # else:
                    #     feedback_ids = self.osu_stubs.choose_detection_feedback_ids(test_id, round_id,
                    #                                                           filenames, feedback_max_ids)

                    # GDD
                    # if self.given_detection:
                    #     print(f' **** About to call choose_detection_feedback_ids, round {round_id}')
                
                    # New round of querying. Add feedback_max_ids to total allowable
                    # queries, round down to compute the number to actually request,
                    # then subtract the number requested from the total allowable
                    # (supports alternating # of queries for fractional budgets)
                    trial_feedback_budget += feedback_max_ids
                    num_ids_to_request = int(trial_feedback_budget)
                    trial_feedback_budget -= num_ids_to_request

                    # Further override for given detection
                    if self.given_detection:
                        num_ids_to_request = len(filenames)
                    feedback_ids, feedback_bboxes = self.osu_stubs.choose_detection_feedback_ids(test_id, round_id,
                                                                                filenames, num_ids_to_request)

                    if len(feedback_ids) > 0:
                        feedback_csv_content = self.request_feedback(session_id, test_id, round_id,
                                                                                feedback_ids)

                    else:
                        feedback_csv_content = 'image_path,filename,width,height,agent1_name,agent1_id,agent1_count,agent2_name,agent2_id,agent2_count,agent3_name,agent3_id,agent3_count,activities,activities_id,environment,novelty_type,master_id\n'
                    self.osu_stubs.record_detection_feedback(test_id, round_id, feedback_csv_content, feedback_bboxes)

                ## LAR
                ## Write characterization files after every round for experimental testing.
                # characterization_filename = os.path.join(
                #     self.results_directory, "%s_%s_%s_characterization_round.csv" % (session_id, test_id, round_id))
                # compute_and_write_clusters(self.history, characterization_filename, max_novel_classes,
                #                            min(max_novel_classes, self.bbn_max_clusters),
                #                            self.bbn_min_novel_frac)
                ## LAR

            round_id += 1

        # if self.characterization:
        #     characterization_filename = os.path.join(
        #         self.results_directory, "%s_%s_%s_characterization.csv" % (session_id, test_id, round_id))
        #     compute_and_write_clusters(self.history, characterization_filename, max_novel_classes,
        #                                min(max_novel_classes, self.bbn_max_clusters),
        #                                self.bbn_min_novel_frac)
        #     response = requests.post(f"{self.url}/session/results", files={
        #         'test_identification': io.StringIO(json.dumps({
        #             'session_id': session_id,
        #             'test_id': test_id,
        #             'round_id': round_id,
        #             'result_types': 'characterization'
        #         })),
        #         'characterization_file': open(
        #             characterization_filename,
        #             "r")
        #     })
        #     print(response)
        #     print('==> sent characterization file')
        # np.savetxt(test_id + "HintsB.csv", np.array(hintsBList), delimiter=",")


        if self.api_stubs:
            self.api_stubs.finish_test(test_id)
            print('api_stubs.finish_test')
            self.osu_stubs.end_test(test_id)
        else:
            test_end_response = requests.delete(
                f"{self.url}/test?session_id={session_id}&test_id={test_id}")
            print(f'==> test end response: {test_end_response}')
        if self.osu_stubs:
            self.osu_stubs.end_test(test_id)

        ## LAR
        #print(f' ======> Missing box problem count: {self.missing_box_problems}')

