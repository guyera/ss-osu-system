import ast
import io
import json
import os
from argparse import ArgumentParser
import requests
from seaborn.distributions import ecdfplot
from session import api_stubs
import itertools
import numpy as np


class BBNSession:
    def __init__(self, protocol, domain, class_count, class_fb, detection_fb, given_detection,
                 image_directory, results_directory, api_url,
                 batch_size, version, detection_threshold,
                 api_stubs_flag, osu_interface):

        # self.agent = Agent(config['detectormodelpath'],
        #                    config['classifiermodelpath'],
        #                    config['redlightdistributionpath'],
        #                    config['redlightscalefactor'],
        #                    config['redlightthreshold'])
        self.version = version
        self.protocol = protocol
        self.domain = domain
        self.class_count = class_count
        self.class_fb = class_fb
        self.detection_fb = detection_fb
        self.given_detection = given_detection
        self.results_directory = results_directory
        self.url = api_url
        self.version = version
        self.detection_threshold = detection_threshold
        self.image_directory = image_directory
        self.results_directory = results_directory
        # self.characterization = characterization
        # self.bbn_max_clusters = config['bbnmaxclusters']
        # self.bbn_min_novel_frac = config["bbnminnovelfrac"]
        # self.red_light_threshold = config['redlightthreshold']
        # self.test_count = test_count
        self.batch_size = batch_size
        self.history = None  # TestHistory for current test
        self.api_stubs = api_stubs_flag
        self.osu_stubs = osu_interface
        # These are also defined in toplevel
        self.num_subject_classes = 5
        self.num_object_classes = 13
        self.num_verb_classes = 8

        # Turn this to True to get readable (S,O,V) triples for our top-3 in classification output.
        # Default is to get the 945 floats that UMD is requesting. 
        self.probs_debug_format = False
        self.triple_list = list(itertools.product(
            np.arange(-1, self.num_subject_classes + 1),
            np.arange(0, self.num_verb_classes + 1),
            np.arange(-1, self.num_object_classes + 1)))
        self.triple_list_count = len(self.triple_list)
        self.triple_dict = {}
        for i, triple in enumerate(self.triple_list):
            self.triple_dict[triple] = i

        if not os.path.exists(self.results_directory):
            os.makedirs(self.results_directory)

        if self.api_stubs:
            api_stubs.clear_results_dirs()


    def request_class_feedback(self, session_id, test_id, round_id, feedback_ids):
        print(f'====> Requesting class feedback on round {round_id} for {len(feedback_ids)} images.')
        # for feedback_item in feedback_ids:
        # print(f'  ====> {feedback_ids}')
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

        feedback_vals = response.content.decode('utf-8').split('\n')
        feedback_vals = [x for x in feedback_vals if x.strip("\n\t\"',.") != ""]
        # feedback_answers = [int(val.split(',')[1]) for val in feedback_vals]
        returned_ids = []
        returned_answers = []

        for val in feedback_vals:
            segments = val.split(',')
            returned_ids.append(segments[0])
            returned_answers.append(segments[1:])
        
        if set(returned_ids) != set(feedback_ids):
            raise Exception('feedback returned ids not matching requested')

        return {id: val for (id, val) in zip(returned_ids, returned_answers)}

    def request_detection_feedback(self, session_id, test_id, round_id, feedback_ids):
        # print(f'====> Requesting detection feedback on round {round_id} for {len(feedback_ids)} images.')
        if api_stubs:
            return api_stubs.detection_feedback(test_id, round_id, feedback_ids)
        else:
            response = requests.get(
                f'{self.url}/session/feedback',
                {'session_id': session_id,
                 'test_id': test_id,
                 'round_id': round_id,
                 'feedback_type': 'detection',
                 'feedback_ids': '|'.join(feedback_ids),
                 }
            )

            if response.status_code == 500:
                raise Exception(f'Got error response from feedback request: {response}')
            
            feedback_vals = response.content.decode('utf-8').split('\n')
            feedback_vals = [x for x in feedback_vals if x.strip("\n\t\"',.") != ""]
            # feedback_answers = [int(val.split(',')[1]) for val in feedback_vals]
            returned_ids = []
            returned_answers = []
            for val in feedback_vals:
                segments = val.split(',')
                returned_ids.append(segments[0])
                returned_answers.append(segments[1] == '1')

            if set(returned_ids) != set(feedback_ids):
                raise Exception('feedback returned ids not matching requested')
            
            return [(id, val) for (id, val) in zip(returned_ids, returned_answers)]

    def force_to_missing(self, s, v, o, missing_s, missing_o, free_s, free_v, free_o):
        """Force our answer to be consistent with missing_s and missing_o"""
        ## Use the free_s, _v, _o vars to ensure that our three new forced answers are distinct.
        if missing_s:
            # S and V must be missing
            s = -1
            v = 0
            # O must not be missing
            if o == -1:
                o = list(free_o)[0]
                free_o.remove(o)
        elif missing_o:
            # O must be missing
            o = -1
            # S must not be missing
            if s == -1:
                s = list(free_s)[0]
                free_s.remove(s)
        else:
            # neither S nor O can be missing
            if s == -1:
                s = list(free_s)[0]
                free_s.remove(s)
            if o == -1:
                o = list(free_o)[0]
                free_o.remove(o)
        return s, v, o, free_s, free_v, free_o
        

    def write_file_entries(self, filename, detection_file, classification_file, predicted_probs,
                           # red_light,
                           red_light_score, image_novelty_score,
                           round_id, missing_s, missing_o):  ##, top_layer):
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
            # TODO: pass missing_s and missing_o flags into this function
            # and use them here to clean up bad values
            output_vals = np.zeros(self.triple_list_count)
            answers = ast.literal_eval(f'({predicted_probs})')
            assert len(answers) == 3
            free_s = set(range(1, self.num_subject_classes+1))
            free_v = set(range(1, self.num_verb_classes+1))
            free_o = set(range(1, self.num_object_classes+1))
            for answer in answers:
                s, v, o, prob = answer
                free_s.discard(s)
                free_v.discard(v)
                free_o.discard(o)
            output_triples = []
            for answer in answers:
                s, v, o, prob = answer
                # for V, id 0 covers both missing and novel
                if v == -1:
                    v = 0
                s, v, o, free_s, free_v, free_o = self.force_to_missing(s, v, o, missing_s, missing_o,
                                                                        free_s, free_v, free_o)
                triple = (s, v, o)
                loop_count = 0
                while triple in output_triples:
                    # change one value randomly to avoid duplication
                    loop_count += 1
                    if loop_count > 3:
                        raise Exception('Excessive looping to avoid duplicates')
                    if s != -1:
                        s = list(free_s)[0]
                        free_s.remove(s)
                    elif o != -1:
                        o = list(free_o)[0]
                        free_o.remove(o)
                    else:
                        raise Exception('One of S or O should not be missing.')
                    triple = (s, v, o)
                output_triples.append(triple)
                output_vals[self.triple_dict[triple]] = prob
            output_probs = ','.join([f'{val:e}' for val in output_vals])            
        
        detection_file.write("%s,%e,%e\n" %
                             (filename, red_light_score, image_novelty_score))
        classification_file.write("%s,%s\n" %
                                  (filename, output_probs))

    def check_for_missing(self, image_line):
        fields = image_line.split(',')
        missing_s = fields[4] == '-1'
        missing_o = fields[8] == '-1'
        return missing_s, missing_o

    def run(self, detector_seed, test_ids=None, given_detection=False):
        if not test_ids:
            # test_ids = requests.get(
            #     f"{self.url}/test/ids?protocol={self.protocol}&detector_seed={detector_seed}").content.decode('utf-8').split('\n')
            ##result = requests.get(f"{self.url}/test/ids?protocol={self.protocol}&detector_seed={detector_seed}")
            ### To test if UMD eval would work.
            ### self.domain = 'image_classification'
            if self.api_stubs:
                test_ids = api_stubs.test_ids()
            else:
                result = requests.get(
                    f"{self.url}/test/ids?protocol={self.protocol}&detector_seed={detector_seed}&domain={self.domain}")

                test_ids = result.content.decode('utf-8').split('\n')

        # Need to remove possible final empty strings
        test_ids = [test_id for test_id in test_ids if test_id.strip("\n\t\"',.") != ""]

        self.hints = ['novelty_instance_detection']
        if self.api_stubs:
            session_id = api_stubs.session_id()
        else:
            response = requests.post(f"{self.url}/session", json={
                'configuration': {
                    'protocol': self.protocol,
                    'novelty_detector_version': self.version,
                    'test_ids': test_ids,
                    'domain': self.domain,
                    'hints': self.hints,
                    'detection_threshold': self.detection_threshold,
                    # 'hints': ['red_light' if given_detection else '']
                }
            })
            session_id = ast.literal_eval(response.content.decode('utf-8'))['session_id']

        if self.osu_stubs:
            self.osu_stubs.start_session(session_id, detection_feedback=True, classification_feedback=False, given_detection=False)

        print(f"=> initialized session: {session_id}")

        for test_id in test_ids:
            print(f"==> starting test: {test_id}")
            self.run_test(session_id, test_id, given_detection)
            # self.agent.reset()

        print("=> terminating session")

        if self.api_stubs:
            pass
        else:
            requests.delete(f"{self.url}/session?session_id={session_id}")
        if self.osu_stubs:
            self.osu_stubs.end_session(session_id)

    def run_test(self, session_id, test_id, given_detection):

        # self.history = TestHistory()

        if self.api_stubs:
            metadata = api_stubs.metadata()
        else:
            metadata = ast.literal_eval(requests.get(
                f"{self.url}/test/metadata?session_id={session_id}&test_id={test_id}")
                                        .content.decode('utf-8'))

        if self.osu_stubs:
            self.osu_stubs.start_test(test_id)

        round_size = metadata['round_size']
        feedback_max_ids = metadata['feedback_max_ids']
        max_novel_classes = metadata['max_novel_classes']
        # print(f'===> Max_novel_classes: {max_novel_classes}')
        red_light_declared = False

        round_id = 0

        while True:
            image_data = None
            filenames = None

            if self.api_stubs:
                image_data = api_stubs.image_data(test_id, round_id)

                if image_data == None:
                    print("==> No more rounds")
                    break

            else:
                filenames_response = requests.get(
                    f"{self.url}/session/dataset?session_id={session_id}&test_id={test_id}&round_id={round_id}")

                if filenames_response.status_code == 204:
                    print("==> No more rounds")
                    break

                print(f"===> Round {round_id}")

                filenames = filenames_response.content.decode('utf-8').split('\n')
                filenames = [x for x in filenames if x.strip("\n\t\"',.") != ""]

                ## Lance
                response_list = ast.literal_eval(filenames_response.content.decode('utf-8'))
                image_lines = []
                for image_data in response_list:
                    image_path = image_data[0]
                    bbox_info = ','.join(image_data[1])
                    image_line = ','.join([image_path,'0', '0', bbox_info])
                    image_lines.append(image_line)
                image_data = '\n'.join(image_lines)
                ## Lance

            detection_filename = os.path.join(
                self.results_directory, "%s_%s_%s_detection.csv" % (session_id, test_id, round_id))
            detection_file = open(detection_filename, "w")
            if round_id == 0:
                detection_file.write('image_path,red_light_prob,per_image_prob\n')
            classification_filename = os.path.join(
                self.results_directory, "%s_%s_%s_classification.csv" % (session_id, test_id, round_id))
            classification_file = open(classification_filename, "w")
            if round_id == 0:
                triple_labels = [f'{s}_{v}_{o}' for (s, v, o) in self.triple_list]
                classification_file.write(f'image_path,{",".join(triple_labels)}\n')

            if self.osu_stubs:
                filenames = []
                novelty_preds, svo_preds = self.osu_stubs.process_round(test_id, round_id, image_data)
                novelty_lines = novelty_preds.splitlines()
                svo_lines = svo_preds.splitlines()
                
                for (novelty_line, svo_line, image_line) in zip(novelty_lines, svo_lines, image_lines):
                    (novelty_image_path, red_light_str, per_image_nov_str) = novelty_line.split(',')
                    (svo_image_path, svo_preds) = svo_line.split(',', 1)
                    assert novelty_image_path == svo_image_path
                    filenames.append(novelty_image_path)

                    if float(red_light_str) > 0.5:
                        red_light_declared = True
                        
                    missing_s, missing_o = self.check_for_missing(image_line)
                    self.write_file_entries(novelty_image_path, detection_file, classification_file,
                                            svo_preds, float(red_light_str), float(per_image_nov_str),
                                            round_id, missing_s, missing_o)
            else:
                pass
                # In Phase 1, we processed images through our network in batches
                # batches = [filenames[ndx:min(ndx + self.batch_size, images_in_round)] for ndx in
                #            range(0, images_in_round, self.batch_size)]
                #
                # for batch in batches:
                #     # predicted_probss, _red_lights, red_light_scores, image_novelty_scores, top_layers = self.agent([os.path.join(self.image_directory, filename) for filename in batch])
                #     for i, filename in enumerate(batch):
                #         # predicted_probs, _red_light, red_light_score, image_novelty_score, top_layer = predicted_probss[i], _red_lights[i], red_light_scores[i], image_novelty_scores[i], top_layers[i]
                #         red_light_score = osu_stubs.red_light_score(round_id, i)
                #         predicted_probs = osu_stubs.predicted_probs(self.class_count)
                #         # We supplied image novelty scores in Phase 1, but it's not clear they'll be needed in Phase 1
                #         image_novelty_score = osu_stubs.image_novelty_score()
                #
                #         # if given_detection and 'red_light' in metadata:
                #         #     if not self.agent.red_light and filename == metadata['red_light']:
                #         #         self.agent.red_light = True
                #         #     red_light_score = 1.0 if self.agent.red_light else 0.0
                #         # else:
                #         #     self.agent.red_light = _red_light
                #
                #         self.write_file_entries(
                #             filename, detection_file, classification_file, predicted_probs,
                #             # self.agent.red_light,
                #             red_light_score, image_novelty_score, round_id)  ##, top_layer)

            detection_file.close()
            classification_file.close()

            ### LAR useful trace
            # print(f'===> Submitting results with test_type: {self.history.test_type}')

            if self.api_stubs:
                api_stubs.record_results(test_id, round_id,
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
            if self.class_fb and red_light_declared:
                feedback_ids = self.osu_stubs.choose_classification_feedback_ids(filenames, feedback_max_ids)
                class_feedback_results = self.request_class_feedback(session_id, test_id, round_id, feedback_ids)
                # self.record_feedback_stub(feedback_results)
                # self.history.record_feedback(known_count, novel_count)

            ## Handle detection feedback
            if (self.detection_fb or self.given_detection) and red_light_declared:
                if self.given_detection:
                    feedback_ids = filenames
                else:
                    feedback_ids = self.osu_stubs.choose_detection_feedback_ids(test_id, round_id,
                                                                           filenames, feedback_max_ids)
                detection_feedback_results = self.request_detection_feedback(session_id, test_id, round_id,
                                                                             feedback_ids)
                self.osu_stubs.record_detection_feedback(test_id, round_id, detection_feedback_results)

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

        if self.api_stubs:
            api_stubs.finish_test(test_id)
            print('api_stubs.finish_test')
        else:
            test_end_response = requests.delete(
                f"{self.url}/test?session_id={session_id}&test_id={test_id}")
            # print(f'==> test end response: {test_end_response}')
        if self.osu_stubs:
            self.osu_stubs.end_test(test_id)
