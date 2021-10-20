import copy
import torch
import pocket
import numpy as np
import pathlib
import pickle
from ensemble.cal_model import LogisticRegression


class Calibrator:
    def __init__(self, nets, num_obj_classes, num_subj_classes, num_verb_classes):
        if len(nets) < 2:
            raise Exception(f'Ensemble size has to be at least two.')

        self.nets = nets
        self.num_obj_classes = num_obj_classes
        self.num_subj_classes = num_subj_classes
        self.num_verb_classes = num_verb_classes
        self.calibrator_path = './ensemble/lr.pt'
        self.train_tuples_path = './ensemble/unique.pkl'

        if not pathlib.Path(self.train_tuples_path).exists():
            raise Exception(f'train tuples were not found in path {self.train_tuples_path}. Please run "find_unique_train_tuples.py" first')

        with open(self.train_tuples_path, 'rb') as handle:
            results = pickle.load(handle)

            self.train_tuples = results['tuples']
            self.train_subjects = results['subjects']
            self.train_objects = results['objects']
            self.train_verbs = results['verbs']
            self.num_unique_triplets = len(self.train_tuples)

    def _clean_result(self, my_net, orig_result, detections):
        num_verb_cls = my_net.interaction_head.num_classes

        # Now getting the indices of object boxes which were not passed as subjects in input
        keep_obj_idx = np.argwhere(
            np.fromiter(
                map(lambda x: not (any([(x == par_elem).all() for par_elem in detections['subject_boxes']])),
                    [elem for elem in orig_result['object_boxes']]), dtype=np.bool_))
        
        keep_obj_idx = torch.from_numpy(keep_obj_idx).squeeze(1)
        
        # Initialising the verb matrix with zero values
        verb_matrix = torch.zeros((len(keep_obj_idx), num_verb_cls))
        verb_matrix_logits = torch.zeros((len(keep_obj_idx), num_verb_cls))
        
        # Filtering the pairs based on these indices
        keep_pair_idx = np.argwhere(
            np.fromiter(map(lambda x: x in keep_obj_idx,
                            [elem for elem in orig_result['index']]), dtype=np.bool_))
        
        keep_pair_idx = torch.from_numpy(keep_pair_idx).squeeze(1)
        orig_pair_idx = torch.index_select(orig_result['index'], 0, keep_pair_idx)
        
        # Getting the new pair indexes for selected pairs
        new_pair_idx = np.searchsorted(keep_obj_idx, orig_pair_idx)
        
        # Getting the verb prediction only on selected pairs
        verbs = torch.index_select(orig_result['verbs'], 0, keep_pair_idx)
        verb_scores = torch.index_select(orig_result['verb_scores'], 0, keep_pair_idx)
        verb_logits = torch.index_select(orig_result['verb_logits'], 0, keep_pair_idx)
        
        # getting the location in 2d matrix. Can comment not needed
        matrix_idx = torch.cat([new_pair_idx.unsqueeze(1), verbs.unsqueeze(1)], dim=1)
        verb_matrix[matrix_idx[:, 0], matrix_idx[:, 1]] = verb_scores
        verb_matrix_logits[matrix_idx[:, 0], matrix_idx[:, 1]] = verb_logits

        # Now re-mapping this 2d matrix (sub-obj pair X verb)  to a 3d matrix (subj X obj X verb)
        kept_result_objs = orig_result['object_boxes'][keep_obj_idx]
        kept_result_subjs = orig_result['subject_boxes'][keep_obj_idx]

        res_to_det_obj = torch.from_numpy(np.asarray(list(
            map(lambda x: np.argwhere([(x == par_elem).all() for par_elem in detections['object_boxes']])[0][0],
                [elem for elem in kept_result_objs]))))

        # Duplicate boxes are not acceptable in either subject or object list. There can be same box in both the
        # lists though
        res_to_det_subj = torch.from_numpy(np.asarray(list(
            map(lambda x: np.argwhere([(x == par_elem).all() for par_elem in detections['subject_boxes']])[0][0],
                [elem for elem in kept_result_subjs]))))

        new_verb_matrix = torch.zeros(
            (len(detections['subject_boxes']), len(detections['object_boxes']), num_verb_cls))
        new_verb_matrix_logits = torch.zeros(
            (len(detections['subject_boxes']), len(detections['object_boxes']), num_verb_cls))

        matrix_idx = torch.cat([res_to_det_subj[new_pair_idx].unsqueeze(1),
                                res_to_det_obj[new_pair_idx].unsqueeze(1),
                                verbs.unsqueeze(1)], dim=1)

        new_verb_matrix[matrix_idx[:, 0], matrix_idx[:, 1], matrix_idx[:, 2]] = verb_scores
        new_verb_matrix_logits[matrix_idx[:, 0], matrix_idx[:, 1], matrix_idx[:, 2]] = verb_logits
                
        new_result = {
            'object_boxes': detections['object_boxes'],
            'subject_boxes': detections['subject_boxes'],
            'object_scores': torch.index_select(orig_result['object_scores'], 0, torch.IntTensor(list(
                (map(lambda x: np.where(res_to_det_obj == x)[0][0], range(len(detections['object_boxes']))))))),
            'subject_scores': torch.index_select(orig_result['subject_scores'], 0, torch.IntTensor(list(
                (map(lambda x: np.where(res_to_det_subj == x)[0][0], range(len(detections['subject_boxes']))))))),
            'img_id': orig_result['img_id'],
            'verb_matrix': new_verb_matrix,
            'valid_subjects': orig_result['valid_subjects'],
            'valid_objects': orig_result['valid_objects'],
            'verb_matrix_logits': new_verb_matrix_logits,
            'object_logits': torch.index_select(orig_result['object_logits'], 0, torch.IntTensor(list(
                (map(lambda x: np.where(res_to_det_obj == x)[0][0], range(len(detections['object_boxes']))))))),
            'subject_logits': torch.index_select(orig_result['subject_logits'], 0, torch.IntTensor(list(
                (map(lambda x: np.where(res_to_det_subj == x)[0][0], range(len(detections['subject_boxes']))))))),
        }
        
        # Now cleaning results based on validity of boxes
        for id in range(len(new_result['subject_boxes'])):
            if id not in orig_result['valid_subjects']:
                # NOTE: We don't care if we are returning the exact supplied invalid boxes. We will, by default,
                # send all co-ordinates as -1 in this case
                new_result['subject_boxes'][id] = -1
                new_result['subject_scores'][id] = -1
                new_result['verb_matrix'][id] = -1
        
        for id in range(len(new_result['object_boxes'])):
            if id not in orig_result['valid_objects']:
                # NOTE: We don't care if we are returning the exact supplied invalid boxes. We will, by default,
                # send all co-ordinates as -1 in this case
                new_result['object_boxes'][id] = -1
                new_result['object_scores'][id] = -1
        
        return new_result

    def _compute_features(self, data_loader):
        features = []
        labels = []

        for _, batch in enumerate(data_loader):
            inputs = batch[:-1]
            img_id = inputs[1][0]['img_id']
            inputs[1][0].pop('img_id', None)

            # inputs_copy = copy.deepcopy(inputs)
            # input_data_copy = pocket.ops.relocate_to_cuda(inputs_copy)

            # _, mod_detections, _, _ = self.nets[0].preprocess(*input_data_copy)  
            # mod_detections = pocket.ops.relocate_to_cpu(mod_detections)

            # gt_triplet expects only one box as the input and will fail otherwise        
            gt_triplet = [batch[-1][0]["subject"].item(), batch[-1][0]["verb"].item(), batch[-1][0]["object"].item()]
            gt_subj = gt_triplet[0]
            gt_obj = gt_triplet[2]
            labels.append(tuple(gt_triplet))

            ensemble_results = [None for _ in range(len(self.nets))]
            ensemble_feature = torch.zeros(self.num_subj_classes + self.num_obj_classes + self.num_verb_classes)
        
            with torch.no_grad():
                for n, net in enumerate(self.nets):
                    inputs_copy = copy.deepcopy(inputs)
                    input_data_copy = pocket.ops.relocate_to_cuda(inputs_copy)
                    _, mod_detections, _, _ = net.preprocess(*input_data_copy) 

                    mod_detections = pocket.ops.relocate_to_cpu(mod_detections)
                    input_data = pocket.ops.relocate_to_cuda(inputs)
                    output = net(*input_data)
                    
                    # Batch size is fixed as 1 for inference
                    assert len(output) == 1, "Batch size is not 1"
                    
                    output = pocket.ops.relocate_to_cpu(output[0])

                    result = {
                        'object_boxes': output['boxes_o'],
                        'subject_boxes': output['boxes_s'],
                        'object_scores': output['object_scores'],
                        'subject_scores': output['subject_scores'],
                        'index': output['index'],  # index of the box pair. Same
                        'verbs': output['prediction'],  # verbs are predicted considering the max score class on objects and subjects
                        'verb_scores': output['scores'],
                        'img_id': img_id,
                        'object_logits': output['logits_object'],
                        'subject_logits': output['logits_subject'],
                        'verb_logits': output['logits_verbs'],
                        'valid_subjects': output['valid_subjects'],
                        'valid_objects': output['valid_objects'],
                    }
                    
                    result = self._clean_result(net, result, mod_detections[0])
                    ensemble_results[n] = result

                    feature = torch.hstack([result["subject_logits"].view(-1), 
                                            result["object_logits"].view(-1), 
                                            result["verb_matrix_logits"].view(-1)])
                    ensemble_feature += feature
                    
                features.append(ensemble_feature)

        return features, labels

    def train(self, data_loader):
        features, labels = self._compute_features(data_loader)

        num_cal_samples = len(features)
        features = [torch.Tensor(f) for f in features]

        X_cal = torch.vstack(features)
        y_cal = torch.zeros(num_cal_samples).long()
        # preds_cal = torch.zeros(num_cal_samples)

        for i in range(num_cal_samples):
            try:
                idx = self.train_tuples.index(labels[i])
                y_cal[i] = idx
            except ValueError:
                print(f'Error: label {labels[i]} not found in unique triplets')
                raise Exception('Fatal error')

        lr = LogisticRegression(input_dim=X_cal.shape[1], output_dim=self.num_unique_triplets)
        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(lr.parameters(), lr=6e-2)
        
        for _ in range(15):
            out = lr(X_cal)

            optimizer.zero_grad()
            loss = loss_func(out, y_cal)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                correct = torch.sum(torch.argmax(torch.nn.functional.softmax(out, dim=1), dim=1) == y_cal)
                print(f'loss: {loss.item()}, acc: {correct.item() / num_cal_samples}')

        torch.save(
            {
                'lr': lr.state_dict()
            }, self.calibrator_path)

        print(f'trained calibrator saved to {self.calibrator_path}')
