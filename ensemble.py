# import math
import os
import copy
import itertools
import torch
# import argparse
import torch.distributed as dist
# import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

import pocket
import numpy as np
# from tqdm import tqdm

from models.scg import SpatiallyConditionedGraph as SCG
from data.data_factory import DataFactory, CustomInput
from utils import custom_collate, get_config

# from models.idn import IDN
# import pickle
import pathlib


class Args:
    def __init__(self):
        self.multiporcessing = None
        self.world_size = None
        self.dataset = None
        self.data_root = None
        self.partitions = None
        self.csv_path = None
        self.num_subj_cls = None
        self.num_obj_cls = None
        self.num_action_cls = None
        self.num_workers = None
        self.batch_size = None
        self.num_classes = None
        self.max_object = None
        self.num_iter = None
        self.max_subject = None
        self.box_score_thresh = None
        self.num_iter = None
        self.top_k = None
        self.checkpoint_path = None
        self.ensemble_path = None


class Test(object):
    def __init__(self, nets, data_loader, top_k):
        if len(nets) < 2:
            raise Exception(f'Ensemble size has to be at least two.')

        self.nets = nets
        self.top_k = top_k
        self.data_loader = data_loader
        self.test = self.scg
        # self.converter = CustomInput(model_name).converter

    def scg(self):
        # method can be an element of set {math.product, sum}
        def select_topk(result, top_k, method=sum):
            obj_val, obj_ind = result['object_scores'].topk(top_k, dim=1)
            subj_val, subj_ind = result['subject_scores'].topk(top_k, dim=1)
            verb_val, verb_ind = result['verb_matrix'].topk(top_k, dim=2)
            object_combinations = itertools.product(range(len(result['object_boxes'])), range(top_k))
            subject_combinations = itertools.product(range(len(result['subject_boxes'])), range(top_k))
            verb_combinations = range(top_k)
            total_combs = list(itertools.product(subject_combinations, verb_combinations, object_combinations))
            probs = list(map(lambda x: (x, method([subj_val[x[0][0]][x[0][1]],
                                                   obj_val[x[2][0]][x[2][1]],
                                                   verb_val[x[0][0]][x[2][0]][x[1]]])), total_combs))
            probs = sorted(probs, key=lambda x: x[1], reverse=True)
            probs = probs[:top_k]
            # Getting the triplets for top-k predictions in the image
            top_k = list(map(lambda x: (x, (subj_ind[x[0][0][0]][x[0][0][1]],
                                            verb_ind[x[0][0][0]][x[0][2][0]][x[0][1]],
                                            obj_ind[x[0][2][0]][x[0][2][1]])), probs))
            result['top_k'] = top_k
            result['top_k_objects'] = obj_ind
            result['top_k_subjects'] = subj_ind
            return result

        def clean_result(my_net, orig_result, detections):
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
                'verb_matrix_logits': new_verb_matrix_logits,
                'object_logits': torch.index_select(orig_result['object_logits'], 0, torch.IntTensor(list(
                    (map(lambda x: np.where(res_to_det_obj == x)[0][0], range(len(detections['object_boxes']))))))),
                'subject_logits': torch.index_select(orig_result['subject_logits'], 0, torch.IntTensor(list(
                    (map(lambda x: np.where(res_to_det_subj == x)[0][0], range(len(detections['subject_boxes']))))))),
            }

            return new_result

        results = list()
        all_ensemble_results = []
        correct = 0.
        incorrect = 0.
        correct_obj = 0.
        incorrect_obj = 0.
        correct_subj = 0.
        incorrect_subj = 0.
        triplet_accuracy = dict()
        for i, batch in enumerate(self.data_loader):
            inputs = batch[:-1]
            img_id = inputs[1][0]['img_id']
            inputs[1][0].pop('img_id', None)

            inputs_copy = copy.deepcopy(inputs)
            input_data_copy = pocket.ops.relocate_to_cuda(inputs_copy)
            
            # This is needed to do box matching and remove the results where subjects 
            # have been made objects. This piece of logic might get moved inside the
            # model class in future releases 
            # mod_detections = pocket.ops.relocate_to_cpu(mod_detections)
            # _, mod_detections, _, _ = self.net.preprocess(*input_data_copy)  
            input_data = pocket.ops.relocate_to_cuda(inputs)

            ensemble_results = [None for _ in range(len(self.nets))]

            with torch.no_grad():
                for n, net in enumerate(self.nets):
                    _, mod_detections, _, _ = net.preprocess(*input_data_copy)  
                    mod_detections = pocket.ops.relocate_to_cpu(mod_detections)

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
                        'verb_logits': output['logits_verbs']
                    }
                    
                    result = clean_result(net, result, mod_detections[0])
                    result = select_topk(result, self.top_k)

                    results.append(result)
                    
                    # gt_triplet expects only one box as the input and will fail otherwise                
                    gt_triplet = [batch[-1][0]["subject"].item(), batch[-1][0]["verb"].item(), batch[-1][0]["object"].item()]
                    pred_triplets = [np.array(x[-1]).tolist() for x in result['top_k']]
                    gt_subj = gt_triplet[0]
                    gt_obj = gt_triplet[2]
                    pred_objs = result['top_k_objects'][0]
                    pred_subjs = result['top_k_subjects'][0]

                    d = dict()
                    d['gt'] = gt_triplet
                    d['pred'] = pred_triplets
                    d['verb_logits'] = result['verb_matrix_logits']
                    d['subject_logits'] = result['subject_logits']
                    d['object_logits'] = result['object_logits']
                    d['img'] = img_id
                    d['object_appearance_features'] = output['object_app_features']
                    d['pairwise_spatial_features'] = output['spatial_features']

                    ensemble_results[n] = d

                    # This creates the accuracy of all the triplets
                    if tuple(gt_triplet) not in triplet_accuracy:
                        triplet_accuracy[tuple(gt_triplet)] = {'correct' : 0, 'incorrect' : 0}
                    
                    if gt_triplet in pred_triplets:
                        correct += 1
                        triplet_accuracy[tuple(gt_triplet)]["correct"] += 1
                    else:
                        incorrect +=1
                        triplet_accuracy[tuple(gt_triplet)]["incorrect"] += 1
                        
                    if gt_obj in pred_objs:
                        correct_obj += 1
                    else:
                        incorrect_obj +=1
                    if gt_subj in pred_subjs:
                        correct_subj += 1
                    else:
                        incorrect_subj +=1

            all_ensemble_results.append(ensemble_results)
                
        print(f"Correct : {correct}, Incorrect :  {incorrect}, Total : {correct+incorrect}")
        print(f"Accuracy :  {correct / (correct + incorrect)}")
        print(f"Correct_obj : {correct_obj}, Incorrect :  {incorrect_obj}, Total : {correct_obj + incorrect_obj}")
        print(f"Accuracy_obj :  {correct_obj / (correct_obj + incorrect_obj)}")
        print(f"Correct_subj: {correct_subj}, Incorrect :  {incorrect_subj}, Total : {correct_subj + incorrect_subj}")
        print(f"Accuracy_subj :  {correct_subj / (correct_subj + incorrect_subj)}")
        print("Triplet level Accuracy")

        for key in triplet_accuracy:
            gt = triplet_accuracy[key]
            print(f"GT : {key}, Correct : {gt['correct']}, Incorrect : {gt['incorrect']}, Accuracy : {gt['correct'] / (gt['correct'] + gt['incorrect'])}")

        return all_ensemble_results, results
        

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"

    args = Args()

    args.world_size = 1
    args.dataset = 'Custom'
    args.data_root = 'Custom'
    args.csv_path = 'Custom/annotations/val_dataset_v1_val.csv'
    args.ensemble_path = 'ensemble'
    args.partitions = ['']
    args.num_iter = 2
    args.box_score_thresh = 0.0
    args.max_subject = 15
    args.max_object = 15
    args.num_subj_cls = 6
    args.num_obj_cls = 9
    args.num_action_cls = 8
    args.num_workers = 1
    args.batch_size = 1
    args.box_score_thresh = 0.0
    args.num_iter = 2
    args.top_k = 5

    rank = 1

    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    if args.multiporcessing:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=args.world_size,
            rank=rank
        )

    valset = DataFactory(
        name=args.dataset, partition=None,
        data_root=args.data_root,
        csv_path=args.csv_path,
        training=False,
        num_subj_cls=args.num_subj_cls,
        num_obj_cls=args.num_obj_cls,
        num_action_cls=args.num_action_cls
    )

    val_loader = DataLoader(
        dataset=valset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True,
        sampler=DistributedSampler(
            valset,
            num_replicas=args.world_size,
            rank=rank) if args.multiporcessing else None
    )

    if not os.path.exists(args.ensemble_path):
        raise Exception(f'pretrained models were not found in path {args.ensemble_path}')

    nets = []
    for p in pathlib.Path(args.ensemble_path).glob('*'):
        if p.suffix != '.pt':
            continue

        nets.append(SCG(
            num_classes=args.num_action_cls,
            num_obj_classes=args.num_obj_cls, num_subject_classes=args.num_subj_cls,
            num_iterations=args.num_iter, postprocess=False,
            max_subject=args.max_subject, max_object=args.max_object,
            box_score_thresh=args.box_score_thresh,
            distributed=args.multiporcessing))

        print(f"Loading pretrained model from {p}")
        checkpoint = torch.load(p, map_location="cpu")
        nets[-1].load_state_dict(checkpoint['model_state_dict'])
        nets[-1].cuda()
        nets[-1].eval()

    tester = Test(nets, val_loader, args.top_k).test
    tester()

    print("testing completed!")
