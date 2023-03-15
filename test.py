import math
import os
import copy
import itertools
import torch
import argparse
from torch.utils.data import DataLoader, DistributedSampler

import pocket
import numpy as np
# from tqdm import tqdm

from models.scg import SpatiallyConditionedGraph as SCG
from data.data_factory import DataFactory, CustomInput
from utils import gen_custom_collate


def get_net(args):
    net = SCG(
        num_classes=args.num_action_cls,
        num_obj_classes=args.num_obj_cls, num_subject_classes=args.num_subj_cls,
        num_iterations=args.num_iter, postprocess=False,
        max_subject=args.max_subject, max_object=args.max_object,
        box_score_thresh=args.box_score_thresh,
        distributed=args.multiporcessing
    )

    return net


class Test(object):
    def __init__(self, net, model_name, data_loader, top_k):
        self.net = net
        self.top_k = top_k
        self.data_loader = data_loader
        self.test = self.scg
        self.converter = CustomInput(model_name).converter

    def scg(self):
        # method can be an element of set {math.prod, sum}
        def select_topk(result, top_k, method=sum):
            obj_val, obj_ind = result['object_scores'].topk(top_k, dim=1)
            subj_val, subj_ind = result['subject_scores'].topk(top_k, dim=1)
            verb_val, verb_ind = result['verb_matrix'].topk(top_k, dim=2)

            # Resetting values at invalid object boxes based on method so that they do not impact the top_k computation
            # for verbs. We are operating under following assumptions
            #       1. Pairs with invalid object boxes will still be used. It's just that their classification scores
            #           should not be used for top-k verb prediction
            #       2. Pairs with invalid subject boxes will not be used. We will filter out all such tripletsval: {len(obj
            invalid_objects = list(set(range(len(obj_val))) - set(result['valid_objects'].tolist()))
            invalid_subjects = list(set(range(len(subj_val))) - set(result['valid_subjects'].tolist()))

            if method == sum:
                obj_val[invalid_objects] = 0.0
            elif method == math.prod:
                obj_val[invalid_objects] = 1.0
            else:
                raise NotImplementedError
            
            obj_ind[invalid_objects] = -1.0
            subj_ind[invalid_subjects] = -1.0
            verb_ind[invalid_subjects] = -1
            object_combinations = itertools.product(range(len(result['object_boxes'])), range(top_k))
            subject_combinations = itertools.product(range(len(result['subject_boxes'])), range(top_k))
            invalid_subject_combinations = itertools.product(invalid_subjects, range(top_k))
            verb_combinations = range(top_k)
            total_combs = list(itertools.product(subject_combinations, verb_combinations, object_combinations))
            invalid_combs = list(itertools.product(invalid_subject_combinations, verb_combinations,
                                                   object_combinations))
            total_combs = list(set(total_combs) - set(invalid_combs))
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
            result['top_k_verbs'] = verb_ind
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
            
            # getting the location in 2d matrix. Can comment not needed
            matrix_idx = torch.cat([new_pair_idx.unsqueeze(1), verbs.unsqueeze(1)], dim=1)
            verb_matrix[matrix_idx[:, 0], matrix_idx[:, 1]] = verb_scores

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
            matrix_idx = torch.cat([res_to_det_subj[new_pair_idx].unsqueeze(1),
                                    res_to_det_obj[new_pair_idx].unsqueeze(1),
                                    verbs.unsqueeze(1)], dim=1)
            new_verb_matrix[matrix_idx[:, 0], matrix_idx[:, 1], matrix_idx[:, 2]] = verb_scores
            
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

        def simplify_result(orig_result):
            """Simplifies the result for sending over the API.

            NOTE: This is intended to and will work only for the case where there is only 1 subject and object box each
            """
            triplet_tensor = torch.zeros((len(orig_result['subject_scores'][0]),
                                          len(orig_result['verb_matrix'][0][0]),
                                          len(orig_result['object_scores'][0])))
            top_k_triplets = list()
            for triplet in orig_result['top_k']:
                triplet_tensor[int(triplet[1][0])][int(triplet[1][1])][int(triplet[1][2])] = triplet[0][1]
                top_k_triplets.append((int(triplet[1][0]), int(triplet[1][1]), int(triplet[1][2])))
            
            new_result = {
                'object_scores': orig_result['object_scores'][0],
                'subject_scores': orig_result['subject_scores'][0],
                'img_id': orig_result['img_id'],
                'verb_scores': orig_result['verb_matrix'][0][0],
                'triplet_tensor': triplet_tensor,
                'top_k_triplets': top_k_triplets,
                'top_k_objects': orig_result['top_k_objects'][0],
                'top_k_subjects': orig_result['top_k_subjects'][0],
                'top_k_verbs': orig_result['top_k_verbs'][0],
            }

            return new_result

        results = list()
        correct = 0.
        incorrect = 0.
        correct_obj = 0.
        incorrect_obj = 0.
        correct_subj = 0.
        incorrect_subj = 0.
        correct_verb = 0.
        incorrect_verb = 0.
        triplet_accuracy = dict()
        for i, batch in enumerate(self.data_loader):
            inputs = batch[:-1]
            img_id = inputs[1][0]['img_id']
            inputs[1][0].pop('img_id', None)

            inputs_copy = copy.deepcopy(inputs)
            input_data_copy = pocket.ops.relocate_to_cuda(inputs_copy)
            
             # This is needed to do box matching and remove
            # the results where subjects have been made objects. This piece of logic might get moved inside the
            # model class in future releases
            _, mod_detections, _, _ = self.net.preprocess(*input_data_copy) 

            mod_detections = pocket.ops.relocate_to_cpu(mod_detections)
            input_data = pocket.ops.relocate_to_cuda(inputs)
            with torch.no_grad():
                output = self.net(*input_data)
                # Batch size is fixed as 1 for inference
                assert len(output) == 1, "Batch size is not 1"
                output = pocket.ops.relocate_to_cpu(output[0])
                result = {
                    'object_boxes': output['boxes_o'],
                    'subject_boxes': output['boxes_s'],
                    'object_scores': output['object_scores'],
                    'subject_scores': output['subject_scores'],
                    'index': output['index'],  # index of the box pair. Same
                    'verbs': output['prediction'],  # verbs are predicted considering the max score class on objects
                    #                                 and subjects
                    'verb_scores': output['scores'],
                    'valid_subjects': output['valid_subjects'],
                    'valid_objects': output['valid_objects'],
                    'img_id': img_id,
                }

                result = clean_result(self.net, result, mod_detections[0])
                result = select_topk(result, self.top_k)
                
                # gt_triplet expects only one box as the input and will fail otherwise                
                gt_triplet = [batch[-1][0]["subject"].item(), batch[-1][0]["verb"].item(), batch[-1][0]["object"].item()]
                pred_triplets = [np.array(x[-1]).tolist() for x in result['top_k']]
                gt_subj = gt_triplet[0]
                gt_verb = gt_triplet[1]
                gt_obj = gt_triplet[2]
                pred_objs = result['top_k_objects'][0]
                pred_subjs = result['top_k_subjects'][0]
                pred_verbs = result['top_k_verbs'][0]

                # This creates the accuracy of all the triplets
                if tuple(gt_triplet) not in triplet_accuracy:
                    triplet_accuracy[tuple(gt_triplet)] = {'correct': 0, 'incorrect': 0}

                if gt_triplet in pred_triplets:
                    correct += 1
                    triplet_accuracy[tuple(gt_triplet)]["correct"] += 1
                else:
                    incorrect += 1
                    triplet_accuracy[tuple(gt_triplet)]["incorrect"] += 1
                if gt_obj in pred_objs:
                    correct_obj += 1
                else:
                    incorrect_obj += 1
                if gt_subj in pred_subjs:
                    correct_subj += 1
                else:
                    incorrect_subj += 1
                if gt_verb in pred_verbs:
                    correct_verb += 1
                else:
                    incorrect_verb += 1

                result = simplify_result(result)
                results.append(result)
                
        print(f"Correct : {correct}, Incorrect :  {incorrect}, Total : {correct+incorrect}")
        print(f"Accuracy :  {correct / (correct + incorrect)}")
        print(f"Correct_obj : {correct_obj}, Incorrect :  {incorrect_obj}, Total : {correct_obj + incorrect_obj}")
        print(f"Accuracy_obj :  {correct_obj / (correct_obj + incorrect_obj)}")
        print(f"Correct_subj: {correct_subj}, Incorrect :  {incorrect_subj}, Total : {correct_subj + incorrect_subj}")
        print(f"Accuracy_subj :  {correct_subj / (correct_subj + incorrect_subj)}")
        print(f"Correct_verb: {correct_verb}, Incorrect :  {incorrect_verb}, Total : {correct_verb + incorrect_verb}")
        print(f"Accuracy_verb :  {correct_verb / (correct_verb + incorrect_verb)}")
        print("Triplet level Accuracy")

        for key in triplet_accuracy:
            gt = triplet_accuracy[key]
            print(f"GT : {key}, Correct : {gt['correct']}, Incorrect : {gt['incorrect']}, Accuracy : {gt['correct'] / (gt['correct'] + gt['incorrect'])}")

        return results        

def main(rank, args):
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False

    valset = DataFactory(
        name=args.dataset,
        data_root=args.data_root,
        csv_path=args.csv_path,
        training=False,
        num_subj_cls=args.num_subj_cls,
        num_obj_cls=args.num_obj_cls,
        num_action_cls=args.num_action_cls
    )

    val_loader = DataLoader(
        dataset=valset,
        collate_fn=gen_custom_collate(), batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True,
        sampler=DistributedSampler(
            valset,
            num_replicas=args.world_size,
            rank=rank) if args.multiporcessing else None
    )

    net = get_net(args)

    if os.path.exists(args.checkpoint_path):
        print("Loading model from ", args.checkpoint_path)
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        net.load_state_dict(checkpoint['model_state_dict'])

    elif len(args.checkpoint_path):
        print("\nWARNING: The given model path does not exist. "
              "Proceed to use a randomly initialised model.\n")

    net.cuda()
    net.eval()
    tester = Test(net, args.net, val_loader, args.top_k).test
    tester()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an interaction head")
    parser.add_argument('--world-size', default=1, type=int,
                        help="Number of subprocesses/GPUs to use")
    parser.add_argument('--dataset', default='Custom', type=str)
    parser.add_argument('--net', default='scg', type=str)
    parser.add_argument('--top-k', default=3, type=int)
    parser.add_argument('--csv-path', default=None, type=str, help="Csv Path is required only for Custom dataset")
    parser.add_argument('--data-root', default='Custom', type=str)
    parser.add_argument('--num-iter', default=2, type=int,
                        help="Number of iterations to run message passing")
    parser.add_argument('--box-score-thresh', default=0.0, type=float)
    parser.add_argument('--num-subj-cls', default=6, type=int)
    parser.add_argument('--num-obj-cls', default=9, type=int)
    parser.add_argument('--num-action-cls', default=8, type=int)
    parser.add_argument('--max-subject', default=15, type=int)
    parser.add_argument('--max-object', default=15, type=int)
    parser.add_argument('--num-workers', default=1, type=int)
    parser.add_argument('--checkpoint-path', default='', type=str)
    parser.add_argument('--batch-size', default=1, type=int,
                        help="Batch size for each subprocess")

    args = parser.parse_args()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"
    
    main(1, args)
    print("testing complete!")
