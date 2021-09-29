import math
import os
import copy
import itertools
import torch
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

import pocket
import numpy as np
from tqdm import tqdm

from models.scg import SpatiallyConditionedGraph as SCG
from data.data_factory import DataFactory, CustomInput
from utils import custom_collate, get_config

from models.idn import IDN
import pickle


def get_net(args):
    if args.net == 'scg':
        net = SCG(
            num_classes=args.num_action_cls,
            num_obj_classes=args.num_obj_cls, num_subject_classes=args.num_subj_cls,
            num_iterations=args.num_iter, postprocess=False,
            max_subject=args.max_subject, max_object=args.max_object,
            box_score_thresh=args.box_score_thresh,
            distributed=args.multiporcessing
        )
    elif args.net == 'idn':
        args_idn = pickle.load(open('configs/arguments.pkl', 'rb'))
        HO_weight = torch.from_numpy(args_idn['HO_weight'])
        config = get_config(args.config_path)
        net = IDN(config.MODEL, HO_weight, num_classes=args.num_classes)
    else:
        net = ''

    if net == '':
        raise NotImplementedError
    return net


class Test(object):
    def __init__(self, net, model_name, data_loader, top_k):
        self.net = net
        self.top_k = top_k
        self.data_loader = data_loader
        self.func_map = {
            'scg': self.scg,
            'drg': self.drg,
            'idn': self.idn,
            'cascaded-hoi': self.cascaded_hoi,
        }
        self.test = self.func_map[model_name]
        self.converter = CustomInput(model_name).converter

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
            }

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
            }
            return new_result

        results = list()
        correct = 0.
        incorrect = 0.
        correct_obj = 0.
        incorrect_obj = 0.
        correct_subj = 0.
        incorrect_subj = 0.
        triplet_accuracy = dict()
        for i, batch in tqdm(enumerate(self.data_loader)):
            inputs = batch[:-1]
            img_id = inputs[1][0]['img_id']
            inputs[1][0].pop('img_id', None)

            inputs_copy = copy.deepcopy(inputs)
            input_data_copy = pocket.ops.relocate_to_cuda(inputs_copy)
            _, mod_detections, _, _ = self.net.preprocess(
                *input_data_copy)  # This is needed to do box matching and remove
            # the results where subjects have been made objects. This piece of logic might get moved inside the
            # model class in future releases
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
                    'img_id': img_id,
                }
                
                result = clean_result(self.net, result, mod_detections[0])
                result = select_topk(result, self.top_k)
                
                # gt_triplet expects only one box as the input and will fail otherwise                
                gt_triplet = [batch[-1][0]["subject"].item(), batch[-1][0]["verb"].item(), batch[-1][0]["object"].item()]
                pred_triplets = [np.array(x[-1]).tolist() for x in result['top_k']]
                gt_subj = gt_triplet[0]
                gt_obj = gt_triplet[2]
                pred_objs = result['top_k_objects'][0]
                pred_subjs = result['top_k_subjects'][0]
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
                result = simplify_result(result)
                results.append(result)
                
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

        return results

    def drg(self, input_data):
        raise NotImplementedError

    def idn(self):
        raise NotImplementedError
        # timer = Timer()
        # bboxes, scores, scores_AE, scores_rev, keys, hdet, odet = [], [], [], [], [], [], []
        # for i in range(80):
        #     bboxes.append([])
        #     scores.append([])
        #     scores_AE.append([])
        #     scores_rev.append([])
        #     keys.append([])
        #     hdet.append([])
        #     odet.append([])
        # args = pickle.load(open('configs/arguments.pkl', 'rb'))
        # verb_mapping = torch.from_numpy(pickle.load(open('configs/verb_mapping.pkl', 'rb'), encoding='latin1')).float()
        # HO_weight = torch.from_numpy(args['HO_weight'])
        # fac_i = args['fac_i']
        # fac_a = args['fac_a']
        # fac_d = args['fac_d']
        # nis_thresh = args['nis_thresh']
        # obj_range = pickle.load(open('configs/idn_configs.pkl', 'rb'), encoding='latin1')['obj_range']
        #
        # timer.tic()
        # for i, batch in enumerate(self.data_loader):
        #     n = batch['shape'].shape[0]
        #     batch['shape'] = batch['shape'].cuda(non_blocking=True)
        #     batch['spatial'] = batch['spatial'].cuda(non_blocking=True)
        #     batch['sub_vec'] = batch['sub_vec'].cuda(non_blocking=True)
        #     batch['obj_vec'] = batch['obj_vec'].cuda(non_blocking=True)
        #     batch['uni_vec'] = batch['uni_vec'].cuda(non_blocking=True)
        #     batch['labels_s'] = batch['labels_s'].cuda(non_blocking=True)
        #     batch['labels_ro'] = batch['labels_ro'].cuda(non_blocking=True)
        #     batch['labels_r'] = batch['labels_r'].cuda(non_blocking=True)
        #     batch['labels_sro'] = batch['labels_sro'].cuda(non_blocking=True)
        #     verb_mapping = verb_mapping.cuda(non_blocking=True)
        #     output = self.net(batch)
        #
        #     batch['spatial'][:, 0] *= batch['shape'][:, 0]
        #     batch['spatial'][:, 1] *= batch['shape'][:, 1]
        #     batch['spatial'][:, 2] *= batch['shape'][:, 0]
        #     batch['spatial'][:, 3] *= batch['shape'][:, 1]
        #     batch['spatial'][:, 4] *= batch['shape'][:, 0]
        #     batch['spatial'][:, 5] *= batch['shape'][:, 1]
        #     batch['spatial'][:, 6] *= batch['shape'][:, 0]
        #     batch['spatial'][:, 7] *= batch['shape'][:, 1]
        #     obj_class = batch['obj_class']
        #     bbox = batch['spatial'].detach().cpu().numpy()
        #
        #     if 's' in output:
        #         output['s'] = torch.matmul(output['s'], verb_mapping)
        #         for j in range(600):
        #             output['s'][:, j] /= fac_i[j]
        #         output['s'] = torch.exp(output['s']).detach().cpu().numpy()
        #
        #     if 's_AE' in output:
        #         output['s_AE'] = torch.matmul(output['s_AE'], verb_mapping)
        #         for j in range(600):
        #             output['s_AE'][:, j] /= fac_a[j]
        #         output['s_AE'] = torch.sigmoid(output['s_AE']).detach().cpu().numpy()
        #     if 's_rev' in output:
        #         output['s_rev'] = torch.matmul(output['s_rev'], verb_mapping)
        #         for j in range(600):
        #             output['s_rev'][:, j] /= fac_d[j]
        #         output['s_rev'] = torch.exp(output['s_rev']).detach().cpu().numpy()
        #
        #     for j in range(bbox.shape[0]):
        #         cls = obj_class[j]
        #         x, y = obj_range[cls][0] - 1, obj_range[cls][1]
        #         keys[cls].append(batch['key'][j])
        #         bboxes[cls].append(bbox[j])
        #         scores[cls].append(np.zeros(y - x))
        #         if 's' in output:
        #             scores[cls][-1] += output['s'][j, x:y]
        #         if 's_AE' in output:
        #             scores[cls][-1] += output['s_AE'][j, x:y]
        #         if 's_rev' in output:
        #             scores[cls][-1] += output['s_rev'][j, x:y]
        #         scores[cls][-1] *= batch['hdet'][j]
        #         scores[cls][-1] *= batch['odet'][j]
        #         hdet[cls].append(batch['hdet'][j])
        #         odet[cls].append(batch['odet'][j])
        #     timer.toc()
        #     if i % 1000 == 0:
        #         print("%05d iteration, average time %.4f" % (i, timer.average_time))
        #     timer.tic()
        #
        # timer.toc()
        #
        # for i in range(80):
        #     keys[i] = np.array(keys[i])
        #     bboxes[i] = np.array(bboxes[i])
        #     scores[i] = np.array(scores[i])
        #     hdet[i] = np.array(hdet[i])
        #     odet[i] = np.array(odet[i])
        #
        # sel = []
        # for i in range(600):
        #     sel.append(None)
        #
        # for i in range(80):
        #     x, y = obj_range[cls][0] - 1, obj_range[cls][1]
        #     for hoi_id in range(x, y):
        #         sel[hoi_id] = list(range(len(bboxes[i])))
        #
        # res = {
        #     'keys': keys,
        #     'bboxes': bboxes,
        #     'scores': scores,
        #     'hdet': hdet,
        #     'odet': odet,
        #     'sel': sel,
        # }
        # return res

    def cascaded_hoi(self, input_data):
        raise NotImplementedError
        

def main(rank, args):
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    if args.multiporcessing:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=args.world_size,
            rank=rank
        )

    if args.net == 'scg':

        valset = DataFactory(
            name=args.dataset, partition=args.partitions[1],
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

    elif args.net == 'idn':
        raise NotImplementedError

    net = get_net(args)
    if net == '':
        raise NotImplementedError

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
    if args.net == 'scg':
        tester()
    elif args.net == 'idn':
        tester()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an interaction head")
    parser.add_argument('--world-size', default=1, type=int,
                        help="Number of subprocesses/GPUs to use")
    parser.add_argument('--dataset', default='Custom', type=str)
    parser.add_argument('--net', default='scg', type=str)
    parser.add_argument('--top-k', default=5, type=int)
    parser.add_argument('--partitions', nargs='+', default=['train2015', 'test2015'], type=str)
    parser.add_argument('--data-root', default='Custom', type=str)
    parser.add_argument('--csv-path', default=None, type=str, help="Csv Path is required only for Custom dataset")
    parser.add_argument('--partition', default='test2015', type=str)
    parser.add_argument('--num-iter', default=2, type=int,
                        help="Number of iterations to run message passing")
    parser.add_argument('--box-score-thresh', default=0.0, type=float)
    parser.add_argument('--num-subj-cls', default=6, type=int)
    parser.add_argument('--num-obj-cls', default=9, type=int)
    parser.add_argument('--num-action-cls', default=8, type=int)
    parser.add_argument('--max-subject', default=15, type=int)
    parser.add_argument('--max-object', default=15, type=int)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--checkpoint-path', default='', type=str)
    parser.add_argument('--batch-size', default=1, type=int,
                        help="Batch size for each subprocess")
    parser.add_argument('--config_path', dest='config_path', help='Select config file', default='configs/IDN.yml',
                        type=str)
    parser.add_argument('--multiporcessing', action='store_true', help="Enable multiporcessing")

    args = parser.parse_args()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"
    if args.multiporcessing:
        mp.spawn(main, nprocs=args.world_size, args=(args,))
    else:
        main(1, args)
    print("testing complete!")
