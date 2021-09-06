import os
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
from utils import custom_collate, Timer, AverageMeter, get_config, DataLoaderX

from models.idn import AE, IDN
from data.dataset_idn import HICO_train_set, HICO_test_set
import yaml
import re
import pickle
from easydict import EasyDict as edict


def get_net(args):
    if args.net == 'scg':
        net = SCG(
            args.object_to_target, num_classes=args.num_classes,
            num_obj_classes=args.num_obj_classes,
            num_iterations=args.num_iter, postprocess=False,
            max_subject=args.max_subject, max_object=args.max_object,
            box_score_thresh=args.box_score_thresh,
            distributed=True
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
    def __init__(self, net, model_name, data_loader):
        self.net = net
        self.data_loader = data_loader
        self.func_map = {
            'scg': self.scg,
            'drg': self.drg,
            'idn': self.idn,
            'cascaded-hoi': self.cascaded_hoi,
        }
        self.test = self.func_map[model_name]
        self.converter = CustomInput(model_name).converter

    def scg(self, ov_interaction_map):
        results = list()
        for batch in tqdm(self.data_loader):
            inputs = batch[:-1]
            # TODO: Optimize this
            inp_image = np.array(inputs[0][0].cpu())
            # Now assuming that data loader will give us separate lists for objects and subjects
            sub_inp_boxes = np.array(inputs[1][0]['subject_boxes'])
            sub_inp_labels = np.array(inputs[1][0]['subject_labels'])
            sub_inp_scores = np.array(inputs[1][0]['subject_scores'])
            obj_inp_boxes = np.array(inputs[1][0]['object_boxes'])
            obj_inp_labels = np.array(inputs[1][0]['object_labels'])
            obj_inp_scores = np.array(inputs[1][0]['object_scores'])
            input_data = self.converter(inp_image, sub_inp_boxes, sub_inp_labels, sub_inp_scores,
                                        obj_inp_boxes, obj_inp_labels, obj_inp_scores)
            input_data = pocket.ops.relocate_to_cuda(input_data)
            with torch.no_grad():
                output = self.net(*input_data)
                # Batch size is fixed as 1 for inference
                assert len(output) == 1, "Batch size is not 1"
                output = pocket.ops.relocate_to_cpu(output[0])
                # Format detections
                box_idx = output['index']
                # interactions = torch.tensor([
                #     ov_interaction_map[o][v]
                #     for s, v, o in zip(output['subject'][box_idx], output['prediction'], output['object'][box_idx])
                # ])
                result = {
                    'object_scores': output['object_scores'],
                    'subject_scores': output['subject_scores'],
                    # 'interactions': interactions,
                    'img_path': inputs[1][0]['img_path'],
                }
                results.append(result)
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
            detection_root=args.detection_dir,
            training=False,
        )

        val_loader = DataLoader(
            dataset=valset,
            collate_fn=custom_collate, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True,
            sampler=DistributedSampler(
                valset,
                num_replicas=args.world_size,
                rank=rank)
        )

    elif args.net == 'idn':
        args_idn = pickle.load(open('configs/arguments.pkl', 'rb'))
        HO_weight = torch.from_numpy(args_idn['HO_weight'])
        config = get_config(args.config_path)

        val_set = HICO_test_set(config.TRAIN.DATA_DIR, split='test')
        val_loader = DataLoaderX(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=val_set.collate_fn,
                                 pin_memory=False, drop_last=False)

    if args.dataset == 'hicodet':
        if args.net == 'scg':
            args.object_to_target = val_loader.dataset.dataset.object_to_verb
            args.num_obj_classes = val_loader.dataset.dataset.num_object_cls
        args.human_idx = 49
        args.num_classes = 117
    elif args.dataset == 'vcoco':
        if args.net == 'scg':
            raise NotImplementedError
        args.human_idx = 1
        args.num_classes = 24
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
    tester = Test(net, args.net, val_loader).test
    if args.net == 'scg':
        tester(val_loader.dataset.dataset.object_n_verb_to_interaction)
    elif args.net == 'idn':
        tester()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an interaction head")
    parser.add_argument('--world-size', required=True, type=int,
                        help="Number of subprocesses/GPUs to use")
    parser.add_argument('--dataset', default='hicodet', type=str)
    parser.add_argument('--net', default='scg', type=str)
    parser.add_argument('--partitions', nargs='+', default=['train2015', 'test2015'], type=str)
    parser.add_argument('--data-root', default='hicodet', type=str)
    parser.add_argument('--detection-dir', default='hicodet/detections/test2015',
                        type=str, help="Directory where detection files are stored")
    parser.add_argument('--partition', default='test2015', type=str)
    parser.add_argument('--num-iter', default=2, type=int,
                        help="Number of iterations to run message passing")
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--max-subject', default=15, type=int)
    parser.add_argument('--max-object', default=15, type=int)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--checkpoint-path', default='', type=str)
    parser.add_argument('--batch-size', default=1, type=int,
                        help="Batch size for each subprocess")
    parser.add_argument('--config_path', dest='config_path', help='Select config file', default='configs/IDN.yml',
                        type=str)

    args = parser.parse_args()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"

    mp.spawn(main, nprocs=args.world_size, args=(args,))
    print("testing complete!")
