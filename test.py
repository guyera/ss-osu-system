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
from utils import custom_collate


def get_net(args):
    nets = {
        'scg': SCG(
                args.object_to_target, args.human_idx, num_classes=args.num_classes,
                num_obj_classes=args.num_obj_classes,
                num_iterations=args.num_iter, postprocess=False,
                max_human=args.max_human, max_object=args.max_object,
                box_score_thresh=args.box_score_thresh,
                distributed=True
            ),
        'idn': '',
        'drg': '',
        'cascaded-hoi': '',
    }
    if args.net not in nets:
        raise NotImplementedError
    return nets[args.net]


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
            inputs = pocket.ops.relocate_to_cuda(batch[:-1])
            # TODO: Optimize this
            inp_image = np.array(inputs[0][0].cpu())
            inp_boxes = np.array(inputs[1][0]['boxes'].cpu())
            inp_labels = np.array(inputs[1][0]['labels'].cpu())
            inp_scores = np.array(inputs[1][0]['scores'].cpu())
            input_data = self.converter(inp_image, inp_boxes, inp_labels, inp_scores)
            input_data = pocket.ops.relocate_to_cuda(input_data)
            with torch.no_grad():
                output = self.net(*input_data)
                # Batch size is fixed as 1 for inference
                assert len(output) == 1, "Batch size is not 1"
                output = pocket.ops.relocate_to_cpu(output[0])
                # Format detections
                box_idx = output['index']
                objects = output['object'][box_idx]
                scores = output['scores']
                verbs = output['prediction']
                interactions = torch.tensor([
                    ov_interaction_map[o][v]
                    for o, v in zip(objects, verbs)
                ])
                # Associate detected pairs with ground truth pairs
                labels = torch.zeros_like(scores)
                result = {
                    'scores': scores,
                    'interactions': interactions,
                    'labels': labels,
                }
                results.append(result)
        return results

    def drg(self, input_data):
        raise NotImplementedError

    def idn(self, input_data):
        raise NotImplementedError

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

    valset = DataFactory(
        name=args.dataset, partition=args.partitions[1],
        data_root=args.data_root,
        detection_root=args.detection_dir
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
    if args.dataset == 'hicodet':
        args.object_to_target = val_loader.dataset.dataset.object_to_verb
        args.human_idx = 49
        args.num_classes = 117
    elif args.dataset == 'vcoco':
        args.object_to_target = val_loader.dataset.dataset.object_to_action
        args.human_idx = 1
        args.num_classes = 24
    args.num_obj_classes = val_loader.dataset.dataset.num_object_cls
    net = get_net(args)
    if net == '':
        raise NotImplementedError

    if os.path.exists(args.checkpoint_path):
        print("Loading model from ", args.checkpoint_path)
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        net.load_state_dict(checkpoint['model_state_dict'])
    elif len(args.model_path):
        print("\nWARNING: The given model path does not exist. "
              "Proceed to use a randomly initialised model.\n")

    net.cuda()
    net.eval()
    tester = Test(net, args.net, val_loader).test
    if args.net == 'scg':
        tester(val_loader.dataset.dataset.object_n_verb_to_interaction)
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
    parser.add_argument('--max-human', default=15, type=int)
    parser.add_argument('--max-object', default=15, type=int)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--checkpoint-path', default='', type=str)
    parser.add_argument('--batch-size', default=1, type=int,
                        help="Batch size for each subprocess")

    args = parser.parse_args()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"

    mp.spawn(main, nprocs=args.world_size, args=(args,))