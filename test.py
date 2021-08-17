import os
import torch
import argparse
import torchvision
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

import pickle
import pocket
import numpy as np

from models.scg import SpatiallyConditionedGraph as SCG
from data.data_factory import DataFactory, CustomInput
from utils import custom_collate

from models.idn import AE, IDN
from prefetch_generator import BackgroundGenerator
from dataset_idn import HICO_train_set, HICO_test_set
from utils_idn import Timer, HO_weight, AverageMeter, fac_i, fac_a, fac_d, nis_thresh
import yaml
import re
from easydict import EasyDict as edict

def get_config(args):
    loader = yaml.FullLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    
    config = edict(yaml.load(open(args.config_path, 'r'), Loader=loader))
    return config

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

verb_mapping = torch.from_numpy(pickle.load(open('verb_mapping.pkl', 'rb'), encoding='latin1')).float()


class Test(object):
    def __init__(self, net, model_name):
        self.net = net
        self.func_map = {
            'scg': self.scg,
            'drg': self.drg,
            'idn': self.idn,
            'cascaded-hoi': self.cascaded_hoi,
        }
        self.test = self.func_map[model_name]

    def scg(self, input_data, ov_interaction_map):
        input_data = pocket.ops.relocate_to_cuda(input_data)
        with torch.no_grad():
            output = self.net(*input_data)
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, "Batch size is not 1"
            output = pocket.ops.relocate_to_cpu(output[0])
            # Format detections
            box_idx = output['index']
            # boxes_h = output['boxes_h'][box_idx]
            # boxes_o = output['boxes_o'][box_idx]
            objects = output['object'][box_idx]
            scores = output['scores']
            verbs = output['prediction']
            interactions = torch.tensor([
                ov_interaction_map[o][v]
                for o, v in zip(objects, verbs)
            ])
            # Associate detected pairs with ground truth pairs
            labels = torch.zeros_like(scores)
        return scores, interactions, labels

    def drg(self, input_data):
        raise NotImplementedError

    def idn(self, batch):
        with torch.no_grad():
            output = self.net(batch)
            batch['spatial'][:, 0] *= batch['shape'][:, 0]
            batch['spatial'][:, 1] *= batch['shape'][:, 1]
            batch['spatial'][:, 2] *= batch['shape'][:, 0]
            batch['spatial'][:, 3] *= batch['shape'][:, 1]
            batch['spatial'][:, 4] *= batch['shape'][:, 0]
            batch['spatial'][:, 5] *= batch['shape'][:, 1]
            batch['spatial'][:, 6] *= batch['shape'][:, 0]
            batch['spatial'][:, 7] *= batch['shape'][:, 1]
            obj_class = batch['obj_class']
            bbox = batch['spatial'].detach().cpu().numpy()

            if 's' in output:
                output['s'] = torch.matmul(output['s'].detach().cpu(), verb_mapping)
                for j in range(600):
                    output['s'][:, j] /= fac_i[j]
                output['s'] = torch.exp(output['s']).detach().cpu().numpy()

            if 's_AE' in output:
                output['s_AE'] = torch.matmul(output['s_AE'].detach().cpu(), verb_mapping)
                for j in range(600):
                    output['s_AE'][:, j] /= fac_a[j]
                output['s_AE'] = torch.sigmoid(output['s_AE']).detach().cpu().numpy()
            if 's_rev' in output:
                output['s_rev'] = torch.matmul(output['s_rev'].detach().cpu(), verb_mapping)
                for j in range(600):
                    output['s_rev'][:, j] /= fac_d[j]
                output['s_rev'] = torch.exp(output['s_rev']).detach().cpu().numpy()
            return output

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

    if args.model_name=='scg':

        valset = DataFactory(
            name=args.dataset, partition=args.partitions[1],
            data_root=args.data_root,
            detection_root=args.val_detection_dir
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
        
    elif args.model_name=='idn': 
        args_idn = pickle.load(open('arguments.pkl', 'rb'))
        HO_weight = torch.from_numpy(args_idn['HO_weight'])
        config = get_config(args)
        
        val_set    = HICO_test_set(config.TRAIN.DATA_DIR, split='test')
        val_loader = DataLoaderX(val_set, batch_size=1, shuffle=False, collate_fn=val_set.collate_fn, pin_memory=False, drop_last=False)
#         train_loader = val_loader
    # Fix random seed for model synchronisation
    torch.manual_seed(42)

    
    if args.dataset == 'hicodet':
        if args.model_name=='scg':
            object_to_target = train_loader.dataset.dataset.object_to_verb
        human_idx = 49
        num_classes = 117
    elif args.dataset == 'vcoco':
        if args.model_name=='scg':
            object_to_target = train_loader.dataset.dataset.object_to_action
        human_idx = 1
        num_classes = 24

    # num_anno = torch.tensor(HICODet(None, anno_file=os.path.join(
    #     args.data_root, 'instances_train2015.json')).anno_interaction)
    # rare = torch.nonzero(num_anno < 10).squeeze(1)
    # non_rare = torch.nonzero(num_anno >= 10).squeeze(1)

    if args.model_name=='scg':
        net = SCG(
            object_to_target, human_idx, num_classes=num_classes,
            num_iterations=args.num_iter, postprocess=False,
            max_human=args.max_human, max_object=args.max_object,
            box_score_thresh=args.box_score_thresh,
            distributed=True
        )
    
    elif args.model_name=='idn':
        args_idn      = pickle.load(open('arguments.pkl', 'rb'))
        HO_weight = torch.from_numpy(args_idn['HO_weight'])
        config = get_config(args)
        net = IDN(config.MODEL, HO_weight)

    if os.path.exists(args.model_path):
        print("Loading model from ", args.model_path)
        checkpoint = torch.load(args.model_path, map_location="cpu")
        net.load_state_dict(checkpoint['model_state_dict'])
    elif len(args.model_path):
        print("\nWARNING: The given model path does not exist. "
              "Proceed to use a randomly initialised model.\n")

    net.cuda()
    net.eval()
    # Sample input test
    test_input = pickle.load(open('idn_sample_input.pkl', 'rb'))
    image = np.array(test_input[0][0].cpu())
    boxes = np.array(test_input[1][0]['boxes'].cpu())
    labels = np.array(test_input[1][0]['labels'].cpu())
    scores = np.array(test_input[1][0]['scores'].cpu())
    # TODO: Pass model_name through args here, also implement conditional calling based on models
    tester = Test(net, args.model_name).test
    converter = CustomInput(args.model_name).converter
    input_data = converter(image, boxes, labels, scores)
    if args.model_name=='scg':
        tester(input_data, val_loader.dataset.dataset.object_n_verb_to_interaction)
    elif args.model_name=='idn':
        tester(input_data)
        
    # timer = pocket.utils.HandyTimer(maxlen=1)

    # with timer:
    #     test_ap = test(net, dataloader)
    # print("Model at epoch: {} | time elapsed: {:.2f}s\n"
    #       "Full: {:.4f}, rare: {:.4f}, non-rare: {:.4f}".format(
    #     epoch, timer[0], test_ap.mean(),
    #     test_ap[rare].mean(), test_ap[non_rare].mean()
    # ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an interaction head")
    parser.add_argument('--world-size', required=True, type=int,
                        help="Number of subprocesses/GPUs to use")
    parser.add_argument('--dataset', default='hicodet', type=str)
    parser.add_argument('--model-name', default='idn', type=str)
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
    parser.add_argument('--model-path', default='', type=str)
    parser.add_argument('--batch-size', default=1, type=int,
                        help="Batch size for each subprocess")
    parser.add_argument('--config_path', dest='config_path',help='Select config file', default='configs/IDN.yml', type=str)

    args = parser.parse_args()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"

    mp.spawn(main, nprocs=args.world_size, args=(args,))
    print("testing complete!")