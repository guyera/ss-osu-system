import os
import torch
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

from models.scg import SpatiallyConditionedGraph as SCG
from models.scg import CustomisedDLE
from data.data_factory import DataFactory
from utils import custom_collate, Timer, AverageMeter, get_config, DataLoaderX

import pickle
import torch.optim as optim
from models.idn import IDN
from data.dataset_idn import HICO_train_set, HICO_test_set

train_timer = Timer()


class Train(object):
    def __init__(self, net, model_name, train_loader, val_loader):
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.func_map = {
            'scg': self.scg,
            'drg': self.drg,
            'idn': self.idn,
            'cascaded-hoi': self.cascaded_hoi,
        }
        self.train = self.func_map[model_name]

    def scg(self, epoch, iteration, args):
        engine = CustomisedDLE(
            self.net,
            self.train_loader,
            self.val_loader,
            num_classes=args.num_classes,
            print_interval=args.print_interval,
            cache_dir=args.cache_dir
        )
        # Seperate backbone parameters from the rest
        param_group_1 = []
        param_group_2 = []
        for k, v in engine.fetch_state_key('net').named_parameters():
            if v.requires_grad:
                if k.startswith('module.backbone'):
                    param_group_1.append(v)
                elif k.startswith('module.interaction_head'):
                    param_group_2.append(v)
                else:
                    raise KeyError(f"Unknown parameter name {k}")
        # Fine-tune backbone with lower learning rate
        optim = torch.optim.AdamW([
            {'params': param_group_1, 'lr': args.learning_rate * args.lr_decay},
            {'params': param_group_2}
        ], lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        lambda1 = lambda epoch: 1. if epoch < args.milestones[0] else args.lr_decay
        lambda2 = lambda epoch: 1. if epoch < args.milestones[0] else args.lr_decay
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim, lr_lambda=[lambda1, lambda2]
        )
        # Override optimiser and learning rate scheduler
        engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)
        engine.update_state_key(epoch=epoch, iteration=iteration)

        engine(args.num_epochs)

    def drg(self, input_data):
        raise NotImplementedError

    def idn(self, epoch, iteration, args):
        raise NotImplementedError
        # def idn_train(net, loader, optimizer, timer, epoch):
        #     net.train()
        #     net.cuda()
        #     global step
        #     step = 0
        #
        #     timer.tic()
        #     meters = {
        #         'L_rec': AverageMeter(),
        #         'L_cls': AverageMeter(),
        #         'L_ae': AverageMeter(),
        #         'loss': AverageMeter()
        #     }
        #     for i, batch in enumerate(loader):
        #         n = batch['spatial'].shape[0]
        #
        #         batch['spatial'] = batch['spatial'].cuda(non_blocking=True)
        #         batch['labels_s'] = batch['labels_s'].cuda(non_blocking=True)
        #         batch['labels_r'] = batch['labels_r'].cuda(non_blocking=True)
        #         batch['labels_ro'] = batch['labels_ro'].cuda(non_blocking=True)
        #         batch['labels_sro'] = batch['labels_sro'].cuda(non_blocking=True)
        #         batch['sub_vec'] = batch['sub_vec'].cuda(non_blocking=True)
        #         batch['obj_vec'] = batch['obj_vec'].cuda(non_blocking=True)
        #         batch['uni_vec'] = batch['uni_vec'].cuda(non_blocking=True)
        #
        #         output = net(batch)
        #         loss = torch.mean(output['loss'])
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #
        #         for key in output.keys():
        #             if key in meters:
        #                 meters[key].update(torch.mean(output[key]).detach().cpu().data, n)
        #
        #         timer.toc()
        #         timer.tic()
        #         if i % 2000 == 0:
        #             print("%03d epoch, %05d iter, average time %.4f, loss %.4f" % (
        #             epoch, i, timer.average_time, loss.detach().cpu().data))
        #         step += 1
        #
        #     timer.toc()
        #
        #     return net, meters
        #
        # config = get_config(args.config_path)
        # optimizer = optim.SGD(self.net.parameters(), lr=config.TRAIN.OPTIMIZER.lr,
        #                       momentum=config.TRAIN.OPTIMIZER.momentum,
        #                       weight_decay=config.TRAIN.OPTIMIZER.weight_decay)
        #
        # for i in range(args.num_epochs):  # config.TRAIN.MAX_EPOCH
        #     train_str = "%03d epoch training" % i
        #     net, train_meters = idn_train(self.net, self.train_loader, optimizer, train_timer, i)
        #     for (key, value) in train_meters.items():
        #         train_str += ", %s=%.4f" % (key, value.avg)

    def cascaded_hoi(self, input_data):
        raise NotImplementedError


def get_net(args):
    if args.net == 'scg':
        net = SCG(
            num_classes=args.num_classes,
            num_obj_classes=args.num_obj_classes,
            num_subject_classes=args.num_subject_classes,
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

    elif args.net == 'cascaded-hoi':
        net = ''

    if net == '':
        raise NotImplementedError
    return net


def main(rank, args):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )

    if args.net == 'scg':
        trainset = DataFactory(
            name=args.dataset, partition=args.partitions[1],
            data_root=args.data_root,
            detection_root=args.train_detection_dir,
            flip=True
        )

        valset = DataFactory(
            name=args.dataset, partition=args.partitions[1],
            data_root=args.data_root,
            detection_root=args.val_detection_dir
        )

        train_loader = DataLoader(
            dataset=trainset,
            collate_fn=custom_collate, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True,
            sampler=DistributedSampler(
                trainset,
                num_replicas=args.world_size,
                rank=rank)
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
        train_set = HICO_train_set(config, split='trainval', train_mode=True)
        train_loader = DataLoaderX(train_set, batch_size=config.TRAIN.DATASET.BATCH_SIZE, shuffle=True,
                                   collate_fn=train_set.collate_fn, pin_memory=False, drop_last=False)

        val_set = HICO_test_set(config.TRAIN.DATA_DIR, split='test')
        val_loader = DataLoaderX(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=val_set.collate_fn,
                                 pin_memory=False, drop_last=False)
    #         train_loader = val_loader
    # Fix random seed for model synchronisation
    torch.manual_seed(args.random_seed)

    if args.dataset == 'hicodet':
        if args.net == 'scg':
            args.num_obj_classes = train_loader.dataset.dataset.num_object_cls
        args.num_classes = 117
    elif args.dataset == 'Custom':
        if args.net == 'scg':
            args.num_obj_classes = train_loader.dataset.dataset.num_object_cls
            args.num_subject_classes = train_loader.dataset.dataset.num_subject_cls
        args.num_classes = 63
    elif args.dataset == 'vcoco':
        if args.net == 'scg':
            args.num_obj_classes = train_loader.dataset.dataset.num_object_cls
        args.num_classes = 24

    net = get_net(args)
    if net == '':
        raise NotImplementedError

    if os.path.exists(args.checkpoint_path):
        print("=> Rank {}: continue from saved checkpoint".format(
            rank), args.checkpoint_path)
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        optim_state_dict = checkpoint['optim_state_dict']
        sched_state_dict = checkpoint['scheduler_state_dict']
        epoch = checkpoint['epoch']
        iteration = checkpoint['iteration']
    else:
        print("=> Rank {}: start from a randomly initialised model".format(rank))
        optim_state_dict = None
        sched_state_dict = None
        epoch = 0
        iteration = 0

    print("Data and model loaded")

    # TODO: Pass model_name through args here, also implement conditional calling based on models

    trainer = Train(net, args.net, train_loader, val_loader).train
    trainer(epoch, iteration, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an interaction head")
    parser.add_argument('--world-size', required=True, type=int,
                        help="Number of subprocesses/GPUs to use")
    parser.add_argument('--dataset', default='Custom', type=str)
    parser.add_argument('--net', default='scg', type=str)
    parser.add_argument('--partitions', nargs='+', default=['train2015', 'test2015'], type=str)
    parser.add_argument('--data-root', default='hicodet', type=str, help="Give full csv path for Custom dataset")
    parser.add_argument('--train-detection-dir', default='hicodet/detections/test2015', type=str)
    parser.add_argument('--val-detection-dir', default='hicodet/detections/test2015', type=str)
    parser.add_argument('--num-iter', default=2, type=int,
                        help="Number of iterations to run message passing")
    parser.add_argument('--num-epochs', default=8, type=int)
    parser.add_argument('--random-seed', default=1, type=int)
    parser.add_argument('--learning-rate', default=0.0001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--batch-size', default=4, type=int,
                        help="Batch size for each subprocess")
    parser.add_argument('--lr-decay', default=0.1, type=float,
                        help="The multiplier by which the learning rate is reduced")
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--max-subject', default=15, type=int)
    parser.add_argument('--max-object', default=15, type=int)
    parser.add_argument('--milestones', nargs='+', default=[6, ], type=int,
                        help="The epoch number when learning rate is reduced")
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--print-interval', default=300, type=int)
    parser.add_argument('--checkpoint-path', default='', type=str)
    parser.add_argument('--cache-dir', type=str, default='./checkpoints')
    parser.add_argument('--config_path', dest='config_path', help='Select config file', default='configs/IDN.yml',
                        type=str)

    args = parser.parse_args()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8889"

    mp.spawn(main, nprocs=args.world_size, args=(args,))
