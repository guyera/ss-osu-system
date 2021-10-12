import os
import torch
from torch.utils.data import DataLoader
from models.scg import SpatiallyConditionedGraph as SCG
from data.data_factory import DataFactory
from utils import custom_collate
import pathlib
from ensemble.ensemble import Ensemble
import itertools
import noveltydetectionfeatures
import unsupervisednoveltydetection

class Args:
    def __init__(self):
        self.world_size = None
        self.dataset = None
        self.data_root = None
        self.csv_path = None
        self.num_subj_cls = None
        self.num_obj_cls = None
        self.num_action_cls = None
        self.num_workers = None
        self.batch_size = None
        self.max_object = None
        self.max_subject = None
        self.num_iter = None
        self.box_score_thresh = None
        self.num_iter = None
        self.top_k = None
        self.checkpoint_path = None
        self.ensemble_path = None
        self.pretrained_unsupervised_novelty_detection_path = None


class UnsupervisedNoveltyDetectionManager:
    def __init__(self, pretrained_path, num_subject_classes, num_object_classes, 
        num_verb_classes, data_root, csv_path):

        self.detector = unsupervisednoveltydetection.UnsupervisedNoveltyDetector(num_appearance_features=12544, 
            num_verb_features=12616, 
            num_hidden_nodes=1024, 
            num_subj_cls=num_subject_classes, 
            num_obj_cls=num_object_classes, 
            num_action_cls=num_verb_classes)

        self.detector = self.detector.to('cuda:0')

        state_dict = torch.load(pretrained_path)
        self.detector.load_state_dict(state_dict)

        self.testing_set = noveltydetectionfeatures.NoveltyFeatureDataset(
            name = 'Custom',
            data_root = data_root,
            csv_path = csv_path,
            num_subj_cls = num_subject_classes,
            num_obj_cls = num_object_classes,
            num_action_cls = num_verb_classes,
            training = False,
            image_batch_size = 4,
            feature_extraction_device = 'cuda:0'
        )

    def get_top3_novel(self):
        spatial_features = []
        subject_appearance_features = []
        object_appearance_features = []
        verb_appearance_features = []

        for example_spatial_features, example_subject_appearance_features, example_object_appearance_features, example_verb_appearance_features, _, _, _ in self.testing_set:
        
            # For testing case 1:
            spatial_features.append(example_spatial_features)
            subject_appearance_features.append(example_subject_appearance_features)
            object_appearance_features.append(example_object_appearance_features)
            verb_appearance_features.append(example_verb_appearance_features)
            
            # For testing case 2:
            # spatial_features.append(example_spatial_features)
            # subject_appearance_features.append(example_subject_appearance_features)
            # object_appearance_features.append(None)
            # verb_appearance_features.append(example_verb_appearance_features)

            # For testing case 3:
            # spatial_features.append(None)
            # subject_appearance_features.append(None)
            # object_appearance_features.append(example_object_appearance_features)
            # verb_appearance_features.append(None)

        with torch.no_grad():
            results = self.detector(spatial_features, subject_appearance_features, verb_appearance_features, object_appearance_features, 
                torch.full((5,), 0.2, device = 'cuda:0'))

        return results


def build_merged_top3_SVOs(top3_non_novel, top3_novel, p_ni):
    """
    Merges top-3 non-novel SVOs and top-3 novel SVOs
 
    Parameters:
    -----------
    top3_non_novel:
        List of length N where each element is a list of length 3 containing tuples of following form:
                ((S, V, O), Probability)
    top3_novel: 
        List of length N where each element is a list of length 3 containing tuples of following form:
                ((S, V, O), Probability)
    p_ni:
        The Probability that each image contains novelty. A tensor with shape [N, 1]
 
    Returns:
    --------
        List of length N where each element is a list of length 3 containing tuples of following form:
                ((S, V, O), Probability)
    """  
    N = len(p_ni)
    # p_ni = p_ni.view(-1).numpy()
    top3_non_novel = [[(x[i][0], x[i][1], x[i][1] * (1 - y)) for i in range(3)] for x, y in zip(top3_non_novel, p_ni)]
    top3_novel = [[(x[i][0], x[i][1], x[i][1] * y) for i in range(3)] for x, y in zip(top3_novel, p_ni)]
    all_tuples = [x + y for x, y in zip(top3_non_novel, top3_novel)]
    comb_iter = [itertools.combinations(all_tuples[i], 3) for i in range(N)]
    scores = [list(map(lambda x: (x, x[0][2] + x[1][2] + x[2][2]), comb_iter[i])) for i in range(N)]
    scores = [sorted(i, key=lambda x: x[1], reverse=True)[0] for i in scores]
    scores = [((a[0][0][0], a[0][0][1]), (a[0][1][0], a[0][1][1]), (a[0][2][0], a[0][2][1])) for a in scores]

    return scores

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"

    args = Args()

    args.world_size = 1
    args.dataset = 'Custom'
    args.data_root = 'Custom'
    args.csv_path = 'Custom/annotations/val_dataset_v1_val.csv'
    args.ensemble_path = 'pretrained'
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
    args.top_k = 3
    args.pretrained_unsupervised_novelty_detection_path = 'unsupervisednoveltydetection/unsupervised_novelty_detection_module.pth'

    rank = 1

    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False

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
        sampler=None
    )

    if not pathlib.Path(args.ensemble_path).exists():
        raise Exception(f'pretrained models were not found in path {args.ensemble_path}')

    print('Initializing SCG ensemble...')

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
            distributed=False))

        print(f"Loading pretrained model from {p}")
        checkpoint = torch.load(p, map_location="cpu")
        nets[-1].load_state_dict(checkpoint['model_state_dict'])
        nets[-1].cuda()
        nets[-1].eval()

    calibrator = Ensemble(nets, val_loader, args.top_k, args.num_obj_cls, args.num_subj_cls, args.num_action_cls)

    print('Completed!\n')

    print('Initializing Unsupervised Novelty Detection...')

    unsupervised_novelty_detection_manager = UnsupervisedNoveltyDetectionManager(args.pretrained_unsupervised_novelty_detection_path, 
        args.num_subj_cls, args.num_obj_cls, args.num_action_cls, args.data_root, args.csv_path)
    
    print('Completed!\n')


    print('Computing top-3 SVOs from SCG ensemble...')

    scg_preds = calibrator.get_top3_SVOs(False)

    print('Completed!\n')


    print('Computing top-3 SVOs from Unsupervised Novelty Detection...')

    unsupervised_novelty_detection_preds = unsupervised_novelty_detection_manager.get_top3_novel()

    print('Completed!\n')


    print('Merging top-3 SVOs...')

    p_ni = [0.5 for i in range(len(scg_preds))]
    merged = build_merged_top3_SVOs(scg_preds, unsupervised_novelty_detection_preds['top3'], p_ni)

    print('Completed!\n')