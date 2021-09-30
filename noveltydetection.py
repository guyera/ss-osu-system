import torch

import noveltydetectionfeatures

def main():
    name = 'Custom'
    data_root = 'Custom'
    csv_path = 'Custom/annotations/val_dataset_v1_train.csv'
    num_subj_cls = 6
    num_obj_cls = 9
    num_action_cls = 8
    training = True
    
    # Construct dataset. If the features haven't already been computed and
    # stored in disk, this will do that as well (so it might take awhile),
    # loading image_batch_size images at a time and computing features from
    # them.
    dataset = noveltydetectionfeatures.NoveltyFeatureDataset(
        name = name,
        data_root = data_root,
        csv_path = csv_path,
        num_subj_cls = num_subj_cls,
        num_obj_cls = num_obj_cls,
        num_action_cls = num_action_cls,
        training = training,
        image_batch_size = 64
    )
    
    batch_size = 256
    num_workers = 1
    world_size = 1
    multiprocessing = False
    val_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers, pin_memory=True,
        sampler=DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank
        ) if multiprocessing else None
    )
    
    for spatial_features,\
            subject_appearance_features,\
            object_appearance_features,\
            verb_appearance_features,\
            subject_labels,\
            object_labels,\
            verb_labels\
            in val_loader:
        print(f'Spatial features shape: {spatial_features.shape}')
        print(f'Subject appearance features shape: {subject_appearance_features.shape}')
        print(f'Object appearance features shape: {object_appearance_features.shape}')
        print(f'Verb appearance features shape: {verb_appearance_features.shape}')
        print(f'Subject labels: {subject_labels}')
        print(f'Object labels: {object_labels}')
        print(f'Verb labels: {verb_labels}')

if __name__ == '__main__':
    main()
