import argparse

import torch

parser = argparse.ArgumentParser()

parser.add_argument(
    '--precomputed-feature-file',
    type=str,
    default='./.features/resizepad=224/none/normalized/training.pth'
)

parser.add_argument(
    '--frequency-save-file',
    type=str,
    default='./.class-frequencies.pth'
)

args = parser.parse_args()

device_id = 0
device = f'cuda:{device_id}'

n_known_species_cls = 10
n_species_cls = 31
n_known_activity_cls = 2
n_activity_cls = 7

species_frequencies = torch.zeros(n_species_cls, dtype=torch.long)
activity_frequencies = torch.zeros(n_activity_cls, dtype=torch.long)

_, species_labels, activity_labels = torch.load(args.precomputed_feature_file)

print(species_labels.shape)
print(activity_labels.shape)

for species_idx in range(n_species_cls):
    n_match = (species_labels == species_idx).to(torch.long).sum()
    species_frequencies[species_idx] = n_match

for activity_idx in range(n_activity_cls):
    n_match = (activity_labels == activity_idx).to(torch.long).sum()
    activity_frequencies[activity_idx] = n_match

print(species_frequencies)
print(species_frequencies.sum())
print(activity_frequencies.sum())

data = (species_frequencies, activity_frequencies)
torch.save(data, args.frequency_save_file)
