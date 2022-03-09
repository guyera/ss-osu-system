import sys
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--novelty-score-dir', type = str, required = True)
parser.add_argument('--figure-dir', type = str, required = True)
parser.add_argument('--max-sample-size', type = int, default = 1000)

args = parser.parse_args()

subject_novelty_scores = []
verb_novelty_scores = []
object_novelty_scores = []

subdirs = os.listdir(args.novelty_score_dir)
for subdir in subdirs:
    subdir_path = os.path.join(args.novelty_score_dir, subdir)
    subject_score_path = os.path.join(subdir_path, 'subject.pth')
    verb_score_path = os.path.join(subdir_path, 'verb.pth')
    object_score_path = os.path.join(subdir_path, 'object.pth')
    
    trial_subject_novelty_scores = torch.load(subject_score_path)
    trial_verb_novelty_scores = torch.load(verb_score_path)
    trial_object_novelty_scores = torch.load(object_score_path)

    subject_novelty_scores += trial_subject_novelty_scores
    verb_novelty_scores += trial_verb_novelty_scores
    object_novelty_scores += trial_object_novelty_scores

if args.max_sample_size < len(subject_novelty_scores):
    sampled_indices = np.random.choice(np.arange(len(subject_novelty_scores)), size = args.max_sample_size, replace = False)
    subject_novelty_scores = [subject_novelty_scores[idx] for idx in sampled_indices]
    verb_novelty_scores = [verb_novelty_scores[idx] for idx in sampled_indices]
    object_novelty_scores = [object_novelty_scores[idx] for idx in sampled_indices]

filtered_subject_scores = []
filtered_verb_scores = []
filtered_object_scores = []
for idx in range(len(subject_novelty_scores)):
    subject_score = subject_novelty_scores[idx]
    verb_score = verb_novelty_scores[idx]
    object_score = object_novelty_scores[idx]
    
    if subject_score is not None:
        filtered_subject_scores.append(subject_score)
    if verb_score is not None:
        filtered_verb_scores.append(verb_score)
    if object_score is not None:
        filtered_object_scores.append(object_score)

filtered_subject_scores = torch.stack(filtered_subject_scores, dim = 0)
filtered_verb_scores = torch.stack(filtered_verb_scores, dim = 0)
filtered_object_scores = torch.stack(filtered_object_scores, dim = 0)

if not os.path.exists(args.figure_dir):
    os.makedirs(args.figure_dir)

# Subject histogram
fig, ax = plt.subplots()
ax.set_title('Subject novelty score distribution')
ax.set_xlabel('Subject novelty scores')
ax.hist(filtered_subject_scores.detach().cpu().numpy())
fig.savefig(os.path.join(args.figure_dir, 'subject_histogram.jpg'))
plt.close(fig)

# Verb histogram
fig, ax = plt.subplots()
ax.set_title('Verb novelty score distribution')
ax.set_xlabel('Verb novelty scores')
ax.hist(filtered_verb_scores.detach().cpu().numpy())
fig.savefig(os.path.join(args.figure_dir, 'verb_histogram.jpg'))
plt.close(fig)

# Object histogram
fig, ax = plt.subplots()
ax.set_title('Object novelty score distribution')
ax.set_xlabel('Object novelty scores')
ax.hist(filtered_object_scores.detach().cpu().numpy())
fig.savefig(os.path.join(args.figure_dir, 'object_histogram.jpg'))
plt.close(fig)
