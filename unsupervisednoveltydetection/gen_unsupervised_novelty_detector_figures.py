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

sv_filtered_subject_scores = []
so_filtered_subject_scores = []

sv_filtered_verb_scores = []
vo_filtered_verb_scores = []

so_filtered_object_scores = []
vo_filtered_object_scores = []
for idx in range(len(subject_novelty_scores)):
    subject_score = subject_novelty_scores[idx]
    verb_score = verb_novelty_scores[idx]
    object_score = object_novelty_scores[idx]
    
    if subject_score is not None:
        if verb_score is not None:
            sv_filtered_subject_scores.append(subject_score)
        if object_score is not None:
            so_filtered_subject_scores.append(subject_score)
    
    if verb_score is not None:
        if subject_score is not None:
            sv_filtered_verb_scores.append(verb_score)
        if object_score is not None:
            vo_filtered_verb_scores.append(verb_score)
    
    if object_score is not None:
        if subject_score is not None:
            so_filtered_object_scores.append(object_score)
        if verb_score is not None:
            vo_filtered_object_scores.append(object_score)
    
sv_filtered_subject_scores = torch.stack(sv_filtered_subject_scores, dim = 0)
so_filtered_subject_scores = torch.stack(so_filtered_subject_scores, dim = 0)

sv_filtered_verb_scores = torch.stack(sv_filtered_verb_scores, dim = 0)
vo_filtered_verb_scores = torch.stack(vo_filtered_verb_scores, dim = 0)

so_filtered_object_scores = torch.stack(so_filtered_object_scores, dim = 0)
vo_filtered_object_scores = torch.stack(vo_filtered_object_scores, dim = 0)

if not os.path.exists(args.figure_dir):
    os.makedirs(args.figure_dir)

# Verb vs. subject scatter plot
fig, ax = plt.subplots()
ax.set_title('Verb novelty scores vs subject novelty scores')
ax.set_xlabel('Subject novelty scores')
ax.set_ylabel('Verb novelty scores')
ax.scatter(sv_filtered_subject_scores.detach().cpu().numpy(), sv_filtered_verb_scores.detach().cpu().numpy())
a = torch.stack((sv_filtered_subject_scores, torch.ones_like(sv_filtered_subject_scores)), dim = 1)
y = sv_filtered_verb_scores
x = torch.matmul(torch.matmul(torch.linalg.inv(torch.matmul(a.T, a)), a.T), y)
y_hat = torch.matmul(a, x)
ax.plot(sv_filtered_subject_scores.detach().cpu().numpy(), y_hat.detach().cpu().numpy(), color = 'black')
r = torch.corrcoef(torch.stack((sv_filtered_subject_scores, sv_filtered_verb_scores), dim = 0))[0, 1]
text = f'r = {float(r):.2f}'
props = dict(boxstyle = 'round', facecolor = 'tab:orange', alpha = 0.85)
ax.text(0.95, 0.05, text, transform = ax.transAxes, fontsize = 14, verticalalignment = 'bottom', horizontalalignment = 'right', bbox = props)
fig.savefig(os.path.join(args.figure_dir, 'subject_verb_scatter.jpg'))
plt.close(fig)

# Object vs. subject scatter plot
fig, ax = plt.subplots()
ax.set_title('Object novelty scores vs subject novelty scores')
ax.set_xlabel('Subject novelty scores')
ax.set_ylabel('Object novelty scores')
ax.scatter(so_filtered_subject_scores.detach().cpu().numpy(), so_filtered_object_scores.detach().cpu().numpy())
a = torch.stack((so_filtered_subject_scores, torch.ones_like(so_filtered_subject_scores)), dim = 1)
y = so_filtered_object_scores
x = torch.matmul(torch.matmul(torch.linalg.inv(torch.matmul(a.T, a)), a.T), y)
y_hat = torch.matmul(a, x)
ax.plot(so_filtered_subject_scores.detach().cpu().numpy(), y_hat.detach().cpu().numpy(), color = 'black')
r = torch.corrcoef(torch.stack((so_filtered_subject_scores, so_filtered_object_scores), dim = 0))[0, 1]
text = f'r = {float(r):.2f}'
props = dict(boxstyle = 'round', facecolor = 'tab:orange', alpha = 0.85)
ax.text(0.95, 0.05, text, transform = ax.transAxes, fontsize = 14, verticalalignment = 'bottom', horizontalalignment = 'right', bbox = props)
fig.savefig(os.path.join(args.figure_dir, 'subject_object_scatter.jpg'))
plt.close(fig)

# Object vs. verb scatter plot
fig, ax = plt.subplots()
ax.set_title('Object novelty scores vs verb novelty scores')
ax.set_xlabel('Verb novelty scores')
ax.set_ylabel('Object novelty scores')
ax.scatter(vo_filtered_verb_scores.detach().cpu().numpy(), vo_filtered_object_scores.detach().cpu().numpy())
a = torch.stack((vo_filtered_verb_scores, torch.ones_like(vo_filtered_verb_scores)), dim = 1)
y = vo_filtered_object_scores
x = torch.matmul(torch.matmul(torch.linalg.inv(torch.matmul(a.T, a)), a.T), y)
y_hat = torch.matmul(a, x)
ax.plot(vo_filtered_verb_scores.detach().cpu().numpy(), y_hat.detach().cpu().numpy(), color = 'black')
r = torch.corrcoef(torch.stack((vo_filtered_verb_scores, vo_filtered_object_scores), dim = 0))[0, 1]
text = f'r = {float(r):.2f}'
props = dict(boxstyle = 'round', facecolor = 'tab:orange', alpha = 0.85)
ax.text(0.95, 0.05, text, transform = ax.transAxes, fontsize = 14, verticalalignment = 'bottom', horizontalalignment = 'right', bbox = props)
fig.savefig(os.path.join(args.figure_dir, 'verb_object_scatter.jpg'))
plt.close(fig)
