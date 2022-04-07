from argparse import ArgumentParser
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch
import pickle
import zipfile

def compute_moving_avg(vals, num_elems):
    base_offset = vals.shape[0] - num_elems
    assert base_offset >= 0, "There has to be at least 8 images."
    res = np.zeros(num_elems)
    
    for i in range(base_offset, base_offset + num_elems):
        if i >= 7:
            res[i - base_offset] = vals[i - 7:i + 1].mean()
            
    return res
    
def plot_un(subject_novelty_scores, verb_novelty_scores, object_novelty_scores, out_dir, max_sample_size=1000):
    if max_sample_size < len(subject_novelty_scores):
        sampled_indices = np.random.choice(np.arange(len(subject_novelty_scores)), size = max_sample_size, replace = False)
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
    # r = torch.corrcoef(torch.stack((sv_filtered_subject_scores, sv_filtered_verb_scores), dim = 0))[0, 1]
    r = np.corrcoef(torch.stack((sv_filtered_subject_scores, sv_filtered_verb_scores), dim = 0).numpy())[0, 1]
    text = f'r = {float(r):.2f}'
    props = dict(boxstyle = 'round', facecolor = 'tab:orange', alpha = 0.85)
    ax.text(0.95, 0.05, text, transform = ax.transAxes, fontsize = 14, verticalalignment = 'bottom', horizontalalignment = 'right', bbox = props)
    # fig.savefig(os.path.join(args.figure_dir, 'subject_verb_scatter.jpg'))
    fig.savefig(out_dir.joinpath('subject_verb_scatter.jpg'))
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
    # r = torch.corrcoef(torch.stack((so_filtered_subject_scores, so_filtered_object_scores), dim = 0))[0, 1]
    r = np.corrcoef(torch.stack((so_filtered_subject_scores, so_filtered_object_scores), dim = 0).numpy())[0, 1]
    text = f'r = {float(r):.2f}'
    props = dict(boxstyle = 'round', facecolor = 'tab:orange', alpha = 0.85)
    ax.text(0.95, 0.05, text, transform = ax.transAxes, fontsize = 14, verticalalignment = 'bottom', horizontalalignment = 'right', bbox = props)
    # fig.savefig(os.path.join(args.figure_dir, 'subject_object_scatter.jpg'))
    fig.savefig(out_dir.joinpath('subject_object_scatter.jpg'))
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
    # r = torch.corrcoef(torch.stack((vo_filtered_verb_scores, vo_filtered_object_scores), dim = 0))[0, 1]
    r = np.corrcoef(torch.stack((vo_filtered_verb_scores, vo_filtered_object_scores), dim = 0).numpy())[0, 1]
    text = f'r = {float(r):.2f}'
    props = dict(boxstyle = 'round', facecolor = 'tab:orange', alpha = 0.85)
    ax.text(0.95, 0.05, text, transform = ax.transAxes, fontsize = 14, verticalalignment = 'bottom', horizontalalignment = 'right', bbox = props)
    # fig.savefig(os.path.join(args.figure_dir, 'verb_object_scatter.jpg'))
    fig.savefig(out_dir.joinpath('verb_object_scatter.jpg'))
    plt.close(fig)

def plot_p_ni(p_ni, test_id, out_dir, red_light_scores, post_red_base):
    plt.figure(figsize=(9, 5))
    plt.scatter(np.arange(p_ni.shape[0]), p_ni, s=15, alpha=0.6, label='P_n')
    
    ma = compute_moving_avg(p_ni, p_ni.shape[0])
    plt.plot(np.arange(ma.shape[0]), ma, color='gray', label='Moving Avg', alpha=0.4)
    plt.plot(np.arange(red_light_scores.shape[0]), red_light_scores, color='green', alpha=0.7, label='Red-light Score')
    # plt.vlines(60, 0, 1, 'black')
    if post_red_base is not None:
        plt.vlines(post_red_base, 0, 1, color='red', alpha=0.4, label='Detection')
    plt.xlabel('Image')
    plt.ylabel('P_ni')
    plt.title(f'P_ni, {test_id}')    
    plt.legend()
    plt.savefig(out_dir.joinpath(f'{test_id}_p_ni.jpg'))
    plt.close()
       
def plot_p_type(p_type_hist, test_id, out_dir):
    plt.figure(figsize=(9, 5))
    plt.plot(np.arange(len(p_type_hist)), [p[0] for p in p_type_hist], label='Type-1', marker='x', alpha=0.5)    
    plt.plot(np.arange(len(p_type_hist)), [p[1] for p in p_type_hist], label='Type-2', marker='o', color='red', alpha=0.5)    
    plt.plot(np.arange(len(p_type_hist)), [p[2] for p in p_type_hist], label='Type-3', marker='v', color='green', alpha=0.5)    
    plt.xlabel('Type-inference invocation')
    plt.ylabel('Probability')
    plt.title(f'P-type, {test_id}')    
    plt.legend()
    plt.savefig(out_dir.joinpath(f'{test_id}_p_type.jpg'))
    plt.close()
       
def plot(p):
    p_ni = None
    subject_novelty_scores = None
    verb_novelty_scores = None
    object_novelty_scores = None
    p_type_hist = None
    red_light_scores = None
    
    with open(p, 'rb') as handle:
        logs = pickle.load(handle)
        p_ni = logs['p_ni']
        subject_novelty_scores = logs['subj_novelty_scores_un']
        verb_novelty_scores = logs['verb_novelty_scores_un']
        object_novelty_scores = logs['obj_novelty_scores_un']
        p_type_hist = logs['p_type']
        red_light_scores = logs['red_light_scores']
        post_red_base = logs['post_red_base']
        
    plot_p_ni(p_ni, p.stem, p.parent, red_light_scores, post_red_base)
    plot_p_type(p_type_hist, p.stem, p.parent)
    plot_un(subject_novelty_scores, verb_novelty_scores, object_novelty_scores, p.parent)
         
if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument('--log-dir', default='./logs')
    
    args = p.parse_args()

    p = Path(args.log_dir)
    if not p.exists():
        raise Exception(f'log directory {p} was not found.')
    
    for log_file_path in p.glob('*/*.pkl'):
        plot(log_file_path)
    
    for log_file_path in p.glob('*/*.pkl'):
        log_file_path.unlink()
        
    with zipfile.ZipFile('plots.zip', mode='w', compression=zipfile.ZIP_DEFLATED) as zipped:
        for test_dir in p.glob('*'):
            if test_dir.is_dir():
                zipped.write(test_dir)
                
                for img in test_dir.glob('*.jpg'):
                    zipped.write(img)
                    img.unlink()

                for img in test_dir.glob('*.png'):
                    zipped.write(img)
                    img.unlink()
                    
    for test_dir in p.glob('*'):
        if test_dir.is_dir():
            test_dir.rmdir()
