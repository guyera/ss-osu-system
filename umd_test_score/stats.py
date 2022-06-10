"""
Collect statistics about our system's answers.
"""

from typing import NamedTuple
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict
from pathlib import Path
import seaborn as sns

# This is now a command-line option
# dataset_root = Path("/nfs/raid92/u10/lramshaw/symbiant/umd/")

elems = ["S", "O", "V-S", "V-SO"]

labels = {}
labels["S"]= {-1: 'absent',
              0: 'novel',
              1: 'person',
              2: 'dog',
              3: 'cat',
              4: 'horse',}
              #5: "new_val",}

labels["O"] = {-1: "absent",
               0: "novel",
               1: "frisbee",
               2: "bag",
               3: "bicycle",
               4: "tie",
               5: "umbrella",
               6: "tv",
               7: "carriage",
               8: "hay",
               9: "dog",
               10: "cat",
               11: "horse"}

labels["V"] = {0: "novel-unk",
               1: "carrying",
               2: "wearing",
               3: "riding",
               4: "catching",
               5: "pulling",
               6: "eating",
               7: "watching"}

def names_S (ids):
    return [labels["S"][id] for id in ids]

def names_O(ids):
    return [labels["O"][id] for id in ids]

def names_V(ids):
    return [labels["V"][id] for id in ids]

cat_names = {'S': labels["S"].values(),
             'O': labels["O"].values(),
             'V': labels["V"].values()}

class Triple(NamedTuple):
    S: int
    O: int
    V: int

class Instance:
    def __init__(self, corr_S, corr_O, corr_V, pred_1_S, pred_1_O, pred_1_V,
                 pred_2_S, pred_2_O, pred_2_V, pred_3_S, pred_3_O, pred_3_V, image_path):
        self.corr = Triple(corr_S, corr_O, corr_V)
        self.pred_1 = Triple(pred_1_S, pred_1_O, pred_1_V)
        self.pred_2 = Triple(pred_2_S, pred_2_O, pred_2_V)
        self.pred_3 = Triple(pred_3_S, pred_3_O, pred_3_V)
        self.image_path = image_path

class Stats:
    def __init__(self):
        self.init()

    def init(self):
        self.instances = []
        self.corrs_S = []
        self.corrs_O = []
        self.corrs_V = []
        self.preds_1_S = []
        self.preds_1_O = []
        self.preds_1_V = []
        self.preds_2_S = []
        self.preds_2_O = []
        self.preds_2_V = []
        self.preds_3_S = []
        self.preds_3_O = []
        self.preds_3_V = []
        self.images = {}
        for elem in elems:
            self.images[elem] = defaultdict(set)

    def add(self, instance):
        self.instances.append(instance)
        self.corrs_S.append(instance.corr.S)
        self.corrs_O.append(instance.corr.O)
        self.corrs_V.append(instance.corr.V)
        self.preds_1_S.append(instance.pred_1.S)
        self.preds_1_O.append(instance.pred_1.O)
        self.preds_1_V.append(instance.pred_1.V)
        self.preds_2_S.append(instance.pred_2.S)
        self.preds_2_O.append(instance.pred_2.O)
        self.preds_2_V.append(instance.pred_2.V)
        self.preds_3_S.append(instance.pred_3.S)
        self.preds_3_O.append(instance.pred_3.O)
        self.preds_3_V.append(instance.pred_3.V)
        self.images["S"][(instance.corr.S, instance.pred_1.S)].add(instance.image_path)
        self.images["O"][(instance.corr.O, instance.pred_1.O)].add(instance.image_path)
        if instance.corr.O == -1:
            self.images["V-S"][(instance.corr.V, instance.pred_1.V)].add(instance.image_path)
        else:
            self.images["V-SO"][(instance.corr.V, instance.pred_1.V)].add(instance.image_path)


    def confusion_matrices_1(self, normalize):
        if normalize:
            norm = "true"
        else:
            norm = None
        matrix_S = confusion_matrix(names_S(self.corrs_S), names_S(self.preds_1_S), normalize=norm,
                                    labels=["absent", "novel", "person", "dog", "cat", "horse"])
        matrix_O = confusion_matrix(names_O(self.corrs_O), names_O(self.preds_1_O), normalize=norm,
                                    labels=["absent", "novel", "frisbee", "bag", "bicycle", "tie",
                                            "umbrella", "tv", "carriage", "hay", "dog", "cat", "horse"])
        matrix_V = confusion_matrix(names_V(self.corrs_V), names_V(self.preds_1_V), normalize=norm,
                                    labels=["novel/unk", "carrying", "wearing", "riding",
                                            "catching", "pulling", "eating", "watching"])
        return matrix_S, matrix_O, matrix_V

    def print_confusion_matrices_1(self, out_file):
        norm_matrices = self.confusion_matrices_1(True)
        count_matrices = self.confusion_matrices_1(False)
        with PdfPages(out_file) as pdf_pages:
            for i, cat in enumerate(['S', 'O', 'V']):
                # print normalized version
                fig = plt.figure(cat, figsize=[11, 8])
                ax = plt.axes()
                sns.heatmap(norm_matrices[i],
                            annot=True,
                            fmt="0.2g",
                            cmap='Blues',)
                ax.xaxis.set_ticklabels(cat_names[cat])
                ax.yaxis.set_ticklabels(cat_names[cat])
                plt.title(f"{cat} matrix, normalized by row")
                pdf_pages.savefig()
                plt.close()
                # Print count version
                fig = plt.figure(cat, figsize=[11, 8])
                ax = plt.axes()
                sns.heatmap(count_matrices[i],
                            annot=True,
                            fmt="d",
                            cmap='Blues',)
                ax.xaxis.set_ticklabels(cat_names[cat])
                ax.yaxis.set_ticklabels(cat_names[cat])
                plt.title(f"{cat} matrix with counts")
                pdf_pages.savefig()
                plt.close()

    def save_image_paths(self, image_dir, dataset_root):
        image_dir.mkdir(exist_ok=True)
        for elem in elems:
            elem_dir = image_dir / elem
            elem_dir.mkdir(exist_ok=True)
            prev_corr = prev_pred = -2
            corr_dir = pred_dir = None
            for corr_pred in sorted(list(self.images[elem].keys())):
                corr, pred = corr_pred
                if corr != prev_corr:
                    corr_dir = elem_dir / f"true_{corr:02d}_{labels[elem[0]][corr]}"
                    corr_dir.mkdir(exist_ok=True)
                    prev_corr = corr
                    prev_pred = -2
                if pred != prev_pred:
                    pred_dir = corr_dir / f"pred_{pred:02d}_{labels[elem[0]][pred]}"
                    pred_dir.mkdir(exist_ok=True)
                    prev_pred = pred
                for image_str in self.images[elem][corr_pred]:
                    image_path = dataset_root / image_str
                    symlink_path = pred_dir / image_path.name
                    symlink_path.symlink_to(image_path)
