# This software was created by Oregon State University under Army Research
# Office (ARO) Award Number W911NF-22-2-0149. ARO, as the Federal awarding
# agency, reserves a royalty-free, nonexclusive and irrevocable right to
# reproduce, publish, or otherwise use this software for Federal purposes, and
# to authorize others to do so in accordance with 2 CFR 200.315(b).

import pandas as pd
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, save_image


last30CSV =  '../session/tests/svo_10_70_last_20_novel/test_trials_csv_10/api_tests/OND/svo_classification/OND.101.000_single_df.csv'
last30_lines = pd.read_csv(last30CSV)[-29:]

# import ipdb; ipdb.set_trace()
for pos, row in last30_lines.iterrows():
    img = read_image('../'+row['new_image_path'])
    print(row['subject_name'])
    if row['subject_name'] != 'absent':
        s_xmin, s_ymin, s_xmax, s_ymax = row['subject_xmin'], row['subject_ymin'], row['subject_xmax'], row['subject_ymax']
    else: 
        s_xmin, s_ymin, s_xmax, s_ymax = row['object_xmin'], row['object_ymin'], row['object_xmax'], row['object_ymax']
    
    o_xmin, o_ymin, o_xmax, o_ymax = row['object_xmin'], row['object_ymin'], row['object_xmax'], row['object_ymax']
    
    v_xmin = min(s_xmin, o_xmin)
    v_ymin = min(s_ymin, o_ymin)
    v_xmax = max(s_xmax, o_xmax)
    v_ymax = max(s_ymax, o_ymax)

    boxes = [(s_xmin, s_ymin, s_xmax, s_ymax), (o_xmin, o_ymin, o_xmax, o_ymax),  (v_xmin, v_ymin, v_xmax, v_ymax)]

    print(boxes)
    boxes = torch.tensor(boxes)
    boxes = boxes.unsqueeze(0)
    img = draw_bounding_boxes(img, boxes, width=5,
                          colors=["green","yellow", "blue"], 
                          fill=False)
    img = torchvision.transforms.ToPILImage()(img)

    save_image(img,row['new_image_path'].split('/')[-1])
