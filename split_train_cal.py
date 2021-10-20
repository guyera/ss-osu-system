import pandas as pd

train = pd.read_csv('./Custom/annotations/val_dataset_v1_train.csv', index_col=0)
groups = train.groupby(['subject_id', 'object_id', 'verb_id'])

NUM_SAMPLES = train.shape[0]
SPLIT = 0.75
NUM_UNIQUE_TUPLES = len(groups)

group_sizes = [g[1].shape[0] for g in groups]
keys = [k for k in groups.indices.keys()]

n = int(SPLIT * group_sizes[0])
new_train = groups.get_group(keys[0]).iloc[:n]
cal = groups.get_group(keys[0]).iloc[n:]

for i in range(1, len(keys)):
    n = int(SPLIT * group_sizes[i])
    new_train = new_train.append(groups.get_group(keys[i]).iloc[:n])
    cal = cal.append(groups.get_group(keys[i]).iloc[n:])

new_train_groups = new_train.groupby(['subject_id', 'object_id', 'verb_id'])
cal_groups = cal.groupby(['subject_id', 'object_id', 'verb_id'])

assert new_train.shape[0] + cal.shape[0] == train.shape[0]
assert new_train.shape[1] == train.shape[1]
assert cal.shape[1] == train.shape[1]
assert set(list(new_train_groups.indices.keys())) == set(keys)
assert set(list(cal_groups.indices.keys())) == set(keys)

new_train.sort_index(inplace=True)
cal.sort_index(inplace=True)

new_train.to_csv('./Custom/annotations/val_dataset_v1_new_train.csv')
cal.to_csv('./Custom/annotations/val_dataset_v1_cal.csv')