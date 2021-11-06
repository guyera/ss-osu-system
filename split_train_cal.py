import pandas as pd
import pickle

# train = pd.read_csv('./Custom/annotations/val_dataset_v1_train.csv', index_col=0)
train = pd.read_csv('./dataset_v3/dataset_v3_2_train.csv', index_col=0)
original_N_train = train.shape[0]

train = train[(train['subject_id'] != 0) & (train['verb_id'] != 0) & (train['object_id'] != 0)]
groups = train.groupby(['subject_id', 'verb_id', 'object_id'])

NUM_TRAIN_SAMPLES = train.shape[0]
SPLIT = 0.75
NUM_UNIQUE_TUPLES = len(groups)

print(f'removed {original_N_train - NUM_TRAIN_SAMPLES} training samples which had 0 labels.')

keys = [k for k in groups.indices.keys() if groups.get_group(k).shape[0] > 2]
group_sizes = [groups.get_group(k).shape[0] for k in keys]

n = int(SPLIT * group_sizes[0])
new_train = groups.get_group(keys[0]).iloc[:n]
cal = groups.get_group(keys[0]).iloc[n:]

for i in range(1, len(keys)):
    n = int(SPLIT * group_sizes[i])
    new_train = new_train.append(groups.get_group(keys[i]).iloc[:n])
    cal = cal.append(groups.get_group(keys[i]).iloc[n:])

new_train_groups = new_train.groupby(['subject_id', 'verb_id', 'object_id'])
cal_groups = cal.groupby(['subject_id', 'verb_id', 'object_id'])

assert new_train.shape[1] == train.shape[1]
assert cal.shape[1] == train.shape[1]
assert set(list(new_train_groups.indices.keys())) == set(list(cal_groups.indices.keys()))

new_train.sort_index(inplace=True)
cal.sort_index(inplace=True)

new_train.to_csv('./dataset_v3/dataset_v3_2_train_modified.csv')
cal.to_csv('./dataset_v3/dataset_v3_2_cal.csv')

with open('./ensemble/train_tuples.pkl', 'wb') as handle:
    results = {}
    results['train_tuples'] = list(new_train_groups.indices.keys())

    pickle.dump(results, handle)

# validation
val = pd.read_csv('./dataset_v3/dataset_v3_2_val.csv', index_col=0)
val = val[(val['subject_id'] != 0) & (val['verb_id'] != 0) & (val['object_id'] != 0)]
original_val_size = val.shape[0]
print(f'removed {original_val_size - val.shape[0]} val samples which had 0 labels.')

val_groups = val.groupby(['subject_id', 'verb_id', 'object_id'])
val_keys = [k for k in val_groups.indices.keys() if k in keys]

new_val = val_groups.get_group(val_keys[0])
for k in val_keys[1:]:
    new_val = new_val.append(val_groups.get_group(k))

print(f'removed {original_val_size - new_val.shape[0]} val samples which had different label than the ones in training.')

new_val.to_csv('./dataset_v3/dataset_v3_2_val_modified.csv')