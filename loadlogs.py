import pickle
import numpy as np
import os

logs_dir = './logs_NoFeedback'

trials = os.listdir(logs_dir)

for trial in trials:
    filename = logs_dir +'/'+ trial + '/' + trial +'.pkl'
    print(filename)
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    p_type_6_7 = np.array(data['p_type_6_7'][1:][-1])
    print(p_type_6_7.argmax())

# per_image_p_type_6_7 = np.array(data['per_image_p_type_separated6_7'])
# print(per_image_p_type_6_7.argmax(1))



# for key in data.keys():
#     print(key)
#     xx =  data[key]
 

# import ipdb; ipdb.set_trace()

