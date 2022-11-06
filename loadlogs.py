import pickle
import numpy as np

filename = './logs/OND.10711.000/OND.10711.000.pkl'

print(filename)
with open(filename, 'rb') as f:
    data = pickle.load(f)

p_type_6_7 = np.array(data['p_type_6_7'])
print(p_type_6_7.argmax(1))
