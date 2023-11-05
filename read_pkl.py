import pickle

# Replace 'yourfile.pkl' with the path to the actual .pkl file you want to read
file_path = 'yourfile.pkl'

# Open the file in binary read mode
with open(file_path, 'rb') as file:
    # Use pickle to load the file
    data = pickle.load(file)

# Now you can check what's inside the data
print(type(data))  # This will print the type of the data object

# Depending on the type of the data, you can inspect it differently
# For example, if it's a list or a dictionary, you can iterate over it

if isinstance(data, dict):
    # It's a dictionary, let's print its keys and data types of its values
    for key, value in data.items():
        print(f'Key: {key}, Value type: {type(value)}')

elif isinstance(data, list):
    # It's a list, let's print the type of each item
    for index, value in enumerate(data):
        print(f'Index: {index}, Value type: {type(value)}')

elif isinstance(data, (int, float, str)):
    # It's a simple data type
    print(data)

else:
    # It's something else, let's print it directly
    print(data)
