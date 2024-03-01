import os
import pandas as pd
import numpy as np
import pickle

def find_matching_file(folder, filename_to_match, match_length=24):
    """
    Find a file in the specified folder where the last 'match_length' characters of the filename match.
    """
    for file in os.listdir(folder):
        if file[-match_length:] == filename_to_match[-match_length:]:
            return os.path.join(folder, file)
    return None

def interpolate_and_save(df1, df2, base_folder, filename, start_row=500, end_row=2000, total_images=1500):
    """
    Interpolate between two dataframes for specific rows and retain the specified rows from df1 unchanged,
    focusing only on numeric columns for interpolation.
    """
    # Ensure operations are only performed on numeric columns
    numeric_columns = df1.select_dtypes(include=np.number).columns
    non_numeric_columns = df1.select_dtypes(exclude=np.number).columns

    interpolated_df = pd.DataFrame(index=df1.index)
    
    # Copy non-numeric data as is
    interpolated_df[non_numeric_columns] = df1[non_numeric_columns]

    # Interpolate only numeric columns
    for index in range(start_row, min(end_row, df1.shape[0]) + 1):
        if index <= total_images + start_row - 1:  # Adjust for the offset of start_row
            lambda_i = (index - start_row) / total_images
        else:
            lambda_i = 1
        # import ipdb; ipdb.set_trace()
        interpolated_df.loc[index-1, numeric_columns] = df1.loc[index-1, numeric_columns] * (1 - lambda_i) + df2.loc[index-1, numeric_columns] * lambda_i
        # print(df1.loc[index-1, numeric_columns] * (1 - lambda_i), df2.loc[index-1, numeric_columns] * lambda_i)
    # Keep the first 500 rows and rows after 2000 from df1 unchanged for numeric columns
    for col in numeric_columns:
        interpolated_df.loc[:start_row-2, col] = df1.loc[:start_row-2, col]
        interpolated_df.loc[end_row:, col] = df1.loc[end_row:, col]

    # Save the interpolated dataframe
    folder_name = os.path.join(base_folder, 'OND/image_classification/')
    os.makedirs(folder_name, exist_ok=True)
    output_filename = os.path.join(folder_name, filename)
    interpolated_df.to_csv(output_filename, index=False)
    print(f'Saved: {output_filename}')



def interpolate_folders(folder1, folder2, output_base_folder):
    """
    Process all files from folder1, find matching files in folder2, and perform interpolation.
    """
    for filename in os.listdir(folder1):
        file1_path = os.path.join(folder1, filename)
        file2_path = find_matching_file(folder2, filename)

        if file2_path:
            df1 = pd.read_csv(file1_path)
            df2 = pd.read_csv(file2_path)
            interpolate_and_save(df1, df2, output_base_folder, filename)
        else:
            print(f"No matching file found for {filename}")

def interpolate_values(arr1, arr2, start=500, end=2000, total_images=1500):
    """Interpolate values between two arrays for indices from start to end."""
    interpolated = arr1.copy()  # Start with a copy of the first array

    for i in range(start, min(end + 1, arr1.shape[0])):  
        index_within_range = i - start
        if index_within_range < total_images:
            lambda_i = index_within_range / total_images
        else:
            lambda_i = 1  # Use full value from second array beyond the specified range
        interpolated[i] = arr1[i] * (1 - lambda_i) + arr2[i] * lambda_i
        # print(f'interpolated {interpolated[i]}  No Retraing Org {arr1[i]} No Retraing Updated {arr1[i] * (1 - lambda_i)} , EWC Updated {arr2[i] * lambda_i} EWC Ori {arr2[i]}')
        print(f'{i} ) interpolated {interpolated[i]:.4f}  No Retraing Org {arr1[i]:.4f} No Retraing Updated {arr1[i] * (1 - lambda_i):.4f} , EWC Updated {arr2[i] * lambda_i:.4f} EWC Ori {arr2[i]:.4f}')

    return interpolated

def interpolate_per_box_predictions(dict1, dict2, start=500, end=2000):
    """Interpolate 'species_probs' and 'activity_probs' within 'per_box_predictions'."""
    interpolated_dict = {}
    for key in dict1:
        if key in dict2:  # Ensure the key exists in both dictionaries
            interpolated_dict[key] = {}
            for subkey in ['species_probs', 'activity_probs']:
                arr1 = dict1[key][subkey]
                arr2 = dict2[key][subkey]
                # Ensure interpolation is only applied to the specified indices
                interpolated_dict[key][subkey] = interpolate_values(arr1, arr2, start, end)
    return interpolated_dict

# Adjust the interpolate_and_save_pickle function accordingly
def interpolate_and_save_pickle(file1_path, file2_path, output_path):
    with open(file1_path, 'rb') as f:
        data1 = pickle.load(f)
    with open(file2_path, 'rb') as f:
        data2 = pickle.load(f)

    interpolated_data = {}
    for key in ['p_ni', 'p_ni_raw', 'per_img_p_type', 'red_light_scores']:
        # Handle both 1D and 2D arrays appropriately
        if data1[key].ndim == 1:
            interpolated_data[key] = interpolate_values(data1[key], data2[key])
        elif data1[key].ndim == 2:
            interpolated_data[key] = np.array([interpolate_values(data1[key][:, j], data2[key][:, j]) for j in range(data1[key].shape[1])]).T

    interpolated_data['per_box_predictions'] = interpolate_per_box_predictions(
        data1['per_box_predictions'], data2['per_box_predictions'], start=500, end=2000
    )

    # Copy other keys without interpolation
    for key in ['characterization', 'queries']:
        interpolated_data[key] = data1[key]

    # import ipdb; ipdb.set_trace()
    # folder_name = os.path.join(output_path, 'logs')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(interpolated_data, f)


folder1 = '/nfs/hpc/share/sail_on3/TestsForPaper/Jan2024_Corrected_Normalization/NoRetraining/OND/image_classification/'
folder2 = '/nfs/hpc/share/sail_on3/TestsForPaper/Jan2024_Corrected_Normalization/EWC_logit_layer_only_100000_1e-4_forInterpolation/OND/image_classification/'
output_base_folder = '/nfs/hpc/share/sail_on3/TestsForPaper/Jan2024_Corrected_Normalization/Interpolation_NoRetraining_EWC_Only_logits/'
interpolate_folders(folder1, folder2, output_base_folder)


base_folder1 = '/nfs/hpc/share/sail_on3/TestsForPaper/Jan2024_Corrected_Normalization/NoRetraining/logs/'
base_folder2 = '/nfs/hpc/share/sail_on3/TestsForPaper/Jan2024_Corrected_Normalization/EWC_logit_layer_only_100000_1e-4_forInterpolation/logs/'
output_base_folder = '/nfs/hpc/share/sail_on3/TestsForPaper/Jan2024_Corrected_Normalization/Interpolation_NoRetraining_EWC_Only_logits/'

for subfolder in os.listdir(base_folder1):
    if subfolder in os.listdir(base_folder2):
        print(subfolder)
        file_name = os.listdir(os.path.join(base_folder1, subfolder))[0]
        file1_path = os.path.join(base_folder1, subfolder, file_name)
        file2_path = os.path.join(base_folder2, subfolder, file_name)
        output_path = os.path.join(output_base_folder,'logs', subfolder, file_name)
        interpolate_and_save_pickle(file1_path, file2_path, output_path)
