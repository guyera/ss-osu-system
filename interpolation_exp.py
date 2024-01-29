import os
import pandas as pd
import numpy as np

def find_matching_file(folder, filename_to_match, match_length=24):
    """
    Find a file in the specified folder where the last 'match_length' characters of the filename match.
    """
    for file in os.listdir(folder):
        if file[-match_length:] == filename_to_match[-match_length:]:
            return os.path.join(folder, file)
    return None

def interpolate_and_save(df1, df2, base_folder, filename, increment=0.1):
    """
    Interpolate between two dataframes and save the output in specified folders.
    Assumes the first column should not be interpolated.
    """
    lambda_values = np.arange(0, 1 + increment, increment)

    # Separate the first column which should not be interpolated
    first_column_df1 = df1.iloc[:, 0]
    numeric_columns_df1 = df1.iloc[:, 1:]
    numeric_columns_df2 = df2.iloc[:, 1:]

    for lambda_val in lambda_values:
        # Interpolate only the numeric columns
        interpolated_numeric_df = numeric_columns_df1 * (1 - lambda_val) + numeric_columns_df2 * lambda_val

        # Reattach the first column
        interpolated_df = pd.concat([first_column_df1, interpolated_numeric_df], axis=1)

        # Create folder and save file
        folder_name = os.path.join(base_folder, f'lambda_{lambda_val:.1f}')
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

# Example usage
folder1 = '/nfs/hpc/share/sail_on3/TestsForPaper/Jan2024_Corrected_Normalization/NoRetraining/OND/image_classification/'
folder2 = '/nfs/hpc/share/sail_on3/TestsForPaper/Jan2024_Corrected_Normalization/Oracle_100budget_EWC/OND/image_classification/'
output_base_folder = '/nfs/hpc/share/sail_on3/TestsForPaper/Jan2024_Corrected_Normalization/'
interpolate_folders(folder1, folder2, output_base_folder)
