import os
import pandas as pd

def merge_selected_columns(folder_path, output_file):
    """
    Reads all CSV files in a folder, extracts 'image_path' and 'annotation' columns,
    merges them into a single DataFrame, and saves to a specified output file.

    Args:
        folder_path (str): Path to the folder containing CSV files.
        output_file (str): Path to save the merged CSV file.
    """
    # List to hold each filtered CSV DataFrame
    csv_list = []
    
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            try:
                # Read each CSV file
                csv_data = pd.read_csv(file_path)
                
                # Check if required columns exist, then filter
                if 'image_path' in csv_data.columns and 'Annotation' in csv_data.columns:
                    csv_filtered = csv_data[['image_path', 'Annotation']]
                    csv_list.append(csv_filtered)
                else:
                    print(f"Skipping {filename}: required columns not found.")
            except Exception as e:
                print(f"Could not read {filename}: {e}")
    
    # Concatenate all filtered CSVs into a single DataFrame
    combined_csv = pd.concat(csv_list, ignore_index=True)
    
    # Save combined DataFrame to output CSV
    combined_csv.to_csv(output_file, index=False)
    print(f"Merged data saved to {output_file}")

# Example usage
folder_path = '/mnt/storage1/Jin/part1'
output_file = '/mnt/storage1/Jin/melfusion/data.csv'
merge_selected_columns(folder_path, output_file)
