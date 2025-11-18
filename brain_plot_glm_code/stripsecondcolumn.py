import pandas as pd
import os
import sys

def combine_second_columns(input_dir, output_filepath):
    """
    Reads all CSV files in a directory, extracts the second column (index 1)
    from each, and saves the combined columns to a single output CSV file.

    Args:
        input_dir (str): The directory containing the source CSV files.
        output_filepath (str): The full path and filename for the output CSV.
    """
    
    # 1. Initialize a list to hold the second column from each file
    all_columns_data = {}
    
    # 2. Get a list of all CSV files in the input directory
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"Error: No CSV files found in directory: {input_dir}")
        return

    print(f"Found {len(csv_files)} CSV files. Combining second column from each...")

    # 3. Iterate through each CSV file
    for filename in csv_files:
        file_path = os.path.join(input_dir, filename)
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Check if the file has at least two columns (index 1 is the second column)
            if df.shape[1] > 1:
                # Extract the second column (index 1). We use iloc[:, 1]
                # to select all rows and the column at index 1.
                column_data = df.iloc[:, 1]
                
                # Use the original filename (without .csv) as the column header
                column_name = filename.replace('.csv', '')
                all_columns_data[column_name] = column_data
                print(f"  -> Successfully extracted column from: {filename}")
            else:
                print(f"  -> Skipping {filename}: file has fewer than two columns.")

        except pd.errors.EmptyDataError:
            print(f"  -> Skipping {filename}: file is empty.")
        except Exception as e:
            print(f"  -> An error occurred processing {filename}: {e}")

    # 4. Create the final DataFrame from the collected columns
    if not all_columns_data:
         print("No data was collected. Exiting.")
         return

    combined_df = pd.DataFrame(all_columns_data)
    
    # 5. Save the combined DataFrame to the specified output file
    try:
        combined_df.to_csv(output_filepath, index=False)
        print("\nProcess Complete!")
        print(f"Combined data successfully saved to: {output_filepath}")
        print(f"Total columns saved: {combined_df.shape[1]}")
    except Exception as e:
        print(f"Error saving output file: {e}")

if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: Update these paths before running the script.
    
    # 1. Directory where your source CSV files (e.g., 'file1.csv', 'file2.csv') are located.
    #    Example: '/path/to/my/regressor/data'
    INPUT_DIRECTORY = '/mnt/labdata/got_project/daisy/data/regressors_combined' 
    
    # 2. The full path and name for the output file that will contain the combined columns.
    #    Example: '/path/to/my/output/combined_features.csv'
    OUTPUT_FILE = '/mnt/labdata/got_project/daisy/data/combined_regressors.csv'
    

    combine_second_columns(INPUT_DIRECTORY, OUTPUT_FILE)