import os
import pandas as pd

def load_group_csvs(base_name, shift_type='base', file_type='regressor_raw'):

    # 1. Construct the folder name
    # Example: 'camera_cuts_shift_2s'
    dir_name = f'{base_name}_{shift_type}'
    group_dir = os.path.join('regressor_output', dir_name)

    # 2. Define the specific file names for bool and amplitude
    bool_file_name = f'{dir_name}_bool_{file_type}.csv'
    amp_file_name = f'{dir_name}_amp_{file_type}.csv'

    # 3. Construct the full file paths
    bool_path = os.path.join(group_dir, bool_file_name)
    amp_path = os.path.join(group_dir, amp_file_name)

    # 4. Load the data
    data = {}

    # Load Boolean data
    if os.path.exists(bool_path):
        data[f'{base_name}_bool'] = pd.read_csv(bool_path)
        print(f"Loaded: {bool_path}")
    else:
        print(f"File not found: {bool_path}")

    # Load Amplitude data
    if os.path.exists(amp_path):
        data[f'{base_name}_amp'] = pd.read_csv(amp_path)
        print(f"Loaded: {amp_path}")
    else:
        print(f"File not found: {amp_path}")

    return data

# --- Example Usage (Assuming you run this AFTER the main script) ---

# Load the raw regressor CSVs for the camera_cuts, shifted by 4 seconds
camera_cuts_4s_regressors = load_group_csvs(
    base_name='camera_cuts',
    shift_type='shift_4s',
    file_type='regressor_raw'
)

print("\n--- Loaded Data Check ---")
if camera_cuts_4s_regressors:
    # Print the first few rows of the loaded Boolean Regressor
    print(f"\nDataFrame Keys: {list(camera_cuts_4s_regressors.keys())}")
    print("\nFirst 5 rows of the Boolean Regressor:")
    print(camera_cuts_4s_regressors['camera_cuts_bool'].head())