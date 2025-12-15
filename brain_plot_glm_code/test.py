import os

file_path = '/mnt/labdata/got_project/test/brain_plot_data_output/regressor_output/scene_cuts_base/scene_cuts_bool_regressor_raw.csv'

# 1. Print the path you are trying to use
print(f"Attempting to check path: {file_path}")

# 2. Check if the file exists
if os.path.exists(file_path):
    print("SUCCESS: os.path.exists() returns TRUE.")
    
    # 3. Try to read the file (e.g., using pandas or simple open)
    try:
        with open(file_path, 'r') as f:
            # You can add a print here if you want to confirm the file can be opened
            # print("Successfully opened the file.")
            pass
        # Execute your actual read operation here, e.g., pd.read_csv(file_path)
    except Exception as e:
        print(f"An error occurred during file open/read: {e}")
else:
    print("FAILURE: os.path.exists() returns FALSE. The path is incorrect or inaccessible.")

# The rest of your script...