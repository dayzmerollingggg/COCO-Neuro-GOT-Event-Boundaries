import os
import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.signal import fftconvolve
from nilearn.glm.first_level import spm_hrf, compute_regressor
import matplotlib.pyplot as plt

# --- Global Settings ---
TR = 2  # fMRI data has a TR of 2s
N_SCANS = 389  # 389 timepoints (389 * 2 = 778s total time)
FRAME_TIMES = np.arange(N_SCANS) * TR
TOTAL_MOVIE_SECONDS = N_SCANS * TR
HRF = spm_hrf(t_r=TR, time_length=32, oversampling=1)

# --- Utility Functions ---

def parse_timestamps(timestamps_txt):
    """Parses timestamps (MM:SS format) into total seconds."""
    parsed_seconds = []
    for line in timestamps_txt.strip().split('\n'):
        if not line:
            continue
        try:
            minutes, seconds = map(int, line.split(':'))
            total_seconds = minutes * 60 + seconds
            parsed_seconds.append(total_seconds)
        except ValueError:
            print(f"Warning: Skipping invalid timestamp format: '{line}'")
    return np.array(parsed_seconds)

def create_bool_and_count_arrays(timestamps_sec, end_time_sec=TOTAL_MOVIE_SECONDS, interval_duration_seconds=TR):
    """
    Creates boolean (1/0) and count arrays based on timestamps falling into 
    2-second intervals.
    """
    num_intervals = int(np.ceil((end_time_sec) / interval_duration_seconds))
    boolean_values = np.zeros(num_intervals, dtype=int)
    count_values = np.zeros(num_intervals, dtype=int)

    for ts_sec in timestamps_sec:
        if 0 <= ts_sec < end_time_sec:
            # Determine which 2-second interval the timestamp falls into
            interval_index = int(ts_sec // interval_duration_seconds)
            if 0 <= interval_index < num_intervals:
                boolean_values[interval_index] = 1
                count_values[interval_index] += 1
                
    return boolean_values, count_values

def save_convolved_regressor(regressor, file_path, file_name):
    """
    Computes convolution with HRF, saves the convolved data, and plots for 
    visual inspection.
    """
    single_array = regressor.flatten()
    # Convolve and truncate the result to the length of the original regressor
    convolved_data = fftconvolve(single_array, HRF)[:len(single_array)]

    # Save the convolved data
    np.save(file_path, convolved_data)

    # Plot (optional, uncomment if needed)
    # fig, ax_mag = plt.subplots(figsize=(12, 4))
    # ax_mag.plot(convolved_data)
    # ax_mag.set_title(f'Convolved {file_name} Signal')
    # ax_mag.set_xlabel('Time (2s intervals)')
    # ax_mag.set_ylabel('Amplitude')
    # plt.show()
    return convolved_data

# --- Regressor Functions ---

def run_regressor_analysis(amplitudes, file_name, duration_len, output_dir):
    """
    Computes, saves, and convolves the regressor using nilearn's compute_regressor.
    Handles both Boolean (unnormalized) and Amplitude (mean-centered) data.
    """
    # Create the events array (onsets, durations, amplitudes)
    onsets = np.arange(0, TOTAL_MOVIE_SECONDS, TR)
    onsets = onsets[:len(amplitudes)] # Ensure match length of amplitudes
    durations = [duration_len] * len(onsets)
    exp_condition = (onsets, durations, amplitudes)

    # 1. Compute the Regressor
    # This creates the standard predictor time series for the GLM
    regressor, _ = compute_regressor(
        exp_condition=exp_condition, 
        frame_times=FRAME_TIMES, 
        hrf_model='spm'
    )
    
    # 2. Save the Regressor (raw, unconvolved)
    regressor_df = pd.DataFrame(data=regressor, columns=[f'{file_name}'])
    regressor_df.to_csv(os.path.join(output_dir, f'{file_name}_regressor_raw.csv'), index=False)

    # 3. Save the Convolved Data
    convolved_data = save_convolved_regressor(
        regressor, 
        os.path.join(output_dir, f'{file_name}_convolved.npy'), 
        file_name
    )
    
    # 4. Save the Boolean/Count array (for verification/storage)
    bool_df = pd.DataFrame({'Boolean_Count_Value': amplitudes})
    bool_df.to_csv(os.path.join(output_dir, f'{file_name}_intervals.csv'), index=False)
    
    return convolved_data

# --- Main Processing Function ---

def process_timestamp_file(timestamps_txt, base_name, duration_len):
    """
    Main function to read timestamps, create Boolean/Count arrays,
    apply time shifts, and run regressor analysis for all groups.
    """
    print(f"Processing data for: {base_name}")
    
    # 1. Parse and Define Base Data
    base_timestamps = parse_timestamps(timestamps_txt)
    base_bool, base_count = create_bool_and_count_arrays(base_timestamps)
    
    # Define groups for processing
    groups = {
        'base': {
            'timestamps': base_timestamps,
            'bool': base_bool,
            'count': base_count,
            'shift': 0,
            'dir_name': f'{base_name}_base'
        },
        'shift_2s': {
            # Shift timestamps by adding 2 seconds
            'timestamps': base_timestamps + 2, 
            'bool': None, # Will be created
            'count': None, # Will be created
            'shift': 2,
            'dir_name': f'{base_name}_shift_2s'
        },
        'shift_4s': {
            # Shift timestamps by adding 4 seconds
            'timestamps': base_timestamps + 4, 
            'bool': None, # Will be created
            'count': None, # Will be created
            'shift': 4,
            'dir_name': f'{base_name}_shift_4s'
        }
    }
    
    # 2. Process each shift group
    for group_key, group_data in groups.items():
        dir_name = group_data['dir_name']
        output_dir = os.path.join('regressor_output', dir_name)
        
        # Create output folder
        os.makedirs(output_dir, exist_ok=True)
        
        # Recalculate bool/count for shifted timestamps
        if group_key != 'base':
             # Re-run the bool/count array creation for the new, shifted timestamps
            group_data['bool'], group_data['count'] = create_bool_and_count_arrays(group_data['timestamps'])
            
        bool_values = group_data['bool']
        count_values = group_data['count']
        
        # --- Run Regressor Bool (Presence/Absence) ---
        bool_file_name = f'{dir_name}_bool'
        print(f"  -> Running Regressor Bool for: {bool_file_name}")
        run_regressor_analysis(bool_values, bool_file_name, duration_len, output_dir)
        
        # --- Run Regressor Amplitude (Mean-Centered Count) ---
        amp_file_name = f'{dir_name}_amp'
        # Mean center the count data
        count_values_centered = count_values - np.mean(count_values)
        print(f"  -> Running Regressor Amp for: {amp_file_name}")
        run_regressor_analysis(count_values_centered, amp_file_name, duration_len, output_dir)
        
    print(f"Processing complete for {base_name}. Outputs are in the 'regressor_output' folder.")


# --- Main Execution Block ---

if __name__ == '__main__':
    # Make the top-level output directory
    os.makedirs('regressor_output', exist_ok=True)

    # 1. Read the input files
    try:
        with open('daisy_timestamps.txt', 'r') as f:
            my_timestamps_txt = f.read()
        with open('daisy_scene_changes.txt', 'r') as f:
            my_scene_changes_txt = f.read()
        with open('medium_length.txt', 'r') as f:
            medium_txt = f.read()
    except FileNotFoundError as e:
        print(f"Error: Required input file not found: {e}")
        exit()

    # 2. Process all data groups
    # The duration is passed as a parameter (2s here)
    process_timestamp_file(my_timestamps_txt, 'camera_cuts', 2)
    print("-" * 30)
    process_timestamp_file(my_scene_changes_txt, 'scene_cuts', 2)
    print("-" * 30)
    process_timestamp_file(medium_txt, 'medium_length', 2)