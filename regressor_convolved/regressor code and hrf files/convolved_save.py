import numpy as np
import os
import matplotlib.pyplot as plt
from nilearn.glm.first_level import spm_hrf
from scipy.signal import fftconvolve
from scipy.stats import zscore
# Import warnings for cleaner execution if files are missing
import warnings

# --- Utility Functions for Saving ---

def save_plot(fig, plot_name, save_dir="plots"):
    """
    Saves a Matplotlib figure to a specified directory.
    
    Args:
        fig (matplotlib.figure.Figure): The figure object to save.
        plot_name (str): The base name for the file (e.g., "camera_regressor").
        save_dir (str): The directory to save the file in.
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    file_path = os.path.join(save_dir, f"{plot_name}.png")
    fig.savefig(file_path, bbox_inches='tight')
    plt.close(fig) # Close the figure to free up memory
    print(f"Saved plot: {file_path}")

def convolve_and_save_npy(regressor, hrf, base_name, save_dir="data"):
    """
    Performs convolution and z-scoring, then saves the results as .npy files.

    Args:
        regressor (np.ndarray): The stimulus regressor array.
        hrf (np.ndarray): The HRF kernel.
        base_name (str): Base name for the output files (e.g., 'camera', 'scene').
        save_dir (str): The directory to save the files in.

    Returns:
        tuple: (convolved_signal, z_scored_signal)
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Convolution (Truncate to original regressor length)
    convolved = fftconvolve(regressor, hrf)[:len(regressor)]
    
    # 2. Z-scoring
    convolved_z = zscore(convolved)
    
    # 3. Saving
    np.save(os.path.join(save_dir, f'convolved_{base_name}.npy'), convolved)
    np.save(os.path.join(save_dir, f'convolved_z_{base_name}.npy'), convolved_z)
    
    print(f"Saved convolved and z-scored data for '{base_name}' to {save_dir}/")
    
    return convolved, convolved_z

# --- Data Processing and Plotting Functions (Modified) ---

def arrange_timestamp_data(timestamps_txt, camOrScene):
    # --- 1. Parse Timestamps ---
    parsed_seconds = []
    for line in timestamps_txt.strip().split('\n'):
        if not line:
            continue
        try:
            # Assuming 'M:S' format. If it's just 'S', handle it
            if ':' in line:
                minutes, seconds = map(int, line.split(':'))
                total_seconds = minutes * 60 + seconds
            else:
                total_seconds = int(line.strip())
            parsed_seconds.append(total_seconds)
        except ValueError:
            warnings.warn(f"Warning: Skipping invalid timestamp format: '{line}'", UserWarning)
            continue

    # --- 2. Define Time Range and Intervals ---
    start_time_seconds = 0
    end_time_minutes = 13
    end_time_seconds = end_time_minutes * 60
    interval_duration_seconds = 2 # TR

    num_intervals = int(np.ceil((end_time_seconds - start_time_seconds) / interval_duration_seconds))
    
    # --- 3. Initialize Graph Data Structures ---
    boolean_values = np.zeros(num_intervals, dtype=int)
    count_values = np.zeros(num_intervals, dtype=int)

    # --- 4. Populate Graph Data ---
    for ts_sec in parsed_seconds:
        if start_time_seconds <= ts_sec < end_time_seconds:
            # Calculate the interval index (TR index)
            interval_index = int(ts_sec // interval_duration_seconds)
            if 0 <= interval_index < num_intervals:
                boolean_values[interval_index] = 1
                count_values[interval_index] += 1
                
    # Retaining the original CSV save functionality for count_values
    # You might want to switch this to .npy too, but following the original code structure for this file
    np.savetxt('_regressorTestcsv'+camOrScene+'.csv', count_values, delimiter=',')
    return boolean_values

def plot_data(regressor, convolved, title_name):
    tr = 2
    
    # 1. Regressor Plot
    fig_regressor, ax_orig = plt.subplots(figsize=(12, 4))
    ax_orig.plot(regressor, drawstyle='steps-post')
    ax_orig.set_title(f'{title_name} Regressor (Stimulus Events)')
    ax_orig.set_xlabel('Time (2s intervals)')
    ax_orig.set_ylabel('Amplitude')
    save_plot(fig_regressor, f"{title_name.replace(' ', '_')}_regressor")

    # 2. Convolved Plot
    fig_convolved, ax_mag = plt.subplots(figsize=(12, 4))
    ax_mag.plot(convolved)
    ax_mag.set_title(f'Convolved {title_name} Signal (Predicted fMRI BOLD Response)')
    ax_mag.set_xlabel('Time (2s intervals)')
    ax_mag.set_ylabel('Amplitude')
    save_plot(fig_convolved, f"{title_name.replace(' ', '_')}_convolved")

def plot_z_scores(zscore_array, title_name):
    # Scatter/Line plot of Z-scores
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(zscore_array)
    ax.set_ylabel('Z-Score Amplitude')
    ax.set_xlabel('Time (2s intervals)')
    ax.set_title(f'Z-Scored Data Points - {title_name}')
    save_plot(fig, f"{title_name.replace(' ', '_')}_zscore")

# --- Main Execution Block ---

# Load data (use error handling in case files are missing)
try:
    with open('daisy_timestamps.txt', 'r') as f:
        my_timestamps_txt = f.read()

    with open('daisy_scene_changes.txt', 'r') as f:
        my_scene_changes_txt = f.read()
except FileNotFoundError as e:
    print(f"Error: Required input file not found: {e}")
    # Create dummy data for demonstration if files are missing, or exit
    my_timestamps_txt = "0:1\n0:3\n0:5\n1:0\n1:2\n1:4"
    my_scene_changes_txt = "0:10\n0:25\n1:30"
    warnings.warn("Using dummy data for demonstration. Replace input files.", UserWarning)


# Generate Regressors
regressor_camera_cut = arrange_timestamp_data(my_timestamps_txt, 'cam')
regressor_scene_cut = arrange_timestamp_data(my_scene_changes_txt, 'scene')

# Generate HRF
n_TRs = 2 
hrf = spm_hrf(t_r=n_TRs, time_length=32, oversampling = 1) 

# ----------------------------------------------------------------------
# Core Task 1: Perform Convolution and Save as .npy Files
# ----------------------------------------------------------------------

# Camera Regressor
convolved_camera, convolved_z_camera = convolve_and_save_npy(
    regressor_camera_cut, hrf, 'camera'
)

# Scene Regressor
convolved_scene, convolved_z_scene = convolve_and_save_npy(
    regressor_scene_cut, hrf, 'scene'
)

# ----------------------------------------------------------------------
# Core Task 2: Plot and Automatically Store Plots to 'plots' Folder
# ----------------------------------------------------------------------

plot_data(regressor_camera_cut, convolved_camera, 'camera cut')
plot_data(regressor_scene_cut, convolved_scene, 'scene cut')

plot_z_scores(convolved_z_camera, 'camera cut')
plot_z_scores(convolved_z_scene, 'scene cut')

# The original script had plotting and saving code after the main logic; 
# the new structure integrates saving into the functions. The previous manual 
# saving attempts using .to_csv() on NumPy arrays are removed, as they are now 
# correctly saved as .npy files within convolve_and_save_npy.