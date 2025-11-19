import numpy as np
from nilearn.glm.first_level import spm_hrf
from scipy.signal import fftconvolve
from scipy.stats import zscore
import matplotlib.pyplot as plt

def arrange_timestamp_data(timestamps_txt, camOrScene):
    # --- 1. Parse Timestamps ---
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
            continue

    # --- 2. Define Time Range and Intervals ---
    start_time_seconds = 0
    end_time_minutes = 13
    end_time_seconds = end_time_minutes * 60
    interval_duration_seconds = 2

    num_intervals = int(np.ceil((end_time_seconds - start_time_seconds) / interval_duration_seconds))
    interval_starts = np.arange(start_time_seconds, end_time_seconds, interval_duration_seconds)
    if len(interval_starts) > num_intervals:
        interval_starts = interval_starts[:num_intervals]

    # --- 3. Initialize Graph Data Structures ---
    boolean_values = np.zeros(num_intervals, dtype=int)
    count_values = np.zeros(num_intervals, dtype=int)

    # --- 4. Populate Graph Data ---
    for ts_sec in parsed_seconds:
        if start_time_seconds <= ts_sec < end_time_seconds:
            interval_index = int(ts_sec // interval_duration_seconds)
            if 0 <= interval_index < num_intervals:
                boolean_values[interval_index] = 1
                count_values[interval_index] += 1
    np.savetxt('_regressorTestcsv'+camOrScene, count_values, delimiter=',')
    return boolean_values

def plot_data(regressor, convolved,title_name):
    tr = 2 # fmri data has a TR of 2s
    n_scans = 389 # 389 timepoints in fmri data (389*2=778s)
    frame_times = np.arange(n_scans) * tr
    target_length = len(frame_times)
    fig, ax_orig = plt.subplots(figsize=(12, 4))

    # Plot the original regressor on the top subplot
    ax_orig.plot(regressor, drawstyle='steps-post') # Use steps for discrete events
    ax_orig.set_title(title_name+' Regressor (Stimulus Events)')
    ax_orig.set_xlabel('Time 2s intervals')
    ax_orig.set_ylabel('Amplitude')
    plt.show()
    fig, ax_mag = plt.subplots(figsize=(12, 4))
    # Plot the convolved signal on the bottom subplot
    # Since we truncated 'convolved' to the length of 'regressor', the x-axis is simple
    ax_mag.plot(convolved)
    ax_mag.set_title('Convolved '+title_name+ ' Signal (Predicted fMRI BOLD Response)')
    ax_mag.set_xlabel('Time 2s intervals')
    ax_mag.set_ylabel('Amplitude')

    #fig.tight_layout()
    plt.show()
def plot_z_scores(zscore,title_name):
    #scatterplot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(zscore)
    ax.set_ylabel('amplitude')
    ax.set_xlabel('Time 2s intervals')
    ax.set_title('ZScored data points - '+title_name)
    plt.show()

# This assumes you have both files in the same directory as your script, one for time stamps one for scene changes
with open('daisy_timestamps.txt', 'r') as f:
    my_timestamps_txt = f.read()

with open('daisy_scene_changes.txt', 'r') as f:
    my_scene_changes_txt = f.read()

regressor_camera_cut = arrange_timestamp_data(my_timestamps_txt,'cam')
regressor_scene_cut = arrange_timestamp_data(my_scene_changes_txt,'scene')

#TR = 2.0 # seconds
n_TRs = 2 # example
#regressor = np.load('cam_shots_per_TR.npy')
hrf = spm_hrf(t_r=n_TRs, time_length=32, oversampling = 1) # covers 32s, peak ~5s, undershoot ~12s
np.savetxt('./hrf.csv', hrf, delimiter=',')
fig, ax_mag = plt.subplots(figsize=(12, 4))
ax_mag.plot(hrf)
ax_mag.set_title('HRF')
ax_mag.set_xlabel('Time points')
ax_mag.set_ylabel('Amplitude')

#fig.tight_layout()
plt.show()
# convolved_camera = fftconvolve(regressor_camera_cut, hrf)[:len(regressor_camera_cut)]
# convolved_z_camera = zscore(convolved_camera)

# convolved_scene = fftconvolve(regressor_scene_cut, hrf)[:len(regressor_scene_cut)]
# convolved_z_scene = zscore(convolved_scene)

# plot_data(regressor_camera_cut,convolved_camera,'camera cut')
# plot_data(regressor_scene_cut,convolved_scene,'scene cut')
# plot_z_scores(convolved_z_camera,'camera cut')
# plot_z_scores(convolved_z_scene,'scene cut')
# convolved_z_scene.to_csv('./z_score_scene.csv')
# convolved_z_camera.to_csv('./z_score_camera.csv')
# convolved_camera.to_csv('./convolved_camera.csv')
# convolved_scene.to_csv('./convolved_scene.csv')
# np.savetxt('./z_score_scene.csv', convolved_z_scene, delimiter=',')
# np.savetxt('./z_score_camera.csv', convolved_z_camera, delimiter=',')
# np.savetxt('./convolved_camera.csv', convolved_camera, delimiter=',')
# np.savetxt('./convolved_scene.csv', convolved_scene, delimiter=',')