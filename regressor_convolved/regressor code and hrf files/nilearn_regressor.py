import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import zscore
import seaborn as sns
from nilearn.glm.first_level import spm_hrf
from scipy.signal import fftconvolve
from nilearn.glm.first_level import compute_regressor

def regressorbool(bool_values,file_name,duration_len):
    #parameters are amplitudes, cam vs scene, and 0.1 vs 0.5 for cam vs scene duration
    #scene_len = 0.5

    # create an "onset" array of 0.1 intervals from 0 to 778 (total number of seconds of movie)
    onsets_cam = np.arange(0, 778, 2)
    #print(onsets_cam)
    # create a "duration" array that is filled with length of event duration
    durations_cam = [duration_len] * len(onsets_cam)
    #print(durations_cam)

    # the amplitude array is the actual annotation data itself
    #trimming to fit the actual precise amount of seconds
    amplitudes = bool_values[:len(onsets_cam)]
    #print(amplitudes)
    # create the exp_condition variable with this data
    exp_condition_2 = (onsets_cam, durations_cam, amplitudes)

    # Using the same settings as above
    tr = 2 # fmri data has a TR of 2s
    n_scans = 389 # 389 timepoints in fmri data (389*2=778s)
    frame_times = np.arange(n_scans) * tr
    n_TRs = 2 # example
    #regressor = np.load('cam_shots_per_TR.npy')
    hrf = spm_hrf(t_r=n_TRs, time_length=32, oversampling = 1)
    # Compute a convolved regressor
    regressor_2, _ = compute_regressor(exp_condition=exp_condition_2, frame_times=frame_times, hrf_model='spm')
    single_array = regressor_2.flatten()    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frame_times, single_array)
    ax.set_ylabel('amplitude')
    ax.set_xlabel('Time (s)')
    ax.set_title('Regressor bool '+file_name)
    plt.show()
    convolved_camera = fftconvolve(single_array, hrf)[:len(single_array)]
    fig, ax_mag = plt.subplots(figsize=(12, 4))
    # Plot the convolved signal on the bottom subplot
    # Since we truncated 'convolved' to the length of 'regressor', the x-axis is simple
    ax_mag.plot(convolved_camera)
    ax_mag.set_title('Convolved '+file_name+ ' Signal')
    ax_mag.set_xlabel('Time 2s intervals')
    ax_mag.set_ylabel('Amplitude')
    plt.show()
    zscore_reg = zscore(regressor_2)
    # Plot the output for reference - it should be the same as above
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frame_times, zscore_reg)
    ax.set_ylabel('amplitude')
    ax.set_xlabel('Time (s)')
    ax.set_title('ZScore bool '+file_name)
    plt.show()
    regressor_df = pd.DataFrame(data=regressor_2, columns=['presence/absence of '+file_name])
    regressor_df.to_csv('./bool_regressor_'+file_name+'.csv')

    timing_data = pd.DataFrame(data=exp_condition_2, index=['onset', 'duration', 'amplitude'])
    timing_df = timing_data.T
    timing_df.to_csv('./bool_timing_'+file_name+'.csv')
    # Check if outputs are the same
    #print(np.array_equal(regressor_1, regressor_2)) # should raise error if outputs are different

def regressoramp(count_values,file_name,duration_len):
    #parameters are amplitudes, cam vs scene, and 0.1 vs 0.5 for cam vs scene duration
    #scene_len = 0.5

    # create an "onset" array of 0.1 intervals from 0 to 778 (total number of seconds of movie)
    onsets_cam = np.arange(0, 778, 2)
    #print(onsets_cam)
    # create a "duration" array that is filled with length of event duration
    durations_cam = [duration_len] * len(onsets_cam)
    #print(durations_cam)

    # the amplitude array is the actual annotation data itself
    #trimming to fit the actual precise amount of seconds
    amplitudes = count_values[:len(onsets_cam)]
    #print(amplitudes)
    # create the exp_condition variable with this data
    exp_condition_2 = (onsets_cam, durations_cam, amplitudes)

    # Using the same settings as above
    tr = 2 # fmri data has a TR of 2s
    n_scans = 389 # 389 timepoints in fmri data (389*2=778s)
    frame_times = np.arange(n_scans) * tr

    # Compute a convolved regressor
    regressor_2, _ = compute_regressor(exp_condition=exp_condition_2, frame_times=frame_times, hrf_model='spm')
    zscore_data = zscore(regressor_2)
    # Plot the output for reference - it should be the same as above
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frame_times, zscore_data)
    ax.set_ylabel('amplitude')
    ax.set_xlabel('Time (s)')
    ax.set_title('Regressor amplitude '+file_name)
    plt.show()
    regressor_df = pd.DataFrame(data=regressor_2, columns=['presence/absence of '+file_name])
    regressor_df.to_csv('./count_regressor_'+file_name+'.csv')

    timing_data = pd.DataFrame(data=exp_condition_2, index=['onset', 'duration', 'amplitude'])
    timing_df = timing_data.T
    timing_df.to_csv('./count_timing_'+file_name+'.csv')
    # Check if outputs are the same
    #print(np.array_equal(regressor_1, regressor_2)) # should raise error if outputs are different

def define_timestamp_data(timestamps_txt,file_name,duration_len):
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
    regressorbool(boolean_values,file_name,duration_len)
    #regressoramp(count_values,file_name,duration_len)



with open('daisy_timestamps.txt', 'r') as f:
    my_timestamps_txt = f.read()

with open('daisy_scene_changes.txt', 'r') as f:
    my_scene_changes_txt = f.read()
#separated by parameters specific to camera and scene
define_timestamp_data(my_timestamps_txt,'camera_cuts',2) 
define_timestamp_data(my_scene_changes_txt,'scene_cuts',2)