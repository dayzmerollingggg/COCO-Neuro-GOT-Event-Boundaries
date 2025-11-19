import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn.glm.first_level import compute_regressor

def regressor(count_values,file_name,duration_len):
    #parameters are amplitudes, cam vs scene, and 0.1 vs 0.5 for cam vs scene duration
    #scene_len = 0.5

    # create an "onset" array of 0.1 intervals from 0 to 778 (total number of seconds of movie)
    onsets_cam = np.arange(0, 778.1, 0.1)
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

    # Plot the output for reference - it should be the same as above
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frame_times, regressor_2[:, 0])
    ax.set_ylabel('amplitude')
    ax.set_xlabel('Time (s)')
    ax.set_title('Regressor - presence/absence of '+file_name)
    plt.show()
    regressor_df = pd.DataFrame(data=regressor_2, columns=['presence/absence of speaking'])
    regressor_df.to_csv('./example_regressor_'+file_name+'.csv')

    timing_data = pd.DataFrame(data=exp_condition_2, index=['onset', 'duration', 'amplitude'])
    timing_df = timing_data.T
    timing_df.to_csv('./example_timing_'+file_name+'.csv')
    # Check if outputs are the same
    #print(np.array_equal(regressor_1, regressor_2)) # should raise error if outputs are different

def define_timestamp_data(timestamps_txt,file_name,duration_len):
    # --- 1. Parse Timestamps ---
    # this is given txt file of timestamps of mm:ss and separated by white line
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
    end_time_minutes = 13 #rounded up from original amount of seconds
    end_time_seconds = end_time_minutes * 60
    
    # Define the new, finer interval duration
    finer_interval_duration = 0.1
    
    # Calculate the number of new, finer intervals
    num_finer_intervals = int(np.ceil((end_time_seconds - start_time_seconds) / finer_interval_duration))
    
    # --- 3. Initialize Data Structures for the Finer Intervals ---
    final_boolean_values = np.zeros(num_finer_intervals, dtype=int)

    # --- 4. Populate Data Structures by first getting counts per 1-second interval ---
    one_sec_interval_duration = 2.0 #1 or 2 second intervals
    num_one_sec_intervals = int(np.ceil((end_time_seconds - start_time_seconds) / one_sec_interval_duration))
    count_values_per_sec = np.zeros(num_one_sec_intervals, dtype=int)
    
    for ts_sec in parsed_seconds:
        if start_time_seconds <= ts_sec < end_time_seconds:
            one_sec_interval_index = int(ts_sec // one_sec_interval_duration)
            if 0 <= one_sec_interval_index < num_one_sec_intervals:
                count_values_per_sec[one_sec_interval_index] = 1 #not counting all, just set bool to 1
    
    # --- 5. Distribute Counts and Populate the Final Array ---
    for i in range(num_one_sec_intervals):
        count = count_values_per_sec[i]
        
        if count > 0:
            # Base timestamp for the current 1-second interval
            base_timestamp = i * one_sec_interval_duration
            
            # Calculate the time step for even distribution within the second
            time_step = 1.0 / (count + 1)
            
            for j in range(count):
                # Calculate the exact timestamp for the current event
                new_event_time = base_timestamp + (j + 1) * time_step
                
                # Find the corresponding index in the finer-grained array
                finer_index = int(round(new_event_time / finer_interval_duration))
                
                if finer_index < num_finer_intervals:
                    final_boolean_values[finer_index] = 1
    regressor(final_boolean_values,file_name,duration_len)



with open('daisy_timestamps.txt', 'r') as f:
    my_timestamps_txt = f.read()

with open('daisy_scene_changes.txt', 'r') as f:
    my_scene_changes_txt = f.read()
#separated by parameters specific to camera and scene
define_timestamp_data(my_timestamps_txt,'camera_cuts',0.1) 
define_timestamp_data(my_scene_changes_txt,'scene_cuts',0.5)