from os import name
from matplotlib import lines
import matplotlib.pyplot as plt
import numpy as np


def plot_timestamp_data(timestamps_txt_1, timestamps_txt_2=None, label_1="Data Set 1", label_2="Data Set 2", scene_changes_txt=None):
    # --- Helper function to parse timestamps ---
    def parse_timestamps(timestamps_str_data):
        parsed_seconds = []
        if not timestamps_str_data:
            return parsed_seconds
        for line in timestamps_str_data.strip().split('\n'):
            if not line: # Skip empty lines
                continue
            try:
                minutes, seconds = map(int, line.split(':'))
                total_seconds = minutes * 60 + seconds
                parsed_seconds.append(total_seconds)
            except ValueError:
                print(f"Warning: Skipping invalid timestamp format: '{line}'")
                continue
        return parsed_seconds

    # --- 1. Parse Timestamps for both data sets ---
    parsed_seconds_1 = parse_timestamps(timestamps_txt_1)
    parsed_seconds_2 = parse_timestamps(timestamps_txt_2) if timestamps_txt_2 else []
    parsed_seconds_3 = parse_timestamps(scene_changes_txt) if scene_changes_txt else []

    # --- 2. Define Time Range and Intervals ---
    start_time_seconds = 0  # 00:00
    end_time_minutes = 13
    end_time_seconds = end_time_minutes * 60  # 13:00 (780 seconds)
    interval_duration_seconds = 2

    # Calculate the number of intervals
    num_intervals = int(np.ceil((end_time_seconds - start_time_seconds) / interval_duration_seconds))

    # Generate interval start times (x-axis values)
    interval_starts = np.arange(start_time_seconds, end_time_seconds, interval_duration_seconds)
    # Ensure interval_starts does not exceed end_time_seconds for plotting consistency
    if len(interval_starts) > num_intervals:
        interval_starts = interval_starts[:num_intervals]

    # --- 3. Initialize Graph Data Structures ---
    boolean_values_1 = np.zeros(num_intervals, dtype=int)
    count_values_1 = np.zeros(num_intervals, dtype=int)
    boolean_values_2 = np.zeros(num_intervals, dtype=int)
    count_values_2 = np.zeros(num_intervals, dtype=int)
    bool_scene_changes = np.zeros(num_intervals, dtype=int)

    # --- 4. Populate Graph Data for Data Set 1 ---
    for ts_sec in parsed_seconds_1:
        # Check if the timestamp is within the desired plotting range
        if start_time_seconds <= ts_sec < end_time_seconds:
            interval_index = int(ts_sec // interval_duration_seconds)
            
            # Ensure index is within bounds (should be due to ts_sec check, but good practice)
            if 0 <= interval_index < num_intervals:
                boolean_values_1[interval_index] = 1
                count_values_1[interval_index] += 1

    # --- 4. Populate Graph Data for Data Set 2 (if provided) ---
    for ts_sec in parsed_seconds_2:
        # Check if the timestamp is within the desired plotting range
        if start_time_seconds <= ts_sec < end_time_seconds:
            interval_index = int(ts_sec // interval_duration_seconds)
            
            # Ensure index is within bounds (should be due to ts_sec check, but good practice)
            if 0 <= interval_index < num_intervals:
                boolean_values_2[interval_index] = 1
                count_values_2[interval_index] += 1
    
    # Bool for num camera cuts list: 0 if counts are equal, 1 if they are not
    for ts_sec in parsed_seconds_3:
        # Check if the timestamp is within the desired plotting range
        if start_time_seconds <= ts_sec < end_time_seconds:
            interval_index = int(ts_sec // interval_duration_seconds)
            
            # Ensure index is within bounds (should be due to ts_sec check, but good practice)
            if 0 <= interval_index < num_intervals:
                bool_scene_changes[interval_index] = 1
    
    # --- 5. New Code to compute the logical OR and write to files ---
    
    # Calculate the logical OR of the two boolean arrays and convert to integers
    or_intersect = np.logical_or(boolean_values_1, boolean_values_2).astype(int)

    # Write the OR intersect data to a text file
    with open("or_intersect_bools.txt", "w") as file:
        for val in or_intersect:
            file.write(f"{val}\n")

    # Write the scene changes data to a text file with newlines
    with open("scene_changes_bools.txt", "w") as file:
        for val in bool_scene_changes:
            file.write(f"{val}\n")


with open('daisy_timestamps.txt', 'r') as f:
    my_timestamps_txt = f.read()
with open('rebecca_timestamps.txt', 'r') as f:
    friend_timestamps_txt = f.read()
with open('daisy_scene_changes.txt', 'r') as f:
    my_scene_changes_txt = f.read()

# Call the modified function
plot_timestamp_data(my_timestamps_txt, friend_timestamps_txt, label_1="Daisy", label_2="Rebecca", scene_changes_txt=my_scene_changes_txt)