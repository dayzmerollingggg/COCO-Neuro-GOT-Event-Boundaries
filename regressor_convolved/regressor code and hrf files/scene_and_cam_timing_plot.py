import matplotlib.pyplot as plt
import numpy as np

def plot_timestamp_data(timestamps_txt, scene_changes_txt):
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

    # --- 5. Plot Graphs ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Timestamp Analysis (00:00 - 13:00 with 2-Second Intervals)', fontsize=16)

    # Graph 1: Boolean Presence
    axes[0].step(interval_starts, boolean_values, where='post', color='blue', linewidth=1.5)
    axes[0].set_ylabel('Timestamp Present (1=Yes, 0=No)', fontsize=12)
    axes[0].set_title('Boolean Presence of Timestamps in 2-Second Intervals', fontsize=14)
    axes[0].set_yticks([0, 1])
    axes[0].set_ylim([-0.1, 1.1])
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # Graph 2: Count of Timestamps
    axes[1].plot(interval_starts, count_values, marker='o', linestyle='-', color='green', label='Timestamps')
    axes[1].set_xlabel('Time (Minutes:Seconds)', fontsize=12)
    axes[1].set_ylabel('Number of Timestamps', fontsize=12)
    axes[1].set_title('Count of Timestamps per 2-Second Interval', fontsize=14)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].set_ylim(bottom=-1, top=max(count_values) + 1)

    # --- 6. Add Scene Dividers from a separate file ---
    scene_changes_seconds = []
    for line in scene_changes_txt.strip().split('\n'):
        if not line:
            continue
        try:
            minutes, seconds = map(int, line.split(':'))
            total_seconds = minutes * 60 + seconds
            scene_changes_seconds.append(total_seconds)
        except ValueError:
            print(f"Warning: Skipping invalid scene change format: '{line}'")
            continue
    
    first_line = True
    for sc_sec in scene_changes_seconds:
        # Draw a vertical line on both graphs
        if first_line:
            label = 'Scene Change'
            first_line = False
        else:
            label = None
            
        axes[0].axvline(x=sc_sec, color='red', linestyle='--', linewidth=2, zorder=2, label=label)
        axes[1].axvline(x=sc_sec, color='red', linestyle='--', linewidth=2, zorder=2, label=label)

    # Add a legend to the top plot to explain the lines
    axes[0].legend(loc='upper right')

    # Format x-axis ticks to mm:ss
    major_tick_interval_seconds = 60
    major_tick_locations = np.arange(start_time_seconds, end_time_seconds + 1, major_tick_interval_seconds)
    x_tick_labels = []
    x_tick_positions = []
    for tick_sec in major_tick_locations:
        if start_time_seconds <= tick_sec <= end_time_seconds:
            mins = int(tick_sec // 60)
            secs = int(tick_sec % 60)
            x_tick_labels.append(f'{mins:02d}:{secs:02d}')
            x_tick_positions.append(tick_sec)
            
    axes[1].set_xticks(x_tick_positions)
    axes[1].set_xticklabels(x_tick_labels, rotation=45, ha='right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# This assumes you have both files in the same directory as your script, one for time stamps one for scene changes
with open('daisy_timestamps.txt', 'r') as f:
    my_timestamps_txt = f.read()

with open('daisy_scene_changes.txt', 'r') as f:
    my_scene_changes_txt = f.read()

plot_timestamp_data(my_timestamps_txt, my_scene_changes_txt)