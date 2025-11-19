from os import name
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import spearmanr


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
    bool_num_cuts = [int(v1 != v2) for v1, v2 in zip(count_values_1, count_values_2)]

    # Fourth list: The average of the counts from the two datasets
    avg_num_cuts = [(v1 + v2) / 2 for v1, v2 in zip(count_values_1, count_values_2)]
    jacc_corr = calculate_jaccard_index(boolean_values_1, boolean_values_2)
    cohen_kappa = calculate_cohens_kappa(boolean_values_1, boolean_values_2)
    spearmans = calculate_spearmans_correlation(boolean_values_1, boolean_values_2)


    # Plot settings
    
    # --- 5. Plot Graphs ---
    plt.style.use('seaborn-v0_8-darkgrid') # A nice style for plots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Timestamp Analysis (00:00 - 13:00 with 2-Second Intervals) - Comparison', fontsize=16)

    # Graph 1: Boolean Presence
    axes[0].step(interval_starts, boolean_values_1, where='post', color='blue', linewidth=1.5, label=label_1,marker='o')
    if timestamps_txt_2:
        axes[0].step(interval_starts, boolean_values_2, where='post', color='orange', linestyle='--',linewidth=1.5, label=label_2,marker='x')
    axes[0].set_ylabel('Camera Cut Present (1=Yes, 0=No)', fontsize=12)
    axes[0].set_title(f'Presence of Camera Cuts in 2-Second Intervals | Jaccard Index: {jacc_corr:.2f} | Cohen\'s Kappa: {cohen_kappa:.2f} | Spearman\'s: {spearmans:.2f}', fontsize=14)
    axes[0].set_yticks([0, 1])
    axes[0].set_ylim([-0.1, 1.1])
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend()

    corr = np.corrcoef(count_values_1, count_values_2)[0, 1]
    amt_correct = bool_num_cuts.count(0)/bool_num_cuts.__len__()
    # Graph 2: Count of Timestamps
    axes[1].plot(interval_starts, count_values_1, marker='o', linestyle='-', color='blue', label=label_1)
    if timestamps_txt_2:
        axes[1].plot(interval_starts, count_values_2, marker='o', linestyle='--', color='orange', label=label_2)
        #axes[1].plot(interval_starts, bool_num_cuts, marker='x', linestyle='--', color='orange', label=label_2)
        axes[1].plot(interval_starts, avg_num_cuts, marker='x', linestyle='--', color='green', label="Average")
    axes[1].set_xlabel('Time (Minutes:Seconds)', fontsize=12)
    axes[1].set_ylabel('Number of Camera Cuts', fontsize=12)
    axes[1].set_title(f'Count of Camera Cuts per 2-Second Interval | Pearson Correlation: {corr:.2f}| Boolean Correlation: {amt_correct:.2f}', fontsize=14)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].set_ylim(bottom=-1, top=max(count_values_1.max(), count_values_2.max()) + 1)
    axes[1].legend()

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
    axes[0].legend(loc='lower right')
    axes[1].legend(loc='upper left')

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

def calculate_spearmans_correlation(arr1, arr2):
    """
    Calculates Spearman's rank correlation coefficient between two arrays.
    Spearman's correlation assesses how well the relationship between two
    variables can be described using a monotonic function.
    For binary data, it's equivalent to Pearson correlation on ranks.
    """
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)

    # Ensure arrays are of the same length
    if len(arr1) != len(arr2):
        print("Error: Arrays must have the same length to calculate Spearman's correlation.")
        return np.nan

    # spearmanr returns a tuple (correlation, p-value)
    # We only need the correlation coefficient
    correlation, _ = spearmanr(arr1, arr2)
    return correlation
def calculate_jaccard_index(daisyinput, rebeccainput):
    """
    Calculates the Jaccard Index between two boolean (0/1) arrays.
    The Jaccard Index is defined as the size of the intersection divided by the size of the union.
    For binary data, this means:
    J(A, B) = |A intersect B| / |A union B|
    where A and B are sets of elements where the value is 1.
    """
    arr1_bool = np.array(daisyinput).astype(bool)
    arr2_bool = np.array(rebeccainput).astype(bool)

    intersection = np.sum(arr1_bool & arr2_bool)

    union = np.sum(arr1_bool | arr2_bool)

    if union == 0:
        return 0.0 
    else:
        return intersection / union
    
def calculate_cohens_kappa(arr1, arr2):
    """
    Calculates Cohen's Kappa coefficient between two binary (0/1) arrays.
    Cohen's Kappa measures the agreement between two raters (or measurements)
    while accounting for the possibility of agreement occurring by chance.
    """
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)

    # Ensure arrays are of the same length
    if len(arr1) != len(arr2):
        print("Error: Arrays must have the same length to calculate Cohen's Kappa.")
        return np.nan

    N = len(arr1) # Total number of observations

    # Calculate observed agreement (Po)
    # n11: both are 1
    # n00: both are 0
    n11 = np.sum((arr1 == 1) & (arr2 == 1))
    n00 = np.sum((arr1 == 0) & (arr2 == 0))
    Po = (n11 + n00) / N

    # Calculate expected agreement (Pe)
    # Marginal probabilities for arr1
    P_arr1_1 = np.sum(arr1 == 1) / N
    P_arr1_0 = np.sum(arr1 == 0) / N

    # Marginal probabilities for arr2
    P_arr2_1 = np.sum(arr2 == 1) / N
    P_arr2_0 = np.sum(arr2 == 0) / N

    # Pe = (P(arr1=1) * P(arr2=1)) + (P(arr1=0) * P(arr2=0))
    Pe = (P_arr1_1 * P_arr2_1) + (P_arr1_0 * P_arr2_0)

    # Handle the case where 1 - Pe is zero to avoid division by zero
    if 1 - Pe == 0:
        return 1.0 if Po == 1.0 else 0.0 # Perfect agreement if Pe is 1, otherwise 0
    else:
        kappa = (Po - Pe) / (1 - Pe)
        return kappa

# Call the function with both timestamp data sets
#plot_timestamp_data(example_timestamps_1, example_timestamps_2, label_1="My Data", label_2="Friend's Data")

# You can also use the function with only one data set:
# plot_timestamp_data(example_timestamps_1, label_1="My Data")

# To load from files:
with open('daisy_timestamps.txt', 'r') as f:
    my_timestamps_txt = f.read()
with open('rebecca_timestamps.txt', 'r') as f:
    friend_timestamps_txt = f.read()
with open('daisy_scene_changes.txt', 'r') as f:
    my_scene_changes_txt = f.read()
plot_timestamp_data(my_timestamps_txt, friend_timestamps_txt, label_1="Daisy", label_2="Rebecca", scene_changes_txt=my_scene_changes_txt)
