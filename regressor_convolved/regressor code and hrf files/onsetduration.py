import csv

def analyze_timestamps_to_tsv(input_file, output_file, seconds_per_line=2):
    """
    Reads a file of boolean values and calculates the time until the next
    positive (True) value occurs, saving the output as a TSV file.

    Args:
        input_file (str): The path to the input .txt file.
        output_file (str): The path for the output .tsv file.
        seconds_per_line (int): The number of seconds each line represents.
    """
    print(f"Reading data from '{input_file}'...")

    # Step 1: Find the line numbers of all 'True' values
    true_indices = []
    try:
        with open(input_file, 'r') as f:
            for i, line in enumerate(f):
                # Safely check for 'true', ignoring case and whitespace
                if line.strip() == '1':
                    true_indices.append(i)
    except FileNotFoundError:
        print(f" Error: The file '{input_file}' was not found.")
        return

    # Check if we have enough data to calculate durations
    if len(true_indices) < 2:
        print("Warning: Not enough 'True' values found to calculate any durations.")
        # Create an empty TSV with just the header
        with open(output_file, 'w', newline='') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t')
            writer.writerow(['onset', 'duration'])
        return

    # Step 2: Calculate onset and duration between consecutive 'True' values
    results = []
    for i in range(len(true_indices) - 1):
        current_index = true_indices[i]
        next_index = true_indices[i+1]

        onset = current_index * seconds_per_line
        duration = (next_index - current_index) * seconds_per_line
        eventtype = 'scene_cut' #camera_cut

        results.append({'onset': onset, 'duration': duration, 'trial_type': eventtype})

    # Step 3: Write the results to a new TSV file
    try:
        # Use 'delimiter=\t' to specify a tab as the separator
        with open(output_file, 'w', newline='') as tsvfile:
            fieldnames = ['onset', 'duration', 'trial_type']
            writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter='\t')

            writer.writeheader()
            writer.writerows(results)

        print(f"Success! Analysis complete. Results saved to '{output_file}'.")
    except IOError:
        print(f"Error: Could not write to the file '{output_file}'.")


# --- Main execution block ---
if __name__ == '__main__':
    # --- Configuration ---
    # ⬇️ 1. Change this to the name of your input file
    INPUT_FILENAME = 'scene_changes_bools.txt'

    # ⬇️ 2. Change this to your desired output filename (using .tsv extension)
    OUTPUT_FILENAME = 'scene_analysis.tsv'
    # ---------------------

    # Run the analysis function
    analyze_timestamps_to_tsv(INPUT_FILENAME, OUTPUT_FILENAME)