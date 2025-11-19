import pandas as pd
import matplotlib.pyplot as plt
import re # Import regex for advanced string matching

def process_and_plot_data(file_path, metrics_to_plot):
    """
    Reads a CSV file, extracts columns containing 'avg' in their header,
    filters specific rows (metrics), and plots the data for each metric.

    Args:
        file_path (str): The path to the CSV file.
        metrics_to_plot (list): A list of strings representing the row labels
                                 (metrics) to be plotted.
    """
    try:
        # Load the CSV file. Assuming the first column contains the metric names.
        # Use 'iloc' to ensure the first column is used as index without relying on its name.
        df = pd.read_csv(file_path, index_col=0)
        print("CSV loaded successfully.")
        print("Original DataFrame head:")
        print(df.head())

        # Identify columns that contain 'avg' (case-insensitive)
        # and parse the clip number from them.
        avg_columns = {}
        for col in df.columns:
            # Use regex to find "avg" and capture the clip number if format is "Clip X - Avg"
            # or just identify columns with "avg" and assume clip order.
            match = re.search(r'Clip (\d+) - Avg', col, re.IGNORECASE)
            if match:
                clip_num = int(match.group(1))
                avg_columns[clip_num] = col
            elif 'avg' in col.lower():
                # Fallback for columns like 'Clip1avg', assuming numerical order if no 'Clip X'
                # This part might need adjustment based on exact column names.
                # For this specific file, "Clip X - Avg" seems to be the format.
                pass # Already handled by the more specific regex above

        # Sort columns by clip number
        sorted_clips = sorted(avg_columns.keys())
        filtered_column_names = [avg_columns[clip] for clip in sorted_clips]

        if not filtered_column_names:
            print("No 'avg' columns found in the CSV based on 'Clip X - Avg' pattern.")
            print("Please check column headers in your CSV. Example: 'Clip 1 - Avg'.")
            return

        # Select only the identified 'avg' columns
        df_avg = df[filtered_column_names]
        print("\nDataFrame with 'avg' columns selected:")
        print(df_avg.head())

        # Filter the DataFrame to include only the specified metrics (rows)
        # Use .reindex to ensure the order of metrics_to_plot is maintained and handle missing ones gracefully.
        df_plot = df_avg.reindex(metrics_to_plot).dropna(how='all')

        if df_plot.empty:
            print("\nNo data found for the specified metrics.")
            print(f"Metrics requested: {metrics_to_plot}")
            print("Please ensure these metric names exactly match the first column in your CSV.")
            return

        print("\nDataFrame ready for plotting (filtered metrics):")
        print(df_plot)

        # Prepare x-axis labels (clip numbers)
        # Extract clip numbers from the column names, e.g., "Clip 1 - Avg" -> 1
        x_labels = [int(re.search(r'Clip (\d+)', col).group(1)) for col in filtered_column_names]

        # Create plots for each specified metric
        plt.style.use('seaborn-v0_8-darkgrid') # A nice looking plot style
        plt.figure(figsize=(15, 10)) # Adjust figure size for better readability

        for metric in metrics_to_plot:
            if metric in df_plot.index:
                # Get the data for the current metric
                data_points = df_plot.loc[metric].values

                # Plot the data
                plt.plot(x_labels, data_points, marker='o', linestyle='-', label=metric)
            else:
                print(f"Warning: Metric '{metric}' not found in the filtered data. Skipping.")

        plt.title('Average Values Across Clips for Selected Metrics')
        plt.xlabel('Clip Number')
        plt.ylabel('Average Value')
        plt.xticks(x_labels[::5], rotation=45, ha='right') # Show every 5th clip number for clarity
        plt.legend(title='Metric')
        plt.grid(True)
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure the path is correct.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check your CSV file format and ensure it's well-formed.")

# Define the path to your uploaded CSV file
csv_file_path = 'rebecca_social.csv'

# Define the list of metrics (rows) you want to plot
metrics_to_plot = [
    "character positivity",
    "positive behavior",
    "positive relationship",
    "character pos past",
    "char pos future",
    "situation pos curr",
    "situation pos past",
    "situation pos future"
]

# Run the function to process and plot
process_and_plot_data(csv_file_path, metrics_to_plot)

