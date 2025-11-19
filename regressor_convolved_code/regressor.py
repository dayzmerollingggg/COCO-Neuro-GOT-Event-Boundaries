import numpy as np
import pandas as pd
from scipy.stats import gamma
import io

# --- 1. SET YOUR SCAN PARAMETERS HERE ---
# Replace with your experiment's Repetition Time in seconds
TR_IN_SECONDS = 2

# Replace with the total number of volumes (time points) in your fMRI run
NUMBER_OF_TIME_POINTS = 450 # Example: A 15-minute scan at TR=2s would be 450 points

# Define a name for your event type
EVENT_NAME = 'task_event'

# --- 3. DEFINE THE HEMODYNAMIC RESPONSE FUNCTION (HRF) ---
def canonical_hrf(t):
    """Defines the canonical double-gamma HRF."""
    peak1_params = {'shape': 6, 'scale': 1}
    peak2_params = {'shape': 16, 'scale': 1}
    g1 = gamma.pdf(t, a=peak1_params['shape'], scale=peak1_params['scale'])
    g2 = gamma.pdf(t, a=peak2_params['shape'], scale=peak2_params['scale'])
    hrf = g1 - (g2 / 6)
    return hrf / np.max(hrf)

if __name__ == "__main__":
    # Specify the name of your input file
    event_filename = 'timestamps_analysis.tsv'
    
    # Load the event data using the new function
    events_df = pd.read_csv(event_filename, sep='\t')

    # Proceed only if the dataframe was loaded successfully
    if events_df is not None:
        # --- Create the Boxcar Model ---
        OVERSAMPLING = 16
        scan_duration = NUMBER_OF_TIME_POINTS * TR_IN_SECONDS
        high_res_time = np.linspace(0, scan_duration, int(scan_duration * OVERSAMPLING), endpoint=False)
        boxcar = np.zeros_like(high_res_time)

        for _, row in events_df.iterrows():
            onset = row['onset']
            duration = row['duration']
            event_indices = (high_res_time >= onset) & (high_res_time < (onset + duration))
            boxcar[event_indices] = 1

        # --- Convolve Boxcar with HRF ---
        hrf_kernel_duration = 32
        hrf_time = np.linspace(0, hrf_kernel_duration, int(hrf_kernel_duration * OVERSAMPLING), endpoint=False)
        hrf = canonical_hrf(hrf_time)
        convolved = np.convolve(boxcar, hrf, mode='full')[:len(boxcar)]

        # --- Downsample to the Scan's TR ---
        scan_time_points = np.arange(NUMBER_OF_TIME_POINTS) * TR_IN_SECONDS
        resampled_indices = [np.argmin(np.abs(high_res_time - t)) for t in scan_time_points]
        regressor_values = convolved[resampled_indices]

        # --- Save the Output ---
        regressor_df = pd.DataFrame({EVENT_NAME: regressor_values})
        output_filename = 'custom_regressor.csv'
        regressor_df.to_csv(output_filename, index=False)

        print(f"Regressor generated successfully and saved to {output_filename}")