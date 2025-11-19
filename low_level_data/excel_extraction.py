import matplotlib.pyplot as plt
import json
import os

# Read the file and parse the data
file_path = "lowlvl_features.txt"
data = []

# Read the file and extract the data
with open(file_path, "r") as f:
    for line in f:
        # Split each line by commas and convert to float where needed
        values = line.strip().split(',')
        file_number = int(values[0])
        hue = float(values[1])
        saturation = float(values[2])
        value = float(values[3])
        motion_energy = float(values[4])
        amplitude = float(values[5])
        pitch = float(values[6])

        # Store the data as a tuple
        data.append((file_number, hue, saturation, value, motion_energy, amplitude, pitch))

# Extract x (file number) and y values (each header column)
file_numbers = [item[0] for item in data]
hues = [item[1] for item in data]
saturations = [item[2] for item in data]
values = [item[3] for item in data]
motion_energies = [item[4] for item in data]
amplitudes = [item[5] for item in data]
pitches = [item[6] for item in data]

with open("motion_energy_output.json", "r") as f:
    motion_data = json.load(f)

# Extract values
motion_values_json = list(motion_data.values())

amplitudes_json = []
pitches_json = []
with open("audio_output.json", "r") as f:
    audio_data = json.load(f)

# Open a text file for writing

for key, value in audio_data.items():
    # Extract values from nested dictionary
    amplitude = value.get("average_amplitude", "N/A")
    pitch = value.get("average_pitch_hz", "N/A")
    
    amplitudes_json.append(amplitude)
    pitches_json.append(pitch)


with open("hsv_output.json", "r") as f:
    hsv_data = json.load(f)

# Initialize empty lists for hue, saturation, and value
hues_json = []
saturations_json = []
values_json = []

# Loop through each item in the data
for key, value in hsv_data.items():
    # Extract the average_hsv list
    hsv_values = value.get("average_hsv", [])

    # Ensure there are exactly 3 values before appending
    if len(hsv_values) == 3:
        hues_json.append(hsv_values[0])
        saturations_json.append(hsv_values[1])
        values_json.append(hsv_values[2])


def plot_graph(x, y1, y2=None, xlabel="X", ylabel="Y", title="Title", label1="Set 1", label2="Set 2", filename="graph"):
    plt.figure(figsize=(10, 5))

    # Plot the first set of points
    plt.plot(x, y1, marker="o", linestyle="-", color="b", label=label1)
    
    # If a second set of points is provided, plot it as well
    if y2 is not None:
        plt.plot(x, y2, marker="x", linestyle="--", color="r", label=label2)

    # Labeling
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Show the plot
    filepath = os.path.join(os.getcwd(), filename + ".png")
    plt.savefig(filepath, dpi=300)
    plt.close()

# Example: Plotting two sets of coordinate points (hue vs file number and motion energy vs file number)
plot_graph(file_numbers, hues, hues_json, "File Number", "Values", "Hue vs File Number", "HueExcel", "HueJson","hue plot")
plot_graph(file_numbers, saturations, saturations_json, "File Number", "Values", "Saturation vs File Number", "SaturationExcel", "SaturationJson", "saturation plot")
plot_graph(file_numbers, values, values_json, "File Number", "Values", "Value vs File Number", "ValueExcel", "ValueJson", "value plot")
plot_graph(file_numbers, motion_energies, motion_values_json, "File Number", "Values", "Motion Energy vs File Number", "MotionExcel", "MotionJson", "motion plot")
plot_graph(file_numbers, amplitudes, amplitudes_json, "File Number", "Values", "Amplitude vs File Number", "AmpExcel", "AmpJson", "amplitude plot")
plot_graph(file_numbers, pitches, pitches_json, "File Number", "Values", "Pitch vs File Number", "PitchExcel", "PitchJson", "pitch plot")

# Function to plot the graph
# def plot_graph(x, y, xlabel, ylabel, title):
#     plt.figure(figsize=(10, 5))
#     plt.plot(x, y, marker="o", linestyle="-", color="b")
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.grid(True)
#     plt.show()

# # Plot graphs for each header
# plot_graph(file_numbers, hues, "File Number", "Hue", "Hue vs File Number")
# plot_graph(file_numbers, saturations, "File Number", "Saturation", "Saturation vs File Number")
# plot_graph(file_numbers, values, "File Number", "Value", "Value vs File Number")
# plot_graph(file_numbers, motion_energies, "File Number", "Motion Energy", "Motion Energy vs File Number")
# plot_graph(file_numbers, amplitudes, "File Number", "Amplitude", "Amplitude vs File Number")
# plot_graph(file_numbers, pitches, "File Number", "Pitch", "Pitch vs File Number")
