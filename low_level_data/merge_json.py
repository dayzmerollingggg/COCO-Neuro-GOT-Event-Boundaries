import json
import matplotlib.pyplot as plt

def graph_motion_energy(motion_json, excel_sheet):
    # Load the JSON file
    with open("motion_energy_output.json", "r") as f:
        data = json.load(f)

    # Extract values
    y_values = list(data.values())

    # Generate x values from 1 to the length of y_values
    x_values = list(range(1, len(y_values) + 1))

    # Plot the graph
    plt.figure(figsize=(10, 5))
    plt.plot(x_values, y_values, marker="o", linestyle="-", color="b", label="Motion Energy")

    # Labeling
    plt.xlabel("Frame Number")
    plt.ylabel("Motion Energy")
    plt.title("Motion Energy Over Time")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()


# # Write values to a text file
# with open("output.txt", "w") as out_file:
#     for value in values:
#         out_file.write(f"{value}\n")
# import json

# Load the JSON file
# with open("audio_output.json", "r") as f:
#     data = json.load(f)

# # Open a text file for writing
# with open("output.txt", "w") as out_file:
#     for key, value in data.items():
#         # Extract values from nested dictionary
#         amplitude = value.get("average_amplitude", "N/A")
#         pitch = value.get("average_pitch_hz", "N/A")
        
#         # Write to file
#         out_file.write(f"{amplitude}, {pitch}\n")

# Open the JSON file and load the data
with open("hsv_output.json", "r") as f:
    data = json.load(f)

# Initialize empty lists for hue, saturation, and value
hues = []
saturations = []
values = []

with open("output.txt", "w") as out_file:
# Loop through each item in the data
    for key, value in data.items():
        # Extract the average_hsv list
        hsv_values = value.get("average_hsv", [])

        # Ensure there are exactly 3 values before appending
        if len(hsv_values) == 3:
            hues.append(hsv_values[0])
            saturations.append(hsv_values[1])
            values.append(hsv_values[2])

    out_file.write(f"{hues}, {saturations}, {values}\n")

# Print the extracted lists
print("Hues:", hues)
print("Saturations:", saturations)
print("Values:", values)



print("Values extracted successfully to output.txt!")
