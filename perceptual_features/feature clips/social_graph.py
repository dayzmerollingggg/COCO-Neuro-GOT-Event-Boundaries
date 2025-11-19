import matplotlib.pyplot as plt
import numpy as np 

# Load Daisy (rows are features, each row has 50 values)
with open('daisyavg.txt', 'r') as f:
    daisy_data = [list(map(float, line.strip().split())) for line in f]

# Load Rebecca (columns are features, each column has 50 values)
with open('rebeccaavg.txt', 'r') as f:
    rebecca_data = [list(map(float, line.strip().split())) for line in f]


# Transpose Rebecca to match feature-wise format for visual features

print("Shape of daisy_data:", len(daisy_data), "features x", len(daisy_data[0]), "clips")
print("Shape of rebecca_data:", len(rebecca_data), "features x", len(rebecca_data[0]), "clips")
clipnum = len(rebecca_data[0]) + 1
# X-axis: 1 to 50 clips
clips = list(range(1, clipnum))
feature_social = ['character_positivity (1-9)good person?',
'positive_behavior (1-9) good behavior?',
'positive_relationship (1-9)',
'char_pos_past good person?',
'char_pos_ future',
'situ_pos_curr',
'situ_pos_past',
'situ_pos_future']
for i, name in enumerate(feature_social):
    plt.figure(figsize=(10, 4))

    # Plot the data
    plt.plot(clips, daisy_data[i], label='Daisy', marker='o')
    plt.plot(clips, rebecca_data[i], label='Rebecca', marker='x')

    # Calculate Pearson correlation
    corr = np.corrcoef(daisy_data[i], rebecca_data[i])[0, 1]

    # Plot settings
    plt.title(f'Feature: {name} | Correlation: {corr:.2f}')
    plt.xlabel('Clip Number')
    plt.ylabel('Value (0 to 9)')
    plt.xticks(range(1, clipnum, 10))
    plt.ylim(-0.1, 10.1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

