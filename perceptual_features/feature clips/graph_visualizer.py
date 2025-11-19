import matplotlib.pyplot as plt
import numpy as np 
from scipy.stats import spearmanr

# Load Daisy (rows are features, each row has 50 values)
with open('daisy.txt', 'r') as f:
    daisy_data = [list(map(float, line.strip().split())) for line in f]

# Load Rebecca (columns are features, each column has 50 values)
with open('rebecca.txt', 'r') as f:
    rebecca_data = [list(map(float, line.strip().split())) for line in f]

# Transpose Rebecca to match feature-wise format for visual features
#rebecca_data = list(map(list, zip(*raw_rebecca)))
print("Shape of daisy_data:", len(daisy_data), "features x", len(daisy_data[0]), "clips")
print("Shape of rebecca_data:", len(rebecca_data), "features x", len(rebecca_data[0]), "clips")
clipnum = len(rebecca_data[0]) + 1
clips = list(range(1,clipnum ))
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

# visual features
# Feature names
feature_bool = ['face_present', 'body_part_present', 'indoor_outdoor', 'place_new', 'round_object', 'animate_object', 'background_music', 'written_words', 'social_interaction', 'spoken_communication', 'speech_involved_chars', 'speech_outside_chars', 'speech_object', 'inter_person_action_present', 'mentalization_present']
feature_int =[ 'valence', 'arousal', 'num_characters']
feature_boolend = ['new_character_present', 'social_hierarchy', 'bodily_pain']
x = len(feature_bool)
y=x+len(feature_int)
#Plotting
for i, name in enumerate(feature_bool):
    plt.figure(figsize=(10, 4))
    plt.plot(clips, daisy_data[i], label='Daisy', marker='o')
    plt.plot(clips, rebecca_data[i], label='Rebecca', marker='x')
    jacc_corr = calculate_jaccard_index(daisy_data[i], rebecca_data[i])
    cohen_kappa = calculate_cohens_kappa(daisy_data[i], rebecca_data[i])
    spearmans = calculate_spearmans_correlation(daisy_data[i], rebecca_data[i])
    # Plot settings
    plt.title(f'Feature: {name} | Jaccard Index: {jacc_corr:.2f} | Cohen\'s Kappa: {cohen_kappa:.2f} | Spearman\'s: {spearmans:.2f}')
    plt.xlabel('Clip Number')
    plt.ylabel('Value (0 or 1)')
    plt.xticks(range(1, clipnum, 10))  # Show every 5 clips on the x-axis
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
for i, name in enumerate(feature_int):
    plt.figure(figsize=(10, 4))
    plt.plot(clips, daisy_data[i+x], label='Daisy', marker='o')
    plt.plot(clips, rebecca_data[i+x], label='Rebecca', marker='x')
    corr = np.corrcoef(daisy_data[i+x], rebecca_data[i+x])[0, 1]

    # Plot settings
    plt.title(f'Feature: {name} | Correlation: {corr:.2f}')
    plt.xlabel('Clip Number')
    plt.ylabel('Value (0 to 9)')
    plt.xticks(range(1, clipnum, 10))  # Show every 5 clips on the x-axis
    plt.ylim(-0.1, 10.1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
for i, name in enumerate(feature_boolend):
    plt.figure(figsize=(10, 4))
    plt.plot(clips, daisy_data[i+y], label='Daisy', marker='o')
    plt.plot(clips, rebecca_data[i+y], label='Rebecca', marker='x')
    jacc_corr = calculate_jaccard_index(daisy_data[i], rebecca_data[i])
    cohen_kappa = calculate_cohens_kappa(daisy_data[i+y], rebecca_data[i+y])
    spearmans = calculate_spearmans_correlation(daisy_data[i+y], rebecca_data[i+y])
    # Plot settings
    plt.title(f'Feature: {name} | Jaccard Index: {jacc_corr:.2f} | Cohen\'s Kappa: {cohen_kappa:.2f} | Spearman\'s: {spearmans:.2f}')
    plt.xlabel('Clip Number')
    plt.ylabel('Value (0 to 1)')
    plt.xticks(range(1, clipnum, 10))  # Show every 5 clips on the x-axis
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# plt.figure(figsize=(10, 4))
# plt.plot(clips, daisy_data[i], label='Daisy', marker='o')
# plt.plot(clips, rebecca_data[i], label='Rebecca', marker='x')
# corr = np.corrcoef(daisy_data[i], rebecca_data[i])[0, 1]

#     # Plot settings
# plt.title(f'Feature: Social Hierarchy | Correlation: {corr:.2f}')
# plt.xlabel('Clip Number')
# plt.ylabel('Value (0 or 1)')
# plt.xticks(range(1, 101, 5))  # Show every 5 clips on the x-axis
# plt.ylim(-0.1, 1.1)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()