import numpy as np


# Function to pad arrays to the same length repeating the last element
def pad_arrays(arrays):
    max_len = max(len(arr) for arr in arrays)
    return np.array([
        np.pad(arr, (0, max_len - len(arr)), mode='edge')
        for arr in arrays
    ])
