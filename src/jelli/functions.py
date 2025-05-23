from itertools import product
import numpy as np
from jax import numpy as jnp, scipy as jsp, jit
from functools import partial

# useful functions for dealing with WC linear and bilinear keys
def keys_product(keys_a, keys_b):
    """Computes the Cartesian product of two sets of keys, producing bilinear combinations.

    Args:
        keys_a (list): A list where each element is a tuple of the form (w, c).
        keys_b (list): Another list with elements of the form (w, c).

    Returns:
        (list): A list of bilinear combinations in the form (w_a, w_b, c_a + c_b).
    """
    if len(keys_a[0]) == 2:
        return [
            (w_a, w_b, c_a+c_b)
            for ((w_a, c_a), (w_b, c_b)) in product(keys_a, keys_b)
        ]
    else:
        raise ValueError("keys must be of the form (w,c)")

def keys_array(keys):
    """Converts a list of tuples into a numpy array.

    Args:
        keys (list): A list containing tuples.

    Returns:
        (np.ndarray): A numpy array with dtype=tuple containing the provided keys.
    """
    array = np.empty(len(keys), dtype=tuple)
    array[:] = keys
    return array

def keys_isin(keys_a, keys_b):
    """Checks if elements in `keys_a` exist in `keys_b`.

    Args:
        keys_a (list): List of keys to check.
        keys_b (list): List of reference keys.

    Returns:
        (np.ndarray): Boolean numpy array indicating presence of each key in `keys_a` within `keys_b`.
    """
    set_b = set(keys_b)
    res = np.array([item in set_b for item in keys_a])
    return res if res.size > 0 else np.array([], dtype=bool)

def linear2bilinear_indices(keys_wcs, keys_coeff):
    """Computes sorted indices mapping linear Wilson coefficients (WCs)
    to bilinear ones that exist in the provided coefficient list.

    Args:
        keys_wcs (list): List of linear Wilson coefficient keys, each element is a tuple (w, c),
                             where w is the WC name and c is 'R' for real or 'I' for imaginary.
        keys_coeff (list): List of bilinear coefficient keys. Each element is a tuple (w1, w2, c),
                               where w1 and w2 are the WC names and c 'RR', 'RI', 'IR', or 'II',
                               denoting all possible interferences.
    Returns:
        (np.ndarray): Sorted indices of bilinear coefficients that match `keys_coeff`.
    """
    # Generate all possible bilinear combinations of keys_wcs
    keys_wcs_bilinears = keys_array(keys_product(
        keys_wcs, keys_wcs
    ))
    bilin_bools = keys_isin(keys_wcs_bilinears, keys_coeff)
    # Take elements of keys_wcs_bilinears that exist in keys_coeff and obtain indices that sort them
    sort_indices = np.argsort(keys_wcs_bilinears[bilin_bools])
    bilin_indices = np.where(bilin_bools)[0]
    bilin_sort_indices = bilin_indices[sort_indices]
    return bilin_sort_indices
