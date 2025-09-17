from jax import vmap, numpy as jnp

def outer_ravel(arr):
    '''
    Compute the outer product of a 1D array and return it as a raveled 1D array.

    Parameters
    ----------
    arr : jnp.ndarray
        A 1D JAX array.

    Returns
    -------
    jnp.ndarray
        A 1D JAX array representing the raveled outer product of the input array.
    '''
    return jnp.outer(arr, arr).ravel()

def batched_outer_ravel(arr):
    '''
    Compute the outer product for each 1D array in a batch and return them as raveled 1D arrays.

    Parameters
    ----------
    arr : jnp.ndarray
        A JAX array of shape (..., N), where ... represents any number of batch dimensions

    Returns
    -------
    jnp.ndarray
        A JAX array of shape (..., N*N), where each slice along the batch dimensions
        corresponds to the raveled outer product of the respective input array.
    '''
    # Dynamically detect batch dimensions
    batch_shape = arr.shape[:-1]  # All dimensions except the last one

    # Reshape to flatten batch dimensions for efficient `vmap`
    arr = arr.reshape((-1, arr.shape[-1]))

    # Vectorize over the flattened batch axis
    result = vmap(outer_ravel)(arr)

    # Reshape result back to original batch structure
    return result.reshape(batch_shape + (-1,))
