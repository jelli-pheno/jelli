from jax import vmap, numpy as jnp

def outer_ravel(arr):
    return jnp.outer(arr, arr).ravel()

def batched_outer_ravel(arr):
    # Dynamically detect batch dimensions
    batch_shape = arr.shape[:-1]  # All dimensions except the last one

    # Reshape to flatten batch dimensions for efficient `vmap`
    arr = arr.reshape((-1, arr.shape[-1]))

    # Vectorize over the flattened batch axis
    result = vmap(outer_ravel)(arr)

    # Reshape result back to original batch structure
    return result.reshape(batch_shape + (-1,))
