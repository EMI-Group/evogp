import numpy as np
from .enum import FUNCS, FUNCS_NAMES
from functools import partial
import jax
import jax.numpy as jnp
from jax import lax

def dict2cdf(prob_dict):
    # Probability Dictionary to Cumulative Distribution Function
    assert len(prob_dict) > 0, "Empty probability dictionary"

    cdf = np.zeros(len(FUNCS))

    for key, val in prob_dict.items():
        assert key in FUNCS_NAMES, f"Unknown function name: {key}, total functions are {FUNCS_NAMES}"
        idx = FUNCS_NAMES.index(key)
        cdf[idx] = val

    # normalize
    cdf = cdf / cdf.sum()

    return np.cumsum(cdf)

@partial(jax.vmap, in_axes=(0, 0, 0, None))
def sub_arr(orig_arr: jax.Array, node_index, subtree_size, max_len):
    # Try not to change the `max_len` parameter or it will cause recompilation.
    if max_len == None: max_len = 1024

    def body_fun(i, val):
        new_arr, orig_arr = val
        return (new_arr.at[i].set(orig_arr[node_index + i]), orig_arr)

    new_arr = jnp.zeros(max_len, dtype=orig_arr.dtype)
    new_arr, _ = lax.fori_loop(0, subtree_size, body_fun, (new_arr, orig_arr))
    return new_arr