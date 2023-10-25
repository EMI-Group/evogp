from functools import partial
import jax.numpy as jnp
from jax import jit


@jit
def where_set(
        ori: jnp.array, tar: jnp.array, ori_start, ori_end, tar_start
):
    """
    This function is equivalent to:
    ori.at[ori_start: ori_end].set(tar[tar_start: tar_start + ori_end - ori_start])
    which is not jit-able.
    """
    indices = jnp.arange(ori.shape[0])
    return jnp.where(
        (ori_start <= indices) & (indices < ori_end),
        tar[indices - ori_start + tar_start],
        ori
    )


# @jit
@partial(jit, static_argnames=["nan_val"])
def replace_sub_array(ori, tar, ori_len, ori_start, ori_size, tar_start, tar_size, nan_val=jnp.nan):
    """
    Use tar[tar_start: tar_start + tar_size] to replace ori[ori_start: ori_start + ori_size] in ori.
    
    For example:
    a = [1, 2, 3, 4, 5, 6, NaN, NaN, NaN, NaN]
    b = [10, 20, 30, 40, 50, 60, NaN, NaN, NaN, NaN]
    replace(a, b, 6, 2, 2, 3, 3)
    means use b[3: 3 + 3] = [40, 50, 60] to replace a[2: 2 + 2] = [3, 4]
    with res = [1, 2, 40, 50, 60, 5, 6, NaN, NaN, NaN]
    """
    new = ori
    new = where_set(new, tar, ori_start, ori_start + tar_size, tar_start)
    new = where_set(new, ori, ori_start + tar_size, ori_len - ori_size + tar_size, ori_start + ori_size)
    new = jnp.where(jnp.arange(new.shape[0]) >= ori_len - ori_size + tar_size, nan_val, new)
    return new


@partial(jit, static_argnames=["new_size"])
def expand(arr, new_size, val):
    # use val to expand array into new_size
    return jnp.concatenate([arr, jnp.full((new_size - arr.shape[0],), val)])


def test_where_set():
    ori = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
    tar = jnp.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    print(where_set(ori, tar, 2, 5, 3))


def test_replace_sub_array():
    a = jnp.array([1, 2, 3, 4, 5, 6, jnp.nan, jnp.nan, jnp.nan, jnp.nan])
    b = jnp.array([10, 20, 30, 40, 50, 60, jnp.nan, jnp.nan, jnp.nan, jnp.nan])
    print(replace_sub_array(a, b, 6, 2, 2, 3, 3))

    print(replace_sub_array(a, b, 6, 0, 6, 0, 1))


def test_expand():
    a = jnp.array([1, 2, 3, 4])
    print(expand(a, 10, 0))


def main():
    test_where_set()
    test_replace_sub_array()
    test_expand()


if __name__ == '__main__':
    main()
