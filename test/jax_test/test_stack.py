from src.core import Stack
import jax
import jax.numpy as jnp


def test_new_tree():
    a = jnp.array([jnp.inf])
    print(a)
    a = a.astype(jnp.int32)
    print(a)
    print(jnp.float32 == jnp.floating)
    print(jnp.float32 == jnp.int32)
    print(jnp.float32 == jnp.float64)
    print(jnp.float32 == jnp.int64)
    print(jnp.float32 == jnp.int16)
    print(jnp.float32 == jnp.int8)
    print(jnp.float32 == jnp.float32)
    print(jnp.float16 == jnp.floating)


if __name__ == '__main__':
    test_new_tree()
