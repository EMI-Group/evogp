class NodeType:
	"""
	The enumeration class for GP node types
	"""
	VAR = 0
	CONST = 1
	UFUNC = 2
	BFUNC = 3

class Function:
	"""
	The enumeration class for GP function types
	"""
	ADD = 0
	SUB = 1
	MUL = 2
	DIV = 3
	TAN = 4
	SIN = 5
	COS = 6
	LOG = 7
	EXP = 8
	MAX = 9
	MIN = 10
	INV = 11
	NEG = 12


import jax
import jax.numpy as jnp

def to_jax_node(node_values : jax.Array, node_types : jax.Array, subtree_sizes : jax.Array) -> jax.Array:
	"""
	Convert the given GP node values, types and subtree sizes to a compact form used in CUDA.

	Can be compiled by `jax.jit`. Inputs shall be a batch of nodes.
	"""
	if node_values.dtype == jnp.float32:
		nt_and_sz = jnp.column_stack([node_types.astype(jnp.uint16), subtree_sizes.astype(jnp.uint16)])
		nt_and_sz = jax.lax.bitcast_convert_type(nt_and_sz, jnp.float32)
		return jax.lax.complex(node_values, nt_and_sz)
	else:
		nt_and_sz = jnp.column_stack([node_types.astype(jnp.uint32), subtree_sizes.astype(jnp.uint32)])
		nt_and_sz = jax.lax.bitcast_convert_type(nt_and_sz, jnp.float64)
		return jax.lax.complex(node_values, nt_and_sz)
	
def from_jax_node(jax_nodes : jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
	"""
	Convert the compact form of GP used in CUDA to distinct GP node values, types and subtree sizes.

	Can be compiled by `jax.jit`. Inputs can either be a batch of nodes (a GP) or a batch of GPs.
	"""
	if jax_nodes.dtype == jnp.complex64:
		nt_and_sz = jax.lax.bitcast_convert_type(jnp.imag(jax_nodes), jnp.uint16)
	else:
		nt_and_sz = jax.lax.bitcast_convert_type(jnp.imag(jax_nodes), jnp.uint32)
	node_values = jnp.real(jax_nodes)
	if len(nt_and_sz.shape) > 2:
		node_types = jax.vmap(lambda x: x[:, 0], in_axes=-2)(nt_and_sz).T
		subtree_sizes = jax.vmap(lambda x: x[:, 1], in_axes=-2)(nt_and_sz).T
	else:
		node_types = nt_and_sz[:, 0]
		subtree_sizes = nt_and_sz[:, 1]
	return node_values, node_types, subtree_sizes