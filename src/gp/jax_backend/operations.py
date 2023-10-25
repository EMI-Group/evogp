import jax
import jax.numpy as jnp
from src.core.utils import replace_sub_array

from src.core import Stack, NType, FUNC_TYPES, CHILDREN_NUM, Tree, FUNCS

eps = 1e-6


def new_tree(key, state, cal_subsize=True):
    def rand_func(k):
        func = jax.random.choice(k, state.func_pool, ())
        type = FUNC_TYPES[func]
        children = CHILDREN_NUM[type]
        return type, func.astype(jnp.float32), children

    def rand_inputs(k):
        return NType.VAR, jax.random.randint(k, (), 0, state.num_inputs).astype(jnp.float32), 0

    def rand_constant(k):
        return NType.CONST, jax.random.choice(k, state.const_pool, ()), 0

    def rand_leaf(k):
        k1, k2 = jax.random.split(k)
        r = jax.random.uniform(k1, (), minval=0, maxval=1)
        var_check = state.var_prob / (state.var_prob + state.const_prob)
        return jax.lax.cond(r < var_check, rand_inputs, rand_constant, k2)

    def rand_item(k, d):
        k1, k2 = jax.random.split(k)
        r = jax.random.uniform(k1, (), minval=0, maxval=1)
        return jax.lax.cond(r < 1 / (d + eps), rand_leaf, rand_func, k2)

    subtree_size = state.empty_subtree.shape[0]
    res = Stack.new((subtree_size, 2))  # result, [node_type, node_val]
    recur_stack = Stack.new((subtree_size, 2))  # recursive stack, [children, max_depth]

    recur_stack = recur_stack.push([1, state.new_tree_depth + 1])

    def cond(carry):
        k, r, rs = carry  # key, res, recur_stack
        return rs.size() > 0  # recur_stack is not empty

    def body(carry):
        k, r, rs = carry  # key, res, recur_stack
        k, k_ = jax.random.split(k)
        (c, d), rs = rs.pop()
        t_, v_, c_ = rand_item(k_, d)
        r = r.push([t_, v_])
        rs = jax.lax.cond(c >= 2, rs.push, rs.not_push, [c - 1, d])  # still has children to create
        rs = jax.lax.cond(c_ > 0, rs.push, rs.not_push, [c_, d - 1])  # new child is a func
        return k, r, rs

    _, res, recur_stack = jax.lax.while_loop(cond, body, (key, res, recur_stack))

    node_type, node_val = res.data[:, 0], res.data[:, 1]
    node_type = jnp.where(jnp.isnan(node_val), NType.NAN, node_type).astype(jnp.int32)
    node_val = jnp.where(jnp.isnan(node_val), 0, node_val)

    if cal_subsize:
        sub_size = get_sub_size(node_type, node_val)
        return Tree(node_type, node_val, sub_size)
    else:
        return node_type, node_val


def get_sub_size(node_types, node_vals):
    max_size = node_types.shape[0]
    res = Stack.new(max_size)
    stack = Stack.new(max_size)

    def cond_func(carry):
        p, r, s = carry  # pointer
        return p >= 0

    def body_func(carry):
        p, r, s = carry

        type, val = node_types[p], node_vals[p]

        def bf(s_, r_):  # binary function
            c1, s_ = s_.pop()
            c2, s_ = s_.pop()
            return s_.push(c1 + c2 + 1), r_.push(c1 + c2 + 1)

        def uf(s_, r_):  # unary function
            c1, s_ = s_.pop()
            return s_.push(c1 + 1), r_.push(c1 + 1)

        def leaf(s_, r_):  # variable
            return s_.push(1), r_.push(1)

        def nan(s_, r_):  # for case nan
            return s_, r_.push(0)

        s, r = jax.lax.switch(type, [leaf, leaf, uf, bf, nan], s, r)
        return p - 1, r, s

    _, res, _ = jax.lax.while_loop(cond_func, body_func, (max_size - 1, res, stack))
    sub_size = res.data[::-1].astype(jnp.int32)

    return sub_size


def cal(tree: Tree, inputs):
    max_size = tree.node_types.shape[0]
    stack = Stack.new(max_size)

    def cond_func(carry):
        p, s = carry  # pointer
        return p >= 0

    def body_func(carry):
        p, s = carry

        type, val = tree.node_types[p], tree.node_vals[p]

        # add new val in stack
        def op(v_, s_):  # operator
            res, s_ = jax.lax.switch(v_.astype(jnp.int32), FUNCS, s_)
            return s_.push(res)

        def var(v_, s_):  # variable
            return s_.push(inputs[v_.astype(jnp.int32)])

        def con(v_, s_):  # constant
            return s_.push(v_)

        def nan(v_, s_):  # for case nan
            return s_  # doing nothing!

        s = jax.lax.switch(type, [var, con, op, op, nan], val, s)

        return p - 1, s

    _, val_stack = jax.lax.while_loop(cond_func, body_func, (max_size - 1, stack))

    return jnp.array([val_stack.peek()])


def crossover(tree1: Tree, tree2: Tree, key):
    len1 = jnp.argmax(tree1.node_types == NType.NAN)
    len2 = jnp.argmax(tree2.node_types == NType.NAN)

    k1, k2 = jax.random.split(key)
    str1 = jax.random.randint(k1, (), 0, len1)
    size1 = tree1.subtree_size[str1]

    str2 = jax.random.randint(k2, (), 0, len2)
    size2 = tree2.subtree_size[str2]

    new_type = replace_sub_array(
        ori=tree1.node_types,
        tar=tree2.node_types,
        ori_len=len1,
        ori_start=str1,
        ori_size=size1,
        tar_start=str2,
        tar_size=size2
    )

    new_val = replace_sub_array(
        ori=tree1.node_vals,
        tar=tree2.node_vals,
        ori_len=len1,
        ori_start=str1,
        ori_size=size1,
        tar_start=str2,
        tar_size=size2
    )

    new_subsize = get_sub_size(new_type, new_val)

    return Tree(new_type, new_val, new_subsize)


def mutation(tree: Tree, key, state):
    k1, k2 = jax.random.split(key)
    sub_type, sub_val = new_tree(k1, state, cal_subsize=False)

    len1 = jnp.argmax(tree.node_types == NType.NAN)
    str1 = jax.random.randint(k2, (), 0, len1)
    size1 = tree.subtree_size[str1]

    len_sub = jnp.argmax(sub_type == NType.NAN)

    new_type = replace_sub_array(
        ori=tree.node_types,
        tar=sub_type,
        ori_len=len1,
        ori_start=str1,
        ori_size=size1,
        tar_start=0,
        tar_size=len_sub,
        nan_val=NType.NAN
    )

    new_val = replace_sub_array(
        ori=tree.node_vals,
        tar=sub_val,
        ori_len=len1,
        ori_start=str1,
        ori_size=size1,
        tar_start=0,
        tar_size=len_sub,
        nan_val=jnp.nan
    )

    new_subsize = get_sub_size(new_type, new_val)

    return Tree(new_type, new_val, new_subsize)


def check_correct(tree: Tree):
    size = jnp.sum(tree.node_types != NType.NAN)
    return size == tree.subtree_size[0]
