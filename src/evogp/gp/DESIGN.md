# Design notes: representation, generation and operators for the evoxc GP stack

Companion to `CONTEXT.md` (the binding brief) and `EVOXC_API.md` (the verified evoxc contract).
This file holds the *how* — the concrete vectorized constructions this subtree uses, so the reasoning is not re-derived or lost.
Every construction here keeps **static shapes** (`[pop_size, max_nodes]`), so one traced graph serves a whole run and only *values* vary per generation.

## Channels (prefix / pre-order encoding, node 0 = root)
Four parallel `[pop_size, max_nodes]` evoxc tensors, replacing `../tree/`'s `(value float32, type int16, size int16)` triple:
- `opcode` — int32: the `Func` id for a function node, or `NType` marker for a leaf (constant vs variable vs output-tag).
- `payload` — float32: the constant value (constant leaf) or the variable index (variable leaf); unused for function nodes.
  The torch stack packs `(idx << 16) | fid` through a float bit-pattern; evoxc has **no `bitcast`**, so the payload is a separate float channel and the opcode an int channel (this is the whole reason for the extra channel).
- `size` — int32: `size[i]` = node count of the subtree rooted at `i`; `size[:, 0]` = the row's real length.
- Live mask is **derived, never stored**: `arange(max_nodes)[None, :] < size[:, 0:1]` (one comparison, no channel, no host sync).

Child addressing stays the repo-wide invariant: `c1 = i + 1`, `c2 = c1 + size[c1]`, `c3 = c2 + size[c2]`.
`MAX_STACK` / `MAX_FULL_DEPTH` / `NType` / `Func` are imported from `../tree/utils.py` (single source; that module is torch-free).

## Generation (`init_population`) — level-wise sampling, then one scatter to pre-order
The decision structure of `../tree/generation.py` is reproduced exactly, because it is what defines the distribution:
- node at depth `d` is a **function** iff `r >= depth2leaf_probs[d]`, else a leaf;
- its function id is `searchsorted(roulette_funcs, r, right=True)` = the number of cdf entries `<= r`, i.e. `reduce_sum(cast(cdf <= r))` (evoxc has no `searchsorted`);
- arity from the id: `IF` → 3, ternary/binary range → 2, unary range → 1;
- leaf: constant with probability `const_prob` (value drawn uniformly from `const_samples`), else variable index `% input_len`;
- multi-output out-tagging (`out_prob`) is a later concern (single output first).

Layout while growing = the **padded complete tree** (as `_generate_padded`): level `d` has `A ** d` slots with `A` = the largest arity the function set can produce, so parent/child links are static (`parent = j // A`, `rank = j % A`) and no compaction is needed while sampling.
Only a Python loop over levels (≤ `MAX_FULL_DEPTH + 1` = 11) and over child ranks (≤ 3) — never over individuals or nodes.
Then, in the same graph:
1. **subtree sizes bottom-up** over the padded levels (children die when their rank ≥ the parent's arity or the parent is a leaf);
2. **pre-order index top-down**: `pre(child at rank c) = pre(parent) + 1 + sum(size of live siblings with rank < c)` — at most two terms, so it is a couple of `select`/`gather`s per level (not a monotonic-stack scan);
3. **one `scatter`** of the sampled opcode/payload/size into the `[pop_size, max_nodes]` pre-order buffers.
Padded slots whose parent is not a live function are dropped by the scatter, exactly as in the torch implementation.

## Splice (crossover, and subtree-replacement mutation) by index arithmetic
Legacy semantics to preserve (`../tree/splicing.py::_splice`, ENGINE_SPEC §7): keep `[0, pos)`; copy the donor subtree to `[pos, pos + new_sub)`; shift the tail by `size_diff = new_sub - old_sub`; add `size_diff` to every **strict ancestor** of `pos`; leave a row whose new length exceeds `max_nodes` untouched.
All of it is expressible in static shapes over `j = arange(max_nodes)`:
- `src_index = select(j < pos, j, select(j < pos + new_sub, donor_pos + j - pos, j + old_sub - new_sub))`
- values from the target = `gather(target, src_index)`; values from the donor = `gather(donor, donor_pos + j - pos)`; `select(j < pos + new_sub, from_donor, from_target)` picks the donor inside the copied window (the donor subtree is contiguous in pre-order, so no remap inside it).
- sizes: region 1 keeps `size[j] + (ancestor ? size_diff : 0)` with `ancestor = (j < pos) & (j + size[j] > pos)`; region 2 takes the donor's sizes verbatim; the tail keeps its own.
- rows that would overflow are reverted with `select(overflow, old, new)`.
Crossover: `pos`/`donor_pos` from `randint` over `[0, size[:, 0])` per row; every offspring is therefore structurally well-formed by construction.

## Selection
Tournament (`k` opponents gathered by `randint` + `argmax` over the gathered fitness) and the rank/default modes; fitness arrives as an input tensor from the host readback, so selection needs no engine knowledge.

## Mutation
`nonzero`/boolean masking is unavailable, so the torch family's *data-dependent row subset* becomes a **full-population masked apply**: draw candidate positions for every row, then `select(changed_mask, mutated, original)` per row.
Point mutation must not change arity silently: only same-arity opcode swaps (or payload-only edits on leaves) keep the parent's `size` channel valid.

## Evaluation and the driver
`evaluate(channels) -> (pop_size,) float32` (maximized) is the whole plug-in surface.
It is **host-side** work per generation (an evoxc engine call or the numpy evaluator), so the driver is a plain Python loop: run one built operator graph per generation, call `evaluate`, read fitness back once — exactly ONE host sync per generation.
The driver never traces `evaluate`.

## Torch bridge placement
The subtree imports **no torch, not even lazily**. Therefore the adapter that turns channels into a `../tree/Forest` (or drives `../tree`'s engines) lives on the caller side, outside this subtree — the tests own that glue.
The subtree itself ships: the numpy evaluator (the CPU-box path and the parity oracle) plus the duck-typed callable protocol.
