# Design notes: representation, generation and operators for the evoxc GP stack

Companion to `CONTEXT.md` (the binding brief) and `EVOXC_API.md` (the verified evoxc contract).
This file holds the *how* — the concrete vectorized constructions this subtree uses, so the reasoning is not re-derived or lost.
Every construction here keeps **static shapes** (`[pop_size, max_nodes]`), so one traced graph serves a whole run and only *values* vary per generation.
That is also why the SAME programs serve both routes — host numpy and device-resident `iree` — with no per-backend branch in the math; the device loop (`variation.run → eval.run`, two graph runs, zero host syncs) is what the "Elitism inside the variation graph" and "Evaluation and the driver" sections build up to.
Device-side figures quoted here come from `DEVICE_LOOP.md` (report of record, harness `measure_generation.py`).

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
Tournament (`k` opponents gathered by `randint` + `argmax` over the gathered fitness) and the rank/default modes; fitness arrives as a graph INPUT tensor, host array or device tensor, so selection needs no engine knowledge and a device generation never has to bring it back.
The vector is sanitized in-graph first (`NaN → -inf`), because a NaN compares false against everything and would corrupt both the tournament and the elite ranking.

## Mutation
`nonzero`/boolean masking is unavailable, so the torch family's *data-dependent row subset* becomes a **full-population masked apply**: draw candidate positions for every row, then `select(changed_mask, mutated, original)` per row.
Point mutation must not change arity silently: only same-arity opcode swaps (or payload-only edits on leaves) keep the parent's `size` channel valid.

## Elitism inside the variation graph
The elite splice is two more ops of a graph the generation already runs, so it lives there instead of in a second executable:
- order = `argsort(multiply(fitness, -1.0), descending=False, stable=True)`, cast to int32. Sorting the NEGATED vector ascending with `stable=True` is the in-graph spelling of `np.argsort(-fitness, kind="stable")`, so a tie keeps the lower row index; `descending=True` must not be used because it flips the tie order.
- the first `elite_count` entries index the parent rows, gathered out of the parent channels (bit-identical copies) and concatenated in front of the first `pop_size - elite_count` offspring rows; the rest of the offspring are dropped because they are exchangeable draws.
- `next_fitness` = the elite values (`gather(fitness, elite_index)`) followed by `UNEVALUATED` (`-inf`) for the offspring, and it is an OUTPUT of the graph — which is why a device `step` needs no readback.
`elite_index=` takes an explicit `[elite_count]` int32 vector instead of the ranking (one upload on a device run), and `elite_count=0` short-circuits to offspring-only with an all-`-inf` vector.
Rationale for the fusion is measured, not aesthetic: a separate splice graph cost a third `evoxc.run`, two H2D uploads and one readback per generation (`DEVICE_LOOP.md` §3), i.e. it was the only thing standing between the loop and a zero-sync generation.

## Evaluation and the driver
`evaluate(channels) -> (pop_size,) float32` (maximized) is the whole plug-in surface, and the driver is a plain Python loop: one built operator graph per generation plus one `evaluate` call.
The driver never traces `evaluate`.
Two kinds of objective are accepted with no declaration: a numpy-returning one (the host path) and an `evoxc`-tensor-returning one (`GraphEvaluator`), whose device then becomes the run's device.
Host path: one readback per generation.
Device path: `step` reads nothing back — the fitness tensor goes straight into the variation graph and the next fitness vector comes out of it — while `run` still reads the numbers back once per evaluated generation because its history and best-tree report are built from them, and downloads `best_channels` once at the end.
`GraphEvaluator` closes the last host step of that path by tracing the STRUCTURE PRELUDE (leaf/function kind, function id, const value, variable index, the three child offsets) from the channels inside the graph and then calling the same `problem._graph_sr_fitness` body the host-pre-pass route uses, so a device generation is `variation.run → eval.run` with the channels never leaving the device and the numerics shared by construction.

## Torch bridge placement
No module of this subtree contains a literal `import torch` / `from torch` / `import evox` statement — that is a pinned invariant, and it holds for the two places that need torch at all:
- the fused backend `fused_eval.py`, which resolves torch, triton and `evogp.tree.fused_engine` through `importlib.import_module` INSIDE its functions and returns numpy, so it is an opt-in `evaluate=` callable and never a driver path;
- the measurement harness `measure_generation.py`, which does the same to build the torch-stack sides it compares against.
The adapter that turns channels into a `../tree/Forest` for any other purpose stays on the caller side, outside this subtree; the tests own that glue.
The subtree itself ships the numpy evaluator (the CPU-box path and the parity oracle), the device-resident in-graph evaluator, and the duck-typed callable protocol.
