# ENGINE_SPEC.md — EvoGP tree engine: evaluation, fitness, generation, splicing

Self-contained semantics spec for the EvoGP tree engine.
It was distilled from the CUDA/HIP kernel sources that used to live in `src/evogp/cuda/`
(`forward.cu`, `generate.cu`, `mutation.cu`, `defs.h`, `kernel.h`, `torch_wrapper.cu`)
at repository snapshot `8a487f19171a40823d5a3ae85c3738cc1bc9483a`, and it is the durable
record of the semantics reproduced by the current extension-free Python implementation
(`etl_engine.py` for evaluation/fitness, `generation.py` / `splicing.py` for the operators).
Section 13 is the only *implementation* section: it records the alternative dense whole-population
route (`dense_engine.py`, design documentation in `DENSE_EVAL.md`) without changing any semantics.
Legacy-only tables and notes are marked as such where they can no longer guide code.
Line references of the form `forward.cu:181` point into that snapshot.

Notation: `value` = `batch_node_value` (float32), `type` = `batch_node_type` (int16),
`size` = `batch_subtree_size` (int16); `var1`, `var2`, `var3` are the operands of a node in pop order.

---

## 1. Tensor and memory contract

- An individual is a **prefix (pre-order) expression**: node 0 is the root; a node's
  children follow depth-first, left to right, recursively (`_treeGPEvalByStack`, generate.cu:123).
- Three parallel row tensors, all shape `[pop_size, max_tree_len]`, all row-major contiguous on CUDA
  (CUDA + contiguity was the legacy-op requirement; the current Python engine reads them from any device):
  `value` float32, `type` int16, `size` int16 (`torch_wrapper.cu:63-66`).
- `size[i]` = number of nodes in the subtree rooted at node `i` (`generate.cu:130-158`).
  It is a **relative** count (not an absolute end index), so a subtree can be copied verbatim.
- `size[0]` = the real tree length. Only `[0, size[0])` is meaningful; the rest of the row
  is uninitialized (`torch::empty`) garbage and must never be read or compared.
- `max_tree_len` is passed as the row stride only; the generator/mutator/crossover write exactly
  `size[0]` entries per output row (`generate.cu:160-172`, `mutation.cu:107-113`).
- Positions of x86/ARM64 hosts are little-endian; the out-node float packing (section 2.3) assumes it.
- Type is int16 → a tree cannot exceed 32767 nodes; `max_tree_len <= MAX_STACK = 1024` is enforced
  in Python, not in the kernels.

---

## 2. Node encoding

### 2.1 Base types (`defs.h:10-22`)

| const | value | meaning |
| --- | --- | --- |
| `VAR` | 0 | variable |
| `CONST` | 1 | constant |
| `UFUNC` | 2 | unary function |
| `BFUNC` | 3 | binary function |
| `TFUNC` | 4 | ternary function |
| `OUT_NODE` | `1<<7` = 128 | output-node flag |
| `TYPE_MASK` | `0x7F` = 127 | extracts the base type |
| `UFUNC_OUT`/`BFUNC_OUT`/`TFUNC_OUT` | 130 / 131 / 132 | function node tagged as output |

### 2.2 `value` meaning per base type

- `VAR`: variable index as a float-encoded int. Evaluation does `int var_num = (int)node_value`
  (C truncation toward zero, `forward.cu:100`); out-of-range or non-finite values are undefined
  (they index the per-thread variable array out of bounds).
- `CONST`: the constant itself.
- `UFUNC`/`BFUNC`/`TFUNC` **not** tagged out: the function id (0..28) as a float (`forward.cu:107`).
- `UFUNC`/`BFUNC`/`TFUNC` **tagged** out (multi-output mode only): bit reinterpretation of the
  packed struct below.

### 2.3 Out-node packing (`kernel.h:107-115`)

```c++
struct OutNodeValue { int16_t function; int16_t outIndex;  // low 16 bits: function, high 16: out index
                      operator float() const { return *(float*)this; } };
```
- 4 bytes, no padding: `bits = *(int32_t*)&value`; `function = bits & 0xFFFF`, `outIndex = bits >> 16`.
- Python-side readers use `bits & 0xFF` for the function (identical for ids 0..28, whose high byte is 0).
- `outIndex` may be any value the generator drew; the evaluator ignores it when `outIdx >= outLen`
  (it does **not** clamp or trap).

---

## 3. Function table (ids 0..28, `defs.h:24-57`)

`var1` = first operand popped (first child in prefix order); `var2` = second; `var3` = third.
`DELTA = 1E-9f` and `MAX_VAL = 1E9f` (`defs.h:7-8`);
`copy_sign(x, s) = copysign(x, s)` = `|x|` with the sign of `s` (`kernel.h:129-137`).

| id | name | arity | type | exact semantics |
| --- | --- | --- | --- | --- |
| 0 | `if` | 3 | TFUNC | `var1 > 0.0f ? var2 : var3` (condition = first child) |
| 1 | `+` | 2 | BFUNC | `var1 + var2` |
| 2 | `-` | 2 | BFUNC | `var1 - var2` |
| 3 | `*` | 2 | BFUNC | `var1 * var2` |
| 4 | `/` | 2 | BFUNC | `var2 == 0.0f ? NAN : var1 / var2` |
| 5 | `loose_div` | 2 | BFUNC | if `|var2| <= DELTA`: `var2 = copy_sign(DELTA, var2)`; then `var1 / var2` |
| 6 | `pow` | 2 | BFUNC | `pow(var1, var2)` |
| 7 | `loose_pow` | 2 | BFUNC | `(var1 == 0 && var2 == 0) ? 0.0f : pow(|var1|, var2)` |
| 8 | `max` | 2 | BFUNC | `var1 >= var2 ? var1 : var2` (ternary, **not** `fmaxf`) |
| 9 | `min` | 2 | BFUNC | `var1 <= var2 ? var1 : var2` (ternary, **not** `fminf`) |
| 10 | `<` | 2 | BFUNC | `var1 < var2 ? 1.0f : -1.0f` |
| 11 | `>` | 2 | BFUNC | `var1 > var2 ? 1.0f : -1.0f` |
| 12 | `<=` | 2 | BFUNC | `var1 <= var2 ? 1.0f : -1.0f` |
| 13 | `>=` | 2 | BFUNC | `var1 >= var2 ? 1.0f : -1.0f` |
| 14 | `sin` | 1 | UFUNC | `sin(var1)` |
| 15 | `cos` | 1 | UFUNC | `cos(var1)` |
| 16 | `tan` | 1 | UFUNC | `tan(var1)` |
| 17 | `sinh` | 1 | UFUNC | `sinh(var1)` |
| 18 | `cosh` | 1 | UFUNC | `cosh(var1)` |
| 19 | `tanh` | 1 | UFUNC | `tanh(var1)` |
| 20 | `log` | 1 | UFUNC | `log(var1)` — **no** zero special case (`log(0) = -inf`, `log(x<0) = NaN`) |
| 21 | `loose_log` | 1 | UFUNC | `var1 == 0.0f ? -MAX_VAL : log(|var1|)` (i.e. `-1e9`) |
| 22 | `exp` | 1 | UFUNC | `exp(var1)` |
| 23 | `inv` | 1 | UFUNC | `var1 == 0.0f ? NAN : 1.0f / var1` |
| 24 | `loose_inv` | 1 | UFUNC | if `|var1| <= DELTA`: `var1 = copy_sign(DELTA, var1)`; then `1.0f / var1` (=> `±1e9`) |
| 25 | `neg` | 1 | UFUNC | `-var1` |
| 26 | `abs` | 1 | UFUNC | `|var1|` |
| 27 | `sqrt` | 1 | UFUNC | `sqrt(var1)` (negative input => NaN) |
| 28 | `loose_sqrt` | 1 | UFUNC | if `var1 <= 0.0f`: `var1 = |var1|`; then `sqrt(var1)` |
| 29 | `END` | — | — | sentinel only, not a function (sizes the roulette cdf) |

Rules that follow from the dispatch structure (`forward.cu:118-224`):

- The unary/binary/ternary dispatch is keyed on the **base type**; within a type the function id is
  matched by an if/else chain. A type/id combination that matches no branch (e.g. a UFUNC node whose
  id is 0, or any id > 28) leaves the result at the initializer `top_val = 0.0f`.
- Any node whose base type is not `VAR`/`CONST`/`UFUNC`/`BFUNC` (including, in single-output mode, a
  node still carrying `OUT_NODE`) falls into the **ternary** branch and is evaluated as `if`
  (pops 3 operands). See section 4.3.
- Arity is implied by the base type: UFUNC = 1, BFUNC = 2, TFUNC = 3. It is never read from a table.
- NaN/±inf are **not** sanitized: the clamping block (`is_nan -> 0`, `|x| > MAX_VAL -> ±MAX_VAL`)
  exists but is commented out (`forward.cu:226-234`). NaN and inf propagate into parents and fitness.
- Comparison operand order follows `v1 op v2`; when the comparison is false (including any NaN
  involvement) the result is `-1.0f`, i.e. NaN picks the `: b` / `-1` branch.
- `max(v1, NaN)` = NaN (its test fails); `max(NaN, v2)` = `v2`. `min` symmetrically.
- `pow` with a negative base and non-integral exponent is NaN (platform `std::pow`).

---

## 4. Evaluation semantics

### 4.1 Stack discipline (`_process_node`, forward.cu:79-244)

- Operand stack = a per-thread `float stack[MAX_STACK]`, `top` starts at 0.
- `CONST`: `stack[top++] = value`; `VAR`: `stack[top++] = vars[(int)value]`. Leaves push, never pop.
- Function nodes pop `arity` operands with pre-decrement, so **the first pop is the last-pushed
  operand = the leftmost child**; `var1 = stack[--top]` is the first child, `var2` the second, etc.
- Hence `-` = `first_child - second_child`; `/` = `first_child / second_child`; `if` tests the first
  child and returns the second when true, else the third.
- After computing, the result is pushed: `stack[top++] = top_val`.
- No bounds checks anywhere: a malformed tree underflows/overflows `top` (silent corruption).

### 4.2 Driver and order (`_treeGPEvalByStack`, forward.cu:246-302)

1. (multi-output only) `s_outs` is carved out of the `infos` scratch buffer and zero-initialized for `outLen` entries.
2. The `varLen` variable values are copied into the per-thread `infos` scratch buffer (`s_vars`).
3. `len = size[0]`; the tree is loaded **in reverse** into the same buffer that will become the operand
   stack: `s_vals[len-1-i] = value[i]`, `s_infos[len-1-i] = type[i]`.
4. Nodes are then processed in ascending index order `i = 0..len-1`, i.e. in reverse prefix order,
   pushing the results back into the same array. This in-place reuse is safe because before processing
   index `i` at most `i` pushes have happened (`top <= i`), so a write at `top` never clobbers a
   not-yet-read slot.
5. After the loop `top == 1` is asserted (debug assert + a `printf` of `top` when it differs; under
   `NDEBUG` evaluation simply continues with a corrupt stack).
- Scratch layout (`infos`, `int16[2*MAX_STACK]` bytes reinterpreted):
  `[MAX_STACK]` int16 = tree types used during load only; `s_vars` at `infos + MAX_STACK` as float
  (`<= MAX_STACK/4` variables asserted); multi-output only: `s_outs` at `infos + MAX_STACK + MAX_STACK/2`
  as float (`<= MAX_STACK/4` outputs asserted).
- `evaluate` picks the template by `outLen > 1` (forward.cu:367-370).
  - Single output: `results[n] = stack[--top]` (i.e. the sole remaining operand slot = the whole
    tree's value) (forward.cu:349).
  - Multi output: `results[n][i] = s_outs[i]` for `i < outLen` (forward.cu:342-347).

### 4.3 Multi-output protocol (template parameter `multiOutput = (outLen > 1)`)

- If `multiOutput`: `is_outNode = type & OUT_NODE`; `type &= TYPE_MASK` **before** the leaf test.
  A tagged leaf (CONST|OUT_NODE, VAR|OUT_NODE) is therefore evaluated as an ordinary untagged leaf and
  the tag is silently ignored.
- For tagged function nodes, `function` and `outIdx` are read from the packed bits (section 2.3);
  untagged function nodes use `node_value` directly as the function id.
- `right_node` is set to the **last-popped** operand (the rightmost child) of every function node:
  unary -> its child, binary -> the second child, ternary -> the third child (`forward.cu:121-123, 173-175, 219-221`).
- After computing `top_val` (before pushing), `if (is_outNode && outIdx < outLen) s_outs[outIdx] += top_val`
  — accumulation (Fan/Zhang multi-output scheme), and out-of-range indices are dropped (`forward.cu:239-240`).
- Then, for **every** function node (tagged or not), `top_val = right_node` before the push
  (`forward.cu:241`): a function node contributes only its rightmost child's value upward, while its
  own value (and the values of its other children) only survive through `s_outs` accumulation.
- The final Q outputs are the accumulated `s_outs`, not the stack value.
- Because the whole forest shares one output vector semantics per tree, `output_len > 1` means
  "sum of tagged subtrees per output", not one independent tree per output.

---

## 5. SR fitness (`forward.cu:373-857`)

- Error per (tree, data point): `multiOutput` -> sum over `o < outLen` of `(labels[o] - outs[o])^2`
  (or `|labels[o] - outs[o]|` when `useMSE=false`); single output -> `(labels[0] - stack[--top])^2`
  (or the absolute value) (`calculate_fit`, forward.cu:375-400).
- `diff = label - prediction` (this sign is irrelevant after squaring / absolute value).
- Sum over all data points (per-block shared-memory reduction + one `atomicAdd` per block), then a
  single division by `dataPoints` on the host-launched `averageFitnessValueKernel`.
- Therefore `fitness = (Σ_points Σ_outputs err) / dataPoints`: **outputs are summed but never divided
  by `outLen`**. The result is a positive loss, `(pop_size,)` float32; the higher-is-better convention
  (negation) is applied by the caller (`evogp.problem.SymbolicRegression`).
- No sanitization: NaN/±inf values, overflow of `diff*diff`, etc. flow straight into the fitness.
- The fitness buffer is cleared with `cudaMemsetAsync` at the start of each driver. The kernels
  accumulate (`+=` / `atomicAdd`), so they must not be invoked twice without a clear.

### 5.1 Legacy `kernel_type` dispatch (`SR_fitness`, forward.cu:827-856)

*Historical: the rewritten wrappers no longer map `execute_mode` to a `kernel_type`; they accept all
five mode strings and evaluate every one of them through the same Python engine path. The table below is the old dispatch.*

| id | name | host driver | launch shape |
| --- | --- | --- | --- |
| 0 | `normal` ("hybrid parallel") | `normal_SR_fitness` (forward.cu:481-512) | grid `{popSize, ceil(dataPoints/1024)}`, block 1024 |
| 1 | `constant_tree` ("data parallel") | `constant_tree_SR_fitness` (forward.cu:649-692) | per tree: copy the row into `__constant__` memory, then grid `ceil(dataPoints/1024) x 1024` |
| 2 | `constant_vars` ("tree parallel") | `constant_vars_SR_fitness` (forward.cu:781-825) | per data point: copy vars+labels into `__constant__` memory, then one thread per tree |
| 3 | `advanced` ("hybrid parallel" from `Tree`) | `advanced_SR_fitness` (forward.cu:514-549) | grid `{popSize, ceil(dataPoints/block)}`, `block = dataPoints <= 256 ? dataPoints : 1024` |
| 4 | `auto` | forward.cu:849-854 | `advanced` if `dataPoints <= cudaCoreCount`, else `constant_tree` |
| other | — | — | nothing is launched **and the buffer is not even cleared** -> the returned tensor is uninitialized garbage |

- `cudaCoreCount = multiProcessorCount * coresPerSM(compute capability)` with `coresPerSM`:
  3.0/3.2/3.5/3.7 -> 192; 5.0/5.2/5.3 -> 128; 6.0 -> 64; 6.1/6.2 -> 128; 7.0/7.2/7.5 -> 64;
  8.0 -> 64; 8.6/8.7/8.9 -> 128; anything else -> 128 (forward.cu:16-77).
- The id is the `execute_mode` string mapped by the Python wrapper: `Forest.SR_fitness` maps
  `"hybrid parallel"` -> 0, `"data parallel"` -> 1, `"tree parallel"` -> 2, `"auto"` -> 4
  (`forest.py:340-347`), while `Tree.SR_fitness` maps `"hybrid parallel"` -> **3**
  (`tree.py:94-101`). None of the two wrappers can emit id 3 other than through that string.
- Requirements of the option-1 driver: `popSize` sequential `cudaMemcpyToSymbolAsync` calls of the row
  (`gpLen` floats / `gpLen` int16) plus the root `subtree_size` (`sizeof(int16_t)` = 2 bytes) into the
  `const_node_value` / `const_node_type` / `const_tree_size` symbols; evaluation then walks
  `const_tree_size - 1 .. 0` (`_constant_tree_treeGPEvalByStack`, forward.cu:551-589) — the same
  reverse-prefix stack walk, reading whole rows from constant memory.
- Option 2 stores `varLen` floats and `outLen` floats per data point in `const_variables` / `const_labels`
  and has each thread accumulate `fitnesses[treeId] += fit` (racy-free because there is one thread per tree).

---

## 6. Random generation (`tree_generate`, `generate.cu:16-234`)

### 6.1 Inputs

- `popSize, gpLen (= max_tree_len, the row stride), varLen (= input_len), outLen (= output_len),
  constSamplesLen, outProb, constProb`, plus four GPU tensors: `keys {2} uint32`,
  `depth2leaf_probs {MAX_FULL_DEPTH=10} float32`, `roulette_funcs {29} float32`,
  `const_samples {constSamplesLen} float32` (`torch_wrapper.cu:34-83`).
- Template selection: `if (outLen > 1)` selects the multi-output template.
- One thread per individual; the tree-generation arrays are per-thread locals:
  `GPNode gp[MAX_STACK]`, `NchildDepth childsAndDepth[MAX_STACK]` (a scratch stack reused later as an
  `int` array), so memory use is bounded by `MAX_STACK` with **no bounds checks in the kernel**.

### 6.2 Per-thread setup

```
leafProbs[0..9]   = depth2leaf_probs[0..9]          // copied into a local float[10]
funcRoulette[0..28] = roulette_funcs[0..28]         // copied into a local float[29]
engine = thrust::random::taus88(hash(n, keys[0], keys[1]))   // section 9
rand   = thrust::uniform_real_distribution<float>(0.0f, 1.0f) // yields [0, 1)
childsAndDepth[0] = { childs: 1, depth: 0 };  top = 1;  topGP = 0
```

### 6.3 Main loop (`while (top > 0)`, generate.cu:62-128)

1. `cd = childsAndDepth[--top]; cd.childs--;`
2. **Leaf test**: `rand() >= leafProbs[cd.depth]` -> function branch; otherwise leaf branch.
   Equivalently: a node at depth `d` is a leaf iff `rand() < leafProbs[d]`. `cd.depth` is the depth of
   the node being generated (root = 0) and indexes the 10-entry local array directly, so
   `depth >= 10` is an out-of-bounds read (undefined), and `leafProbs[d] == 0` disables the leaf test
   entirely at that depth (runaway tree).
3. **Function branch**:
   - `r = rand()`; scan `i = 28 .. 0`, take the first `i` with `r >= funcRoulette[i]` and set `k = i + 1`;
     if none matches (i.e. `r < funcRoulette[0]`) then `k = 0`.
     This is an inverse-cdf draw assuming `roulette_funcs` is a non-decreasing cdf of the 29 slots
     ending at exactly `1.0`; the used slot probabilities are the successive differences.
     A sliding slot (`roulette_funcs[i] == roulette_funcs[i+1]`) has probability 0 and is never
     selected except on an exact tie; if the last slot is `< 1.0`, `k = 29` (= `END`) is reachable and
     yields a UFUNC node with an invalid id (evaluates to 0.0f).
   - `type = (k <= 0) ? TFUNC : (k <= 13) ? BFUNC : UFUNC` (i.e. arity 3 / 2 / 1).
   - Multi-output only: `if (rand() <= outProb)` (inclusive) draws `outIndex = engine() % outLen` and
     creates the out-node; the sentinel for "tagged" is a non-zero temporary subtree size
     (`GPNode{outNode, type | OUT_NODE, 1}`, generate.cu:86-101). The `outProb` comparison therefore
     consumes one RNG value per function node whether or not it is tagged, and the `engine()` call for
     the index consumes another one only when tagged.
   - If not tagged (or single-output), the node is `{value = float(k), type, subtreeSize = 1}`.
   - Pending children: `cdNew = {childs = type - 1, depth = cd.depth + 1}` (1, 2 or 3 children).
4. **Leaf branch**: `value = rand() <= constProb ? constSamples[engine() % constSamplesLen]
   : float(engine() % varLen)`, with `type = CONST` in the first case and `type = VAR` in the second;
   node `{value, type, 1}`. Leaves are **never** out-nodes.
5. `gp[topGP++] = node;`
6. `if (cd.childs > 0) childsAndDepth[top++] = cd;` (remaining siblings at the same depth)
7. `if (cdNew.childs > 0) childsAndDepth[top++] = cdNew;` (this node's children, at depth + 1)

Because the child entry is pushed **after** the sibling entry and the structure is a LIFO stack, the
children of a node are visited before its remaining siblings, and each entry carries its own depth, so
`gp` ends up in valid depth-first prefix order with correct per-node depths.

### 6.4 Leaf schedule (what the Python side must build)

`depth2leaf_probs` must be exactly 10 float32 values. `GenerateDescriptor` builds it as
`[layer_leaf_prob] * (max_layer_cnt - 1) + [1.0] * (MAX_FULL_DEPTH - (max_layer_cnt - 1))`
(`descriptor.py:33-39`), i.e. depths `0 .. max_layer_cnt-2` use `layer_leaf_prob` and depth
`max_layer_cnt-1` onward is `1.0`. Since `rand() < 1.0` always holds, `1.0` is the **hard terminator**
that bounds the depth and therefore the size. Consequences:
- `max_layer_cnt <= MAX_FULL_DEPTH` (10) is required so that slot 9 holds the terminator;
  `max_layer_cnt >= 11` can drop the terminator (`leafProbs[10]` OOB) and grow without bound.
- `max_tree_len` must be >= the worst-case complete tree of `max_layer_cnt` layers over the used
  functions (`check_tree_length`, `descriptor.py:8-31`) because the kernel never checks `topGP`.

### 6.5 Subtree-size pass (generate.cu:130-158)

- Reuses `childsAndDepth` as an `int* nodeSize` scratch stack and walks `i = topGP-1 .. 0` (reverse prefix):
  - VAR/CONST -> push 1;
  - UFUNC -> `size1 = pop(); push(size1 + 1)`;
  - BFUNC -> `size1 = pop(); size2 = pop(); push(size1 + size2 + 1)`;
  - TFUNC -> `size1..size3 = pop(); push(size1 + size2 + size3 + 1)`;
  - `gp[i].subtreeSize = nodeSize[top++]` (the just-computed value; `top` is decremented by the pops).
- The type test is `(nodeType & TYPE_MASK) <= CONST` for leaves, so out-node tags do not disturb the
  size computation.
- `len = gp[0].subtreeSize` and the first `len` entries of each of the three row tensors are written
  (values, types, sizes); the remaining `gpLen - len` entries stay uninitialized.

---

## 7. Splicing: `_gpTreeReplace` (mutation.cu:5-115)

The single primitive behind both `tree_mutate` and `tree_crossover`. It splices
`new[new_node_idx .. new_node_idx+new_subsize)` into `old[old_node_idx ..)` of a single old tree.

```
_gpTreeReplace(old_node_idx, new_node_idx, new_subsize, old_offset, old_size, size_diff,
               old{value,type,size}, new{value,type,size}, out{value,type,size})

  # per-thread scratch (MAX_STACK entries each): value_stack, type_stack, subtree_size_stack
1 for i in [0, old_node_idx):
      stack[i] = old[i]                                  # unchanged prefix (incl. the replaced node's parents' data)
2 current = 0
  while current < old_node_idx:                          # ancestor fix-up; root (old_node_idx == 0) skips it entirely
      stack.size[current] += size_diff                   # int16 arithmetic
      node_type = stack.type[current] & TYPE_MASK
      current += 1
      if current >= old_node_idx: break                  # the node at old_node_idx never gets size_diff
      switch node_type:
        UFUNC: break                                     # only child: current is already it
        BFUNC: right = current + stack.size[current]
               if old_node_idx >= right: current = right  # else stay in the left child
        TFUNC: mid   = current + stack.size[current]
               if old_node_idx < mid: break               # in the first child
               right = mid + stack.size[mid]
               current = (old_node_idx < right) ? mid : right
        default: break                                    # leaf: cannot happen on a well-formed path
3 for i in [0, new_subsize):
      stack[i + old_node_idx] = new[i + new_node_idx]     # donor subtree copied verbatim
4 for i in [old_offset, old_size):
      stack[i + size_diff] = old[i]                       # shifted tail; old_offset = old_node_idx + old.size[old_node_idx]
5 len = stack.size[0]; copy stack[0..len) to out[0..len)
```

- `size_diff = new_subsize - old.size[old_node_idx]`.
- Only the ancestor subtree sizes change (`+= size_diff`), including the root's; `stack.size[0]` after
  the splice is `old_size + size_diff` and becomes the new tree length.
- Every copied `size` value stays relative, so donor subtree sizes need no adjustment.
- The walk in step 2 uses the **old** types/sizes and assumes `old_node_idx` is a node boundary of a
  well-formed tree. Nothing validates that in this primitive.
- Tail copying assumes `old_offset == old_node_idx + old.size[old_node_idx]` (an exact node boundary);
  that is why the caller computes it from the old row rather than from `subtree_size_stack`.
- Scratch arrays are `MAX_STACK` entries: a resulting length > `MAX_STACK` silently corrupts memory
  (the callers only bound the length by `max_tree_len`).

### 7.1 `tree_mutate` driver (mutation.cu:118-219)

One thread per individual `n`; `old_node_idx = mutateIndices[n]`, `old_size = old_size_row[0]`:
- If `old_node_idx < 0 || old_node_idx >= old_size`: copy the old row unchanged (silent no-op).
- The donor is row `n` of `new_sub_forest` **in full**: `new_node_idx = 0`,
  `new_subsize = new_row.size[0]`.
- `size_diff = newSubtreeSize - old.size[old_node_idx]`, `old_offset = old_node_idx + old.size[old_node_idx]`.
- If `old_size + size_diff > maxGPLen`: copy the old row unchanged (silent no-op, no error).
- Otherwise splice. Output rows are freshly allocated (`torch::empty`) and only `[0, len)` is written;
  the source rows are never modified.
- Python contract (`Forest.mutate`, `forest.py:178-233`): `replace_pos` shape `(pop_size,)`,
  `new_sub_forest.pop_size == pop_size`, and matching `input_len` / `output_len` / `max_tree_len`.

### 7.2 `tree_crossover` driver (mutation.cu:224-309)

One thread per **offspring** `n < pop_size_new = len(left_indices)`; `left = left_idx[n]`,
`right = right_idx[n]` index the **parent population** (`pop_size_ori`):
- If `right < 0 || right >= pop_size_ori`: copy the left parent row verbatim (first `left_size` entries).
- Otherwise `left_node_size = left_row.size[left_node_idx[n]]`,
  `right_node_size = right_row.size[right_node_idx[n]]`, `size_diff = right_node_size - left_node_size`,
  `left_offset = left_node_idx[n] + left_node_size`.
- If `left_size + size_diff > maxGPLen`: copy the left parent row verbatim.
- Otherwise splice the right parent's subtree (`new_node_idx = right_pos`, `new_subsize = right_node_size`)
  into the left parent (`old_node_idx = left_pos`, `old_size = left_size`).
- **No validation** of `left_idx`, of `left_pos` or of `right_pos`: garbage indices read garbage
  `subtree_size` values from wherever they point and can produce a bogus (even negative) `size_diff`;
  the `maxGPLen` check is the only guard against overflowing the row. Callers must supply valid
  in-range positions themselves.
- Output pop size equals `len(left_indices)`; parents are read-only.

---

## 8. Determinism and RNG (`kernel.h:22, 159-182`)

- RNG = `thrust::random::taus88`, seeded per tree with `hash(n, keys[0], keys[1])`.
- `hash` is FNV-1a (offset basis `14695981039346656037`, prime `1099511628211`, 64-bit accumulator)
  over the 12 raw little-endian bytes of the array `{n, keys[0], keys[1]}`, truncated to `unsigned int`.
- `keys` was drawn in Python per call: `torch.randint(low=0, high=1000000, size=(2,), dtype=torch.uint32,
  device="cuda", requires_grad=False)` (`forest.py:51-58` in that snapshot), so a full run was reproducible
  via `torch.manual_seed` / `torch.cuda.manual_seed` but generation consumed no per-op torch CUDA RNG stream.
  The current pure-torch generator draws from torch's global RNG instead: seed-reproducible, but not
  bit-exact with the legacy taus88 stream (`generation.py` documents the distribution-equivalent deviations).
- `engine() % k` (used for constants, variables and the out index) is an unbiased-unaware modulo of the
  raw 32-bit state output.

---

## 9. Limits, edge cases and undefined behavior

- `MAX_STACK = 1024`, `MAX_FULL_DEPTH = 10` were duplicated between `defs.h:5` and `utils.py:10-11`
  because the kernels sized per-thread arrays with them; the current code defines them only in
  `tree/utils.py`, which is the single source of truth.
- The legacy kernels did not validate tree length against `max_tree_len` or `MAX_STACK`: undersized
  `max_tree_len` / `max_layer_cnt > 10` / `layer_leaf_prob = 0` produced silent memory corruption, not
  exceptions (the Python-side asserts in `GenerateDescriptor` / `check_tree_length` were the only guard).
  The current generator raises `RuntimeError` on a length overflow instead of corrupting memory.
- `top != 1` after an evaluation is only a debug `assert` plus a `printf`; in release builds evaluation
  continues and the single-output read `stack[--top]` takes the wrong slot.
- Out-of-range variable indices read outside the per-thread variable scratch; out-of-range function
  ids yield 0 (unmatched branch) except in the ternary case, which always computes `if`.
- A node tagged `OUT_NODE` handed to the **single-output** template is not masked: it reaches the
  ternary branch and is evaluated as `if` (pops 3 values), corrupting the stack. Multi-output and
  single-output trees are therefore not interchangeable.
- Values are unclamped (`MAX_VAL` clamping commented out); `DELTA`-based "loose" operators are the only
  numerical guards.

---

## 10. Appendix A — historical legacy-extension op schemas (`torch_wrapper.cu:291-306`)

**Historical reference only.** These five ops were the interface of the legacy compiled CUDA/HIP
extension that used to back this layer. The rewritten `evogp.tree` does not bind them any more:
generation and splicing are pure torch (`generation.py`, `splicing.py`) and evaluation/fitness run
through the ETL engine (`etl_engine.py`). The schemas are kept because those replacements were
required to reproduce the ops' exact semantics, and because the tables below double as the
specification of the call contract the old wrappers relied on.

All ops were registered under `TORCH_LIBRARY(<legacy_extension>, ...)` with a CUDA dispatch key and were
called as `torch.ops.<legacy_extension>.<name>(...)` from `tree.py` / `forest.py` only.
`float` in the schema = Python float (= C++ double). `int` = Python int (int64).
Argument order in the call is exactly the C++ parameter order below; tensor argument order in the
returned tuple is `(value, node_type, subtree_size)`.

| op | schema | argument list |
| --- | --- | --- |
| `tree_generate` | `(int i1, int i2, int i3, int i4, int i5, float f1, float f2, Tensor t1, Tensor t2, Tensor t3, Tensor t4) -> (Tensor t5, Tensor t6, Tensor t7)` | `pop_size, gp_len, var_len, out_len, const_samples_len, out_prob, const_prob, keys{2}, depth2leaf_probs{10}, roulette_funcs{29}, const_samples{const_samples_len}` |
| `tree_mutate` | `(int i1, int i2, Tensor t1..Tensor t7) -> (Tensor t8, Tensor t9, Tensor t10)` | `pop_size, gp_len, value_ori, type_ori, subtree_size_ori, mutateIndices{pop_size}, value_new, type_new, subtree_size_new` |
| `tree_crossover` | `(int i1, int i2, int i3, Tensor t1..Tensor t7) -> (Tensor t8, Tensor t9, Tensor t10)` | `pop_size_ori, pop_size_new, gp_len, value_ori, type_ori, subtree_size_ori, left_idx{pop_size_new}, right_idx{pop_size_new}, left_node_idx{pop_size_new}, right_node_idx{pop_size_new}` |
| `tree_evaluate` | `(int i1, int i2, int i3, int i4, Tensor t1..Tensor t4) -> Tensor t5` | `pop_size, gp_len, var_len, out_len, value, type, subtree_size, variables{pop_size, var_len}` -> `{pop_size, out_len}` |
| `tree_SR_fitness` | `(int i1, int i2, int i3, int i4, int i5, bool b1, Tensor t1..Tensor t5, int i6) -> Tensor t6` | `pop_size, data_points, gp_len, var_len, out_len, use_MSE, value, type, subtree_size, variables{data_points, var_len}, labels{data_points, out_len}, kernel_type` -> `{pop_size}` |

- Shared validation: every tensor must be a **contiguous CUDA** tensor with the exact shape above;
  `0 < gp_len <= MAX_STACK`, `0 < var_len`, `0 < out_len`, `0 < data_points`, `pop_size > 0`,
  `0 <= out_prob <= 1`, `0 <= const_prob <= 1`, and `depth2leaf_probs` / `roulette_funcs` /
  `const_samples` must have exactly 10 / 29 / `const_samples_len` elements.
- Tensors are checked for device, contiguity and shape — **not dtype**: a wrong dtype is silently
  reinterpreted as garbage.
- Output tensors are always freshly allocated `torch::empty` with `requires_grad=False`; tail entries
  beyond `size[0]` are uninitialized in every op that returns tree rows.
- The current wrappers assert `execute_mode` is one of `"torch"`, `"hybrid parallel"`, `"data parallel"`,
  `"tree parallel"`, `"auto"` and evaluate all of them through the same Python engine path; the
  `kernel_type` mapping of section 5.1 is historical.

---

## 11. Appendix B — Python call-site map

| capability | Python caller (in `src/evogp/tree/`) | notes |
| --- | --- | --- |
| generation | `Forest.random_generate(pop_size, descriptor)` (static) | pure-torch vectorized generation (`generation.py`); draws from torch's global RNG |
| evaluation | `Forest.forward`, `Forest.batch_forward`, `Tree.forward` | `Forest.forward` requires `x.shape[0] == pop_size` |
| SR fitness | `Forest.SR_fitness`, `Tree.SR_fitness` | accepts the five mode strings; all evaluate through the same Python engine path |
| mutation | `Forest.mutate(replace_pos, new_sub_forest)` | returns a new `Forest` |
| crossover | `Forest.crossover(left_indices, right_indices, left_pos, right_pos)` | returns a new `Forest` of size `len(left_indices)` |
| (not in this engine) | `Forest.zero_generate` (pure torch) | degenerate all-`CONST`-0 trees; needs no kernel |

- `MAX_STACK` / `MAX_FULL_DEPTH` / `NType` / `Func` are defined only in `tree/utils.py`, the
  single source of truth; the function ids must stay positional across `Func`,
  `FUNCS_NAMES`, `FUNCS_DISPLAY` and `SYMPY_MAP`.
- `descriptor.roulette_funcs` is `cumsum(dict2prob(using_funcs))` over the 29 slots (`descriptor.py:106-111`);
  `roulette_ufuncs` / `roulette_bfuncs` / `roulette_tfuncs` are only used by the torch-side mutation
  operators, not by this engine. `check_tree_length` rejects `max_tree_len` smaller than the
  worst-case complete tree for the chosen functions/`max_layer_cnt`.

---

## 12. Validation notes

- `Tree.__python_forward` (`tree.py:336`) is the in-repo **pure-Python reference implementation** of
  sections 3 / 4.1: a reverse prefix-order stack walk in which the first popped operand is the *first*
  (leftmost) child (`-` = `first - second`, `/` = `first / second`, the ternary `if` tests the first
  child), covering the complete op table — ternary `IF`, comparisons to ±1, the `MAX`/`MIN` NaN
  asymmetry, the loose/plain `log` / `sqrt` / `inv` / `pow` edge rules, `0.0` for unmatched function ids
  and the ternary fallback for non-standard base types. It is debug-only (name-mangled, single-output,
  slow) and is invoked by `Tree.__assert_valid`, so it is a cross-check reference for the evaluator,
  never the production path.
- The primary differential evidence for the rewritten engine was the **legacy extension oracle**: the
  replaced compiled ops were run on the same populations/inputs and the replacement's outputs compared
  against them (generation, splicing, evaluation and SR fitness). Those runs live in scratch space
  (`$TMPDIR`), not in this repository — re-running the comparison needs the old compiled extension and a
  CUDA/HIP GPU.
- `Tree.to_sympy_expr` was brought in line as well: the multi-output walk accumulates each tagged
  node's *own* value per `outIndex`, lets every function node pass its right-most child up, and maps
  relational results to the engine's ±1 (sections 2.3 / 4.3).
- Sympy renderers (`utils.SYMPY_MAP`, the `to_sympy_expr` walkers) are display artifacts, not the
  semantics. Both walkers map `Lt`/`Gt`/`Le`/`Ge` to the engine's ±1
  (`Piecewise((1, cond), (-1, True))`). Loose-op rendering matches the engine edge cases:
  `loose_log(0)` renders `-MAXVAL`, and `loose_div` / `loose_inv` render `x * copy_sign(MAXVAL, y)`
  for `|y| <= DELTA` (positive for an exact zero divisor, since sympy has no negative zero) and a
  plain quotient otherwise.
- Cheap self-consistency checks when reimplementing: (a) compare a new evaluator against an
  independently written CPU prefix-order tree walker (the same `-`/`/` operand order as section 4.1)
  on random inputs for `output_len == 1`, including NaN/inf cases; (b) check `size[0]` against an arity
  walk over `[0, size[0])`; (c) check that every `size[i]` equals `1 + Σ size[children]`; (d) after
  mutate/crossover, verify `size[0] == old_size + (donor_size - replaced_size)`, that only the
  ancestor path sizes moved, and that the output row is a byte-exact concatenation of
  `old[0 .. pos)` + `donor[dpos .. dpos+dsize)` + `old[pos + old_size[pos] .. old_size)`;
  (e) for multi-output, verify that each `outs[o]` equals the sum of the values of all out-nodes with
  `outIndex == o`, where each out-node's value is its own function value (not its rightmost child's).

---

## 13. The dense one-graph evaluation path (implementation, not semantics)

**Nothing in sections 1–12 changes.** A population can be evaluated by two interchangeable
implementations, and they are required to agree:

- the **specialized engine** (`etl_engine.py`, sections 3–5): one ETL graph per *tree structure*,
  one Python invocation and one device readback per tree;
- the **dense evaluator** (`dense_engine.py`): ONE ETL graph per *shape signature*, with the
  population's node tensors passed as runtime inputs (host-side vectorized encoding of
  opcode / function id / constant / variable index / child positions, a trace-time reverse-position
  unroll, chunked execution, one readback per call).

The dense graph computes every node value through the *same* private op emitters as the specialized
engine (`etl_engine._emit_unary` / `_emit_binary` / `_emit_ternary`), so the op table (§3), the
leftmost-child-first operand order (§4.1), the ±1 comparisons, the loose-op edge rules, the unary /
binary / ternary arity dispatch and the SR-fitness formula (§5) are identical by construction, and the
deviations listed in §12 apply to both. Section 4.3 (multi-output) is implemented by the specialized
engine only: a multi-output population makes the dense path decline and fall back.

Selection is per call, by the `EVOGP_EVAL_MODE` environment variable (an explicit `eval_mode=`
argument wins): `specialized` never uses dense, `dense` always attempts it, `auto` (the default)
resolves to *non-numpy backend, `pop_size >= EVOGP_DENSE_AUTO_MIN_POP` (16)* and — for `per_row=False`
calls, where the data-row count is known — *at most `EVOGP_DENSE_AUTO_MAX_ROWS` (2048) rows* (dense is
rows-linear there while the specialized engine is rows-flat; the wired call sites pass `pop_size` only,
so that half of the rule is enforced inside `evaluate_dense`). A population outside the dense subset
(multi-output, a tree longer than `EVOGP_DENSE_MAX_TREE_LEN` = 32, more rows than the `auto` ceiling,
any structural doubt) raises `DenseUnsupported` and is re-evaluated by the specialized engine, which
owns the error contract of §9 for malformed trees; that fallback is silent, every other exception
propagates.

Design rationale, the chunking / peak-memory model, all knobs, the numeric tolerances and the measured
economics live in **`DENSE_EVAL.md`** (this directory); the wiring is `Forest.forward` /
`Forest.batch_forward` / `Forest.SR_fitness` + `Tree.forward` / `Tree.SR_fitness`, with the public API
contracts (shapes, dtypes, `use_MSE`, `per_row`, the five `execute_mode` strings) unchanged.
