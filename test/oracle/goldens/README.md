# EvoGP golden fixtures (dev-only, frozen)

Frozen `.npz` fixtures for the Triton `fused` route's golden check
(`python -m evogp.fused.check --stage goldens`, tolerance 1e-4 rtol/atol).
They were produced by the evoxc ETL engine of the pre-2.0.0a2 tree stack at repo
commit `0242bae1bc736f38c46d0d3951047dedf7a34305` (that engine was itself
validated against the old CUDA kernels and `ENGINE_SPEC.md`), so they are
historical captures: treat them as READ-ONLY data and never re-capture them.
Format: one `.npz` per case with `node_value (pop, max_tree_len) f32`,
`node_type i16`, `subtree_size i16`, `x (N, input_len) f32`, plus
`expected (pop, N, out_len) f32` (value path) or `expected_fitness (pop,) f32`
(+ `labels (N, out_len) f32`, `use_mse` bool) (SR loss path).
Cases:
- `01_random_single_out`: 16 random trees (max_layer_cnt=3), x 32x2 -> batched value path.
- `02_edge_finite`: 24 handcrafted finite edge cases (operand order, loose ops, comparisons, IF, VAR usage).
- `03_edge_nan`: 16 nan/-inf cases (div0, log0/neg, sqrt neg, inv0, MIN/MAX NaN asymmetry, 0^0, overflow).
  NaN-AWARE comparison required. NOTE: `pow(0,0)=1` and `pow(-2,3)=-8` follow SOURCE semantics
  (documented deviation from the old `-use_fast_math` kernels).
- `04_fitness_mse` / `05_fitness_mae`: same population as 01, x 32x2, labels 32x1.
- `06_multi_output`: `output_len=2` population, x 32x2 -> batched value path (out-node accumulation).
- `07_deep_skew`: one 401-node right-skewed chain, x 8x2.
The RNG state in the fixtures is data (not reproducible generation) based; generation
RNG is distribution-equivalent only.
