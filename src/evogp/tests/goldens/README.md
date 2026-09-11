# EvoGP golden fixtures (engine-based)
Produced by the ETL engine (`src/evogp/tree/etl_engine.py`) at repo commit `0242bae1bc736f38c46d0d3951047dedf7a34305`
(engine was validated vs the old CUDA kernels / `ENGINE_SPEC.md`; see the tree node report).
Copy to `test/goldens/` (or point `$EVOGP_GOLDEN_DIR` here).

Format (per `src/evogp/tests/CONTEXT.md`): one `.npz` per case with
`node_value (pop, max_tree_len) f32`, `node_type i16`, `subtree_size i16`, `x (N, input_len) f32`,
plus `expected (pop, N, out_len) f32` (compare against `Forest(...).batch_forward(x)`)
or `expected_fitness (pop,) f32` (+ `labels (N, out_len) f32`, `use_mse` bool) (compare against `SR_fitness`).

Cases:
- 01_random_single_out: 16 random trees (max_layer_cnt=3), x 32x2 -> batch_forward.
- 02_edge_finite: 24 handcrafted finite edge cases (operand order, loose ops, comparisons, IF, VAR usage).
- 03_edge_nan: 16 nan/-inf cases (div0, log0/neg, sqrt neg, inv0, MIN/MAX NaN asymmetry, 0^0, overflow).
  NaN-AWARE comparison required. NOTE: `pow(0,0)=1` and `pow(-2,3)=-8` follow SOURCE semantics
  (documented deviation from the old `-use_fast_math` kernels).
- 04_fitness_mse / 05_fitness_mae: same population as 01, x 32x2, labels 32x1.
- 06_multi_output: output_len=2 population, x 32x2 -> batch_forward (out-node accumulation).
- 07_deep_skew: one 401-node right-skewed chain, x 8x2.
RNG in fixtures is data (not reproducible-generation) based; generation RNG is distribution-equivalent only.
