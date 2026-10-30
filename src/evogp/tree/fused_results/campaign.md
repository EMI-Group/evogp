# Fused campaign index

- schema: `evogp-fused-campaign-v1`
- created: 2026-10-30T20:33:39Z
- results: `/root/evogp/.genesis/workers/worker_T1_A61/src/evogp/tree/fused_results/main`
- probes: `/root/evogp/.genesis/workers/worker_T1_A61/src/evogp/tree/fused_results/probes`
- counts: ok=77 failed=0 missing=0
- cells: `p1000-L64-i32-N1024`, `p2048-L128-i32-N1024-d7`, `p256-L32-i8-N1024`, `p256-L64-i32-N1024`, `p256-L64-i32-N4096`, `p4096-L64-i32-N1024`
- arms: `evoxc-batch-iree`, `evoxc-dense-iree`, `evoxc-dense-iree-raised`, `evoxc-fused-iree`, `torch-batched`

## Configs (one row per cell x arm x workload)

| cell | arm | workload | status | median ms | cold ms |
| --- | --- | --- | --- | --- | --- |
| p1000-L64-i32-N1024 | evoxc-batch-iree | sr_fitness | ok | 7.2061 | 15951.3091 |
| p1000-L64-i32-N1024 | evoxc-batch-iree | batch_forward | ok | 6.9349 | 10530.0638 |
| p1000-L64-i32-N1024 | evoxc-batch-iree | forward | ok | 2.4343 | 5296.9280 |
| p1000-L64-i32-N1024 | evoxc-dense-iree | sr_fitness | ok | 5.6936 | 6609.0028 |
| p1000-L64-i32-N1024 | evoxc-dense-iree | batch_forward | ok | 7.3829 | 6371.5803 |
| p1000-L64-i32-N1024 | evoxc-dense-iree | forward | ok | 0.8690 | 4946.2462 |
| p1000-L64-i32-N1024 | evoxc-dense-iree | gp_step | ok | 12.6769 | - |
| p1000-L64-i32-N1024 | evoxc-fused-iree | sr_fitness | ok | 0.1435 | 472.9874 |
| p1000-L64-i32-N1024 | evoxc-fused-iree | batch_forward | ok | 0.1053 | 2.1153 |
| p1000-L64-i32-N1024 | evoxc-fused-iree | forward | ok | 0.0956 | 1.8008 |
| p1000-L64-i32-N1024 | evoxc-fused-iree | gp_step | ok | 2.8982 | - |
| p1000-L64-i32-N1024 | torch-batched | sr_fitness | ok | 4.9105 | 68.1843 |
| p1000-L64-i32-N1024 | torch-batched | batch_forward | ok | 4.8867 | 4.9251 |
| p1000-L64-i32-N1024 | torch-batched | forward | ok | 4.7694 | 4.8923 |
| p1000-L64-i32-N1024 | torch-batched | gp_step | ok | 68.0191 | - |
| p2048-L128-i32-N1024-d7 | evoxc-dense-iree-raised | sr_fitness | ok | 105.7609 | 14279.1407 |
| p2048-L128-i32-N1024-d7 | evoxc-dense-iree-raised | batch_forward | ok | 110.9945 | 14276.4718 |
| p2048-L128-i32-N1024-d7 | evoxc-dense-iree-raised | forward | ok | 1.2759 | 11160.3310 |
| p2048-L128-i32-N1024-d7 | evoxc-dense-iree-raised | gp_step | ok | 124.5304 | - |
| p2048-L128-i32-N1024-d7 | evoxc-fused-iree | sr_fitness | ok | 0.2005 | 469.4269 |
| p2048-L128-i32-N1024-d7 | evoxc-fused-iree | batch_forward | ok | 0.1673 | 2.1091 |
| p2048-L128-i32-N1024-d7 | evoxc-fused-iree | forward | ok | 0.1275 | 1.7466 |
| p2048-L128-i32-N1024-d7 | evoxc-fused-iree | gp_step | ok | 2.7468 | - |
| p2048-L128-i32-N1024-d7 | torch-batched | sr_fitness | ok | 26.2540 | 84.3819 |
| p2048-L128-i32-N1024-d7 | torch-batched | batch_forward | ok | 26.2510 | 26.2543 |
| p2048-L128-i32-N1024-d7 | torch-batched | forward | ok | 16.1361 | 15.8734 |
| p2048-L128-i32-N1024-d7 | torch-batched | gp_step | ok | 232.4833 | - |
| p256-L32-i8-N1024 | evoxc-batch-iree | sr_fitness | ok | 3.4803 | 3833.2589 |
| p256-L32-i8-N1024 | evoxc-batch-iree | batch_forward | ok | 3.3566 | 2674.3246 |
| p256-L32-i8-N1024 | evoxc-batch-iree | forward | ok | 2.0029 | 1540.2693 |
| p256-L32-i8-N1024 | evoxc-dense-iree | sr_fitness | ok | 1.1185 | 3396.8071 |
| p256-L32-i8-N1024 | evoxc-dense-iree | batch_forward | ok | 1.9857 | 3237.3167 |
| p256-L32-i8-N1024 | evoxc-dense-iree | forward | ok | 0.6651 | 2549.5090 |
| p256-L32-i8-N1024 | evoxc-dense-iree | gp_step | ok | 5.3738 | - |
| p256-L32-i8-N1024 | evoxc-fused-iree | sr_fitness | ok | 0.1299 | 469.2028 |
| p256-L32-i8-N1024 | evoxc-fused-iree | batch_forward | ok | 0.0995 | 2.0740 |
| p256-L32-i8-N1024 | evoxc-fused-iree | forward | ok | 0.0871 | 1.7089 |
| p256-L32-i8-N1024 | evoxc-fused-iree | gp_step | ok | 2.6473 | - |
| p256-L32-i8-N1024 | torch-batched | sr_fitness | ok | 4.1639 | 65.6331 |
| p256-L32-i8-N1024 | torch-batched | batch_forward | ok | 4.0547 | 4.1122 |
| p256-L32-i8-N1024 | torch-batched | forward | ok | 4.0248 | 4.1354 |
| p256-L32-i8-N1024 | torch-batched | gp_step | ok | 23.3581 | - |
| p256-L64-i32-N1024 | evoxc-batch-iree | sr_fitness | ok | 3.6764 | 4017.2363 |
| p256-L64-i32-N1024 | evoxc-batch-iree | batch_forward | ok | 3.4437 | 2797.1094 |
| p256-L64-i32-N1024 | evoxc-batch-iree | forward | ok | 1.8506 | 1601.2962 |
| p256-L64-i32-N1024 | evoxc-dense-iree | sr_fitness | ok | 2.0800 | 6595.6970 |
| p256-L64-i32-N1024 | evoxc-dense-iree | batch_forward | ok | 2.8460 | 6361.9841 |
| p256-L64-i32-N1024 | evoxc-dense-iree | forward | ok | 0.8475 | 5042.3366 |
| p256-L64-i32-N1024 | evoxc-dense-iree | gp_step | ok | 6.7315 | - |
| p256-L64-i32-N1024 | evoxc-fused-iree | sr_fitness | ok | 0.1399 | 1400.8983 |
| p256-L64-i32-N1024 | evoxc-fused-iree | batch_forward | ok | 0.1016 | 953.2324 |
| p256-L64-i32-N1024 | evoxc-fused-iree | forward | ok | 0.0911 | 330.7412 |
| p256-L64-i32-N1024 | evoxc-fused-iree | gp_step | ok | 2.6372 | - |
| p256-L64-i32-N1024 | torch-batched | sr_fitness | ok | 4.5626 | 65.5413 |
| p256-L64-i32-N1024 | torch-batched | batch_forward | ok | 4.4274 | 4.4847 |
| p256-L64-i32-N1024 | torch-batched | forward | ok | 4.4368 | 4.5586 |
| p256-L64-i32-N1024 | torch-batched | gp_step | ok | 28.3937 | - |
| p256-L64-i32-N4096 | evoxc-dense-iree | sr_fitness | ok | 5.8490 | 6587.8461 |
| p256-L64-i32-N4096 | evoxc-dense-iree | batch_forward | ok | 7.8461 | 6391.0790 |
| p256-L64-i32-N4096 | evoxc-dense-iree | forward | ok | 0.8995 | 4870.2112 |
| p256-L64-i32-N4096 | evoxc-dense-iree | gp_step | ok | 10.6846 | - |
| p256-L64-i32-N4096 | evoxc-fused-iree | sr_fitness | ok | 0.1282 | 461.8229 |
| p256-L64-i32-N4096 | evoxc-fused-iree | batch_forward | ok | 0.1095 | 2.2204 |
| p256-L64-i32-N4096 | evoxc-fused-iree | forward | ok | 0.0857 | 1.7359 |
| p256-L64-i32-N4096 | evoxc-fused-iree | gp_step | ok | 2.6795 | - |
| p256-L64-i32-N4096 | torch-batched | sr_fitness | ok | 4.9519 | 64.2643 |
| p256-L64-i32-N4096 | torch-batched | batch_forward | ok | 4.9177 | 4.9286 |
| p256-L64-i32-N4096 | torch-batched | forward | ok | 4.2940 | 4.4185 |
| p256-L64-i32-N4096 | torch-batched | gp_step | ok | 25.1004 | - |
| p4096-L64-i32-N1024 | evoxc-fused-iree | sr_fitness | ok | 0.1786 | 478.1242 |
| p4096-L64-i32-N1024 | evoxc-fused-iree | batch_forward | ok | 0.1460 | 2.1239 |
| p4096-L64-i32-N1024 | evoxc-fused-iree | forward | ok | 0.1070 | 1.7790 |
| p4096-L64-i32-N1024 | evoxc-fused-iree | gp_step | ok | 2.7304 | - |
| p4096-L64-i32-N1024 | torch-batched | sr_fitness | ok | 10.6602 | 68.5372 |
| p4096-L64-i32-N1024 | torch-batched | batch_forward | ok | 10.6400 | 10.6505 |
| p4096-L64-i32-N1024 | torch-batched | forward | ok | 4.5302 | 4.6050 |
| p4096-L64-i32-N1024 | torch-batched | gp_step | ok | 296.3993 | - |

## Probes

| cell | route engine | served | workload medians (ms) | cold (ms) |
| --- | --- | --- | --- | --- |
| p1000-L64-i32-N1024 | fused | True | batch_forward=0.1338, forward=0.0896, sr_fitness=0.1251 | batch_forward=433.8772, forward=1.6606, sr_fitness=29.8402 |
| p2048-L128-i32-N1024-d7 | fused | True | batch_forward=0.1707, forward=0.1205, sr_fitness=0.1871 | batch_forward=436.6938, forward=1.6656, sr_fitness=29.8802 |
| p256-L32-i8-N1024 | fused | True | batch_forward=0.0995, forward=0.0873, sr_fitness=0.134 | batch_forward=430.1395, forward=1.6444, sr_fitness=29.9315 |
| p256-L64-i32-N1024 | fused | True | batch_forward=0.1125, forward=0.0922, sr_fitness=0.1292 | batch_forward=435.778, forward=1.6362, sr_fitness=29.3957 |
| p256-L64-i32-N4096 | fused | True | batch_forward=0.1191, forward=0.0905, sr_fitness=0.1415 | batch_forward=432.2878, forward=1.6459, sr_fitness=29.384 |
| p4096-L64-i32-N1024 | fused | True | batch_forward=0.1478, forward=0.0984, sr_fitness=0.154 | batch_forward=442.5351, forward=1.5912, sr_fitness=29.7078 |
