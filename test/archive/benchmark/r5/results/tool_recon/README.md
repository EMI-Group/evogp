# `results/tool_recon/` — kernel-level profiling is unavailable this round

Evidence records for the R5 tool reconnaissance (`nsys` / `ncu`). There is no code here: this is the
"why the GPU-side mechanism is measured indirectly" proof for `RCA.md` (Q2).

## `import_log.txt`

Verbatim copy of `/tmp/evogp_r5recon/import_log.txt` (session-scoped scratch; sha256
`0ddd7fd996e0fe88aa8dd6c5cbccc2ead58b7e09a0ff41c46dc2a07f3f1d3eb4`, copied 2026-09-13), the stdout of
the `nsys` **injection + QDSTRM import/analysis** attempt on a trivial CUDA workload.

What it shows:

* the target-side **collection succeeds** (`Processing [... 100%]`, a `.qdstrm` is produced);
* the host-side **import/analysis then fails** with
  `QuadDCommon::InvalidArgumentException … = Unknown runtime API function index: 440`
  (`TraceProcessEvent.cpp(45)`, `GetCudaCallbackName`), i.e. nsys 2022.4.2.1 does not know a CUDA
  runtime API that the installed driver 535.247.01 / CUDA 12.4 exposes;
* the report that is nevertheless written (`dummy3.nsys-rep`) has **empty kernel tables** — kernel
  names/durations are unusable, so a Q2 timeline cannot be built from it.

The host toolchain has a standalone importer (`/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter`),
but it links the same analysis code and fails identically — there is no separate "import only, skip
analysis" path that would recover the CUDA backtrace.

## Tool identities on this box (2026-09-13)

| tool | version | state |
|---|---|---|
| `nsys` | `2022.4.2.1-df9881f` | collection works, **QDSTRM import/analysis fails** (`Unknown runtime API function index: 440`) |
| `ncu` | `2022.3.0.0` (build 31729285) | **unusable**: every profile attempt dies with `CUDA error: API call is not supported in the installed CUDA driver` |
| driver | `535.247.01` (CUDA 12.2 runtime line, torch reports CUDA 12.4) | too new for the 2022-era Nsight toolchain |

## Consequence for R5 (do not re-fight this)

* **No kernel-level counters, no kernel timeline.** Q2 (the GPU-side mechanism of the pure forward)
  is therefore pinned **indirectly**: by the pure-forward number itself, by the
  `--data-rows` row-scaling probe, by compile-side dispatch-graph dumps
  (`IREE_FLOW`-family flags via `ETL_IREE_COMPILE_ARGS`), and by `torch.profiler` (kineto), which
  does capture torch kernels on this box — one cheap retry is allowed to see whether it also sees
  IREE kernels, then the fallback plan is used.
* Per the round rules (CONTEXT.md): **at most two attempts per tool, then fall back and record**.
  The `.qdstrm` / `.nsys-rep` / `.sqlite` scratch files themselves are **not** committed — they live
  in `/tmp/evogp_r5recon/` (and `/tmp` is 97 % full); this log is the record of the failure.
