# Boltz throughput results — 27 September 2026

Reusable CLI processes reduced six-prediction Boltz batch wall time by a median
**15.2%**, equivalent to **18.0% higher batch throughput**,
with exact prediction parity. This is an independent-evaluator speed improvement;
AF2 design runtime and model accuracy are unchanged.

## Paired results

| Round | Fresh processes (s) | Reusable processes (s) | Wall-time reduction | Peak GPU MiB, fresh / reusable |
| --- | ---: | ---: | ---: | ---: |
| 1 | 115.72 | 98.09 | 15.2% | 8425 / 8425 |
| 2 | 114.89 | 98.93 | 13.9% | 8447 / 8957 |
| 3 | 116.49 | 97.62 | 16.2% | 8825 / 8425 |

Each batch contains six requests: IFNAR/Top7, PD-L1/ankyrin and EGFR/barrel,
each at two fixed seeds. Both arms use two workers. Three rounds alternate arm
order and include service startup and shutdown. OS page caches are warm. Scorer
sidecar initialization and the larger generation/evaluation campaign are outside
these timings. GPU memory is sampled once per second for the entire device,
including background/display allocations. These results do not establish a memory
saving or a whole-campaign speedup. Three repeats on one Bizon GPU characterize
this workload, not other hardware or long-running memory behavior.

The prespecified gates were exact artifact parity, at least 10% median wall-time
reduction and no round more than 5% slower. All passed. See the
[frozen plan](BOLTZ_THROUGHPUT_PLAN.md) and
[protocol](docs/review/evidence/throughput-2026-09-27/throughput-02/protocol.json).

## What changed

`baseline/boltz_worker.py` provides isolated reusable processes. Each request
executes the existing Boltz CLI with its own seed and output directory. Imported
modules and runtime startup are amortized; the model and trainer are still rebuilt
for every prediction. Skipping model initialization would alter random-number
consumption and needs a separate parity study. This change does not reduce the
number of diffusion steps, recycles, samples, or evaluated candidates.

The service processes are spawned without inheriting parent CUDA state. Within
them, the platform-native data-loader process context is restored (Linux/Python
3.12 uses fork in the tested Boltz CLI). Stable log streams support handlers retained
across requests, and request-owned models are garbage-collected with CUDA allocator
cleanup after each request. Prediction receipts and scorer signatures identify the
backend and worker implementation. Historical scoring records are not relabeled.

Enable the validated backend with:

```bash
python baseline/score_boltz2.py DESIGN_DIR --no-msa --no-kernels \
  --execution-backend persistent --workdir OUTPUT_DIR
python scripts/score_boltz_paired.py POOL_DIR \
  --execution-backend persistent --workdir OUTPUT_DIR
```

The paired scorer uses persistent workers by default; `--execution-backend
subprocess` restores fresh processes. The standalone scorer retains its subprocess
default; opting into persistent execution requires local single-sequence scoring,
portable kernels and no FlashAttention. GPU validation covers the installed
Boltz-2 fork, checkpoint and Bizon runtime recorded in the evidence. MSA-server and
accelerated-kernel combinations were not benchmarked.

## Validation and retained evidence

- 36 predictions across three paired rounds; all 18 persistent predictions and
  12 repeated fresh-process predictions exactly match their six fresh references.
- Equality covers complete confidence JSON, structure bytes, and all prediction
  NPZ arrays by shape, dtype and content hash. Paired inputs/settings are frozen.
- Both actual scoring CLIs completed on GPU with the persistent backend, including
  the paired scorer without an explicit backend flag. Repeating
  each CLI reused matching scores without creating or changing prediction artifacts.
- 255 CPU tests pass on macOS and Linux, including process reuse, failure recovery,
  retained log handlers, native process context, scorer routing and configuration
  guards. Runtime/source fingerprints are retained before and after the benchmark.
- The first four-prediction pilot had exact parity but was 6.8% slower due to worker
  startup behavior. Its data-loader context and retained-log-stream problems were
  fixed. The first full run was deliberately interrupted; it is not timing evidence.
  Both preliminary records are retained rather than excluded silently.

A separate cProfile diagnostic records startup/model-loading/inference costs in
[profile summary](docs/review/evidence/throughput-2026-09-27/throughput-validation-01/profile-summary.json).
Profiling overhead is excluded from the throughput comparison. Cumulative times
are nested and must not be added together.

Reports, receipts, confidence files, logs, source snapshots and a hash manifest are
in [the evidence directory](docs/review/evidence/throughput-2026-09-27/README.md).
Large raw structures, prediction NPZs and the full profile remain on Bizon under
`/home/bizon/projects/foldcraft-gpu-20260927/throughput-02` and
`/home/bizon/projects/foldcraft-gpu-20260927/throughput-validation-01`.

The next distinct hypothesis is model/checkpoint or trainer reuse with explicitly
preserved RNG semantics. This study does not demonstrate that optimization or an
accuracy improvement.
