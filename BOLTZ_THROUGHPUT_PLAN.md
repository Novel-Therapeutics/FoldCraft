# Boltz throughput experiment (2026-09-27)

Compare isolated fresh CLI processes with reusable spawned CLI processes. Keep
model loading/initialization, diffusion settings, per-request seeds, data-loader
behavior and all generated output fields unchanged. Cache imported Python modules
only. Model or trainer reuse is out of scope because it can change RNG consumption.

Use the existing expanded benchmark pool: baseline seed 20261001, draw 0, for
IFNAR/Top7, PD-L1/ankyrin and EGFR/barrel. For each complex predict at seeds
20261021 and 20261022: six requests per batch. Both arms run two workers. Three
paired rounds alternate arm order (fresh/persistent, persistent/fresh,
fresh/persistent). Include worker startup and teardown in each batch's wall time.
OS page caches are warm; this is sustained local scoring, not machine cold boot.

Promotion requires exact equality of complete confidence JSON, structure bytes,
and every prediction NPZ array (shape, dtype and contents), median batch wall-time
reduction at least 10%, and no round more than 5% slower. Record peak total device
memory at one-second intervals, including display/background allocations. Do not
claim model accuracy improvement, AF2 speedup, or campaign-wide speedup from this
scorer-only benchmark. Three repeats on one GPU provide engineering evidence, not
population-wide confidence intervals.

The initial serial two-seed pilot reproduced predictions exactly but was 6.8%
slower: spawned service workers also spawned their data-loader children, and
retained log handlers referenced a closed stream. Restore the platform-native
child-process context within the service worker and use stable per-request log
streams. Stop the first full run before accepting timings, then repeat all timed
rounds using frozen corrected source. Retain initial/interrupted run receipts.

Validate real process reuse, native data-loader context, failure recovery and
retained logger behavior with a spawned CPU stand-in. Run the complete CPU suite
on macOS and Linux, then test the actual scoring CLI and cache resume on GPU.
Retain runtime/source/checkpoint fingerprints and exact-parity evidence.
