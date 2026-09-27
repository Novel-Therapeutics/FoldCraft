# Boltz process-reuse evidence

`throughput-02` is the final, frozen three-round paired benchmark: 36 predictions,
exact output parity, and a 15.2% median reduction in six-request batch wall time.
Its protocol records sequences, seeds, settings and source/checkpoint hashes.
Runtime Python-source and package fingerprints match before and after execution.
The benchmark compares prediction-array content hashes, not compressed NPZ bytes.

`throughput-validation-01` tests both real scoring CLIs, the paired scorer's
implicit persistent default, and exact cache resume. Its separate cProfile run is
diagnostic only; it is excluded from batch timings. Profile cumulative times are
nested and affected by instrumentation, and must not be summed.

`throughput-pilot-01` retains the slower initial serial pilot (exact predictions,
6.8% slower) before fixing native data-loader context and retained log streams.
`throughput-01` was manually interrupted while those lifecycle corrections were
being prepared. It has no accepted batch timing. These preliminary runs are not
included in the final performance estimate; their original worker sources were
not separately archived. The `source` snapshots are the final validated code.

Both platform CPU logs record 255 passing tests. `final-state.json` confirms
unchanged benchmark source/checkpoint hashes and no remaining GPU compute process.
The final paired-scoring default was enabled after benchmark qualification and
validated by the actual CLI without an explicit backend flag.

Confidence files, selection receipts and logs are retained here. Full structures,
NPZ arrays and cProfile statistics remain under matching directories on Bizon:
`/home/bizon/projects/foldcraft-gpu-20260927/`. Receipts retain raw artifact hashes;
full numerical replay audits require those raw files or rerunning the protocol.
`evidence_manifest.json` hashes every evidence file except itself.
