# Scorer correctness — 27 September 2026

GPU fixes were pushed as `46cec9a`. This next step makes scorer reuse and physical
energy diagnostics trustworthy before comparing model changes.

## Changes

- AF2, ESMFold and Boltz cache identities now include SHA-256 hashes of the actual
  checkpoint files. ESMFold uses a resolved local snapshot for both tokenizer and
  model loading; Boltz receives explicit checkpoint/cache paths and uses the same
  Python interpreter as its scorer. Relevant molecule files, force-field XML,
  preprocessing helpers and package versions are included. Weight changes during
  a scoring session are rejected; changed contents at the same path invalidate
  cached scores on the next invocation.
- AF2 and Boltz scoring use recorded seeds. Boltz's fork-specific attention flag
  is opt-in, and `--no-kernels` supports the portable Torch implementation.
- Each Boltz selection receipt identifies and hashes the exact confidence file
  and matching structure. Missing/ambiguous structures or wrong sample counts
  fail. Stock Boltz's absent interface PAE stays unknown.
- Scorer failures have reasons in their sidecars. OpenMM comparison uses the
  shared AF2/ensemble gate, including rejection by the second model.
- OpenMM records initial/final energies, iteration count, RMS and maximum forces,
  actual platform and convergence status. A capped, unconverged minimization
  cannot publish an energy score; CLI failure is nonzero and the score remains
  unknown. `--min-iters 0` is explicitly labeled raw and performs no minimization
  of the repaired complex (PDBFixer still prepares atoms/hydrogens).

OpenMM's force tolerance is the RMS over Cartesian force components, in
kJ/mol/nm. Static energy minimization now uses **no bond constraints**, allowing
that raw Cartesian gradient to be a meaningful convergence check. Previously
HBond constraints could leave finite constraint-balanced physical forces.
This is a versioned scoring-protocol change: do not compare these energies with
historical constrained/unverified energies as if the protocol were unchanged.
The default tolerance is 10; forces and finite coordinates are verified after
minimization rather than assuming that returning from the minimizer implies
convergence. See the [OpenMM API](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.LocalEnergyMinimizer.html).

Interaction energy remains a geometry/force-field diagnostic. It is neither a
binding free energy nor a calibrated predictor of experimental binding.

## Verification

The CPU suite passes **185 tests**, including same-path checkpoint replacement,
mid-run weight changes, snapshot file-set changes, target changes before scoring,
finite force QC, Boltz dispatch identity, and ensemble-aware comparison gates.

Real GPU checks on Bizon:

- OpenMM 8.3.1 with **OpenMM-CUDA-12 8.3.1**. Pin both: installing the old OpenMM
  CUDA extra alone selected an incompatible newer plugin in this environment.
- A generated complex scored raw with zero minimization iterations. A one-step
  minimization was rejected (RMS force approximately 5,559 kJ/mol/nm). With a
  5,000-step cap, it converged in 254 iterations (RMS approximately 5.75).
  Successful scoring and a subsequent zero-work cache hit were also verified
  through the CLI. Exact energies depend on repair seed and numerical geometry;
  no exact cross-platform OpenMM replay claim is made.
- AF2 explicitly loaded `model_1_ptm` from the recorded weight directory and
  reproduced the rounded design-time metrics (pLDDT 0.764, ipTM 0.406, iPAE 0.529).
  A second invocation reused the checkpoint-aware cache.
- Boltz succeeded with the existing confidence checkpoint after isolating
  RDKit 2025.9.6. The unrelated environment's RDKit 2023.09.6 could not deserialize
  the cached molecule files. The existing environment was left unchanged.
  Repeated seeded predictions gave ipTM 0.4203319549560547; the next invocation
  reused the cache. Missing interface PAE remained unknown.
- ESMFold GPU testing exposed a confidence-unit bug: Hugging Face returns
  fractions, while the scorer labeled its output 0–100. The scorer now validates
  the fractional range and multiplies by 100 before rounding. The immutable
  snapshot `75a3841ee059df2bf4d56688166c8fb459ddd97a` produced confidence 84.5/100
  and CA RMSD 0.84 Å. A second invocation reused the corrected cache. Legacy
  unconverted scores are invalidated when rescored. The loading warning about
  two ESM contact-head parameters concerns an unused contact-prediction head;
  this folding path consumes language-model hidden states instead.

Evidence and commands: [scoring-2026-09-27](docs/review/evidence/scoring-2026-09-27/).
The isolated remote test root remains `/home/bizon/projects/foldcraft-gpu-20260927`.
