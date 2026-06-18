# FoldCraft baseline benchmark — vs BoltzProt-1 (PD-L1)

**Purpose.** An *internal* baseline to calibrate our own future model improvements
against the current state of the art — **not** a publication-grade benchmark.
Rigor is kept where it is cheap (unbiased scoring); relaxed where it costs
compute without changing the decision (single target, default configs, modest n).

**Scope (now).** FoldCraft (fold-conditioned) **vs BoltzProt-1** (the June-2026
SOTA de novo binder designer), one target (PD-L1). A fuller panel (BindCraft,
RFdiffusion) is **deferred** — see *Future expansion*.

---

## Methods

| Method | Family | How | Status |
|--------|--------|-----|--------|
| **FoldCraft** | AF2 (hallucination) | the reproduction run — 6 fold campaigns × 200 designs vs PD-L1, author configs | **done** (`baseline/repro/`); needs oracle re-scoring |
| **BoltzProt-1** | Boltz (Boltz-PPI) | Boltz API: de novo binder design vs PD-L1, same epitope, ~100 designs | pending (needs API key) |

FoldCraft is fold-*conditioned* (the binder fold is specified); BoltzProt-1
designs unconstrained binders/nanobodies. The comparison is therefore not
per-fold but **"confident PD-L1 binders produced per method"**, with FoldCraft's
entry pooled across its fold campaigns.

## Fairness rules

1. **Same target + epitope.** Both design against `examples/targets/pd-l1-1.pdb`
   with the same epitope FoldCraft used: `30-34,50-54,69-76` (renumbered-from-1).
2. **Judge by an external oracle, never by self-scores.** A method's own model is
   biased toward its designs (it optimized against it). Because the two methods
   are from different families, the clean unbiased judge for each is the *other's*
   model:
   - FoldCraft (AF2-family) → judged by **Boltz-2** (independent).
   - BoltzProt-1 (Boltz-family) → judged by **AF2-multimer** (independent).
   - **Consensus** (both models agree) = the fair gate reported for both.
3. **Re-score raw designs.** BoltzProt-1 pre-filters/ranks its output (Boltz-PPI +
   developability); we score *its raw designs* through the same gate as FoldCraft,
   not its self-reported hit rate.

## Oracle stack (kept rigorous — scoring is cheap vs generation)

| Signal | Tool | Notes |
|--------|------|-------|
| Interface confidence (primary) | **AF2-multimer** + **open Boltz-2** | Report each + agreement. Boltz-2 gives ipTM / `pair_chains_iptm` / PAE. **Open Boltz-2 has no protein-protein affinity head** (small-molecule only) — so this is interface *confidence*, not a Kd. |
| Fold fidelity | **ESMFold** | Predict the binder monomer from sequence → RMSD/TM to intended fold. Independent of both families. |
| (reference only) | each method's self-score | reported but **not** used for the gate |

**Gate (per design).** Reuse the FoldCraft criteria as the AF2 leg
(pLDDT>0.8, ipTM>0.5, iPAE<0.35) and a matched Boltz-2 leg (ipTM/`pair_chains_iptm`
threshold TBD on a small calibration set). A design "passes consensus" if **both**
the AF2 and Boltz-2 legs pass. Report success rate ± Wilson 95% CI.

## Caveats (state with every number)

- **In-silico only** — no wet-lab; all "success" is predicted interface confidence.
- **BoltzProt-1 is a moving, closed API** — snapshot results once with the date and
  any version string the API returns; not bit-reproducible later.
- **Confidence ≠ affinity** — the calibrated protein-protein affinity (Boltz-PPI)
  is API-only; the local baseline measures interface confidence agreement.
- **IP** — PD-L1 is public, so sending it to the Boltz API is fine; proprietary
  targets must stay on the local MIT Boltz-2 oracle.

## Compute

- **BoltzProt-1:** API, ~hours, **$-tens within the $2k company launch credit**.
- **Oracle re-scoring:** Boltz-2 ~3–5 min/complex on a 4090 (~10–20/h); ESMFold
  seconds/seq; AF2-multimer for the ~100 BoltzProt designs (FoldCraft already has
  AF2 scores). Re-scoring ~1300 FoldCraft + ~100 BoltzProt designs ≈ ~1–2 GPU-days,
  parallelizable via `baseline/scheduler.py`.

## Open items (some need you)

- [ ] **BoltzProt-1 API key** (signup at api.boltz.bio; $2k company credits) — *you*.
- [ ] Confirm the live API schema vs `baseline/boltzprot_design.py` scaffold.
- [ ] Install **open Boltz-2** (`pip install boltz[cuda]`, MIT) + **ESMFold** on the
      GPU box (after the box re-sync).
- [ ] Box re-sync to reconciled `pure-tier-refactor` (still pending; box busy).

## Future expansion (deferred — for investor DD / publication)

Full panel adds **BindCraft** (AF2-hallucination; needs a **PyRosetta commercial
license**) and **RFdiffusion** (BSD; RFdiffusion→ProteinMPNN→AF2-initial-guess,
~3–5k backbones/target). Both ~few 4090-days/target; cloud fan-out (Lambda 8×A100,
~$200, ~1 day) is the cost-effective venue. Multi-target sweep + per-tool fairness
tuning + larger n also belong to that phase. Tooling assessments captured in the
session notes.
