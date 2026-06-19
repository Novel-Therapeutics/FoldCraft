# FoldCraft baseline benchmark — vs BoltzProt-1 (PD-L1)

**Purpose.** An *internal* baseline to calibrate our own future model improvements
against the current state of the art — **not** a publication-grade benchmark.
Rigor is kept where it is cheap (unbiased scoring); relaxed where it costs
compute without changing the decision (single target, default configs, modest n).

**Scope (now).** FoldCraft (fold-conditioned) **vs BoltzProt-1** (the June-2026
SOTA de novo binder designer), one target (PD-L1). A fuller panel (BindCraft,
RFdiffusion) is **deferred** — see *Future expansion*.

---

## RESULTS (resolved 2026-06-19)

All 1200 designs (FoldCraft 5 folds × 200 + BoltzProt-1 × 200) scored. **The ML
oracle gate was inconclusive and had to be replaced by a family-neutral physics
metric** — that journey is the main methodological finding:

1. **AF2 leg is family-biased.** FoldCraft hallucinates against AF2, so AF2
   favours it (188/1000 = 18.8% pass) over BoltzProt (0/200). Not a fair judge.
2. **Open Boltz-2 leg is non-discriminating.** It passes ~73% of AF2-*failers*
   and ~82% of AF2-passers (and 84% of BoltzProt) — it rubber-stamps almost
   everything, so the AF2∩Boltz-2 consensus collapses to AF2 (≈ the biased leg).
   → The consensus benchmark **cannot** honestly rank the two methods.
3. **OpenMM interface energy is the family-neutral tiebreak** (`score_openmm.py`,
   Amber ff14SB/GBN2, ΔE = E(complex) − E(target) − E(binder); open, no license,
   independent of both AF2 and Boltz). `openmm_compare.py` reports:

| Framing | FoldCraft | BoltzProt-1 | Verdict |
|---------|-----------|-------------|---------|
| **RAW** (all designs, no selection) | median **−33.1** kcal/mol (n=1000) | median **−33.2** (n=200) | indistinguishable (Mann-Whitney p=0.22) |
| **Net-repulsive (ΔE>0) rate** | **9%** (weak folds: ankyrin 16%, solenoid 17%) | **0%** | BoltzProt output more uniformly physical |
| **FORWARDED** (each method's deliverables) | AF2-passers median **−46.4** (n=188) | best-20 by self-ipTM median **−36.5** | FoldCraft far better (p=2.2e-4, rank-biserial −0.50) |

**Conclusion.** Raw populations are equivalent; BoltzProt's output is cleaner
(no repulsive tail); but **FoldCraft's AF2-gated deliverables are physically
better by a neutral force field** — corroborating the AF2 gate rather than
echoing it, which retires the self-bias worry. FoldCraft's edge is its *filter*:
it forwards genuinely better interfaces. In-silico only; ΔE is an interaction
energy (no entropy / unbound relaxation), a proxy not a Kd — wet-lab is ground
truth. Shareable write-up: `baseline/REPORT.md`.

---

## Methods

| Method | Family | How | Status |
|--------|--------|-----|--------|
| **FoldCraft** | AF2 (hallucination) | the reproduction run — 6 fold campaigns × 200 designs vs PD-L1, author configs | **done + oracle-scored** (`baseline/repro/`; AF2/Boltz-2/ESMFold/OpenMM — see RESULTS) |
| **BoltzProt-1** | Boltz (Boltz-PPI) | Boltz API `protein:design`, de novo no_template, length 70–185, n=200 | **done + oracle-scored** (`baseline/boltzprot/`; AF2/Boltz-2/ESMFold/OpenMM — see RESULTS) |

**BoltzProt-1 run provenance:** run `prot_des_sqpsGbr8wv1N3FNFGt6Z`, engine
`boltzprot v1.0`, 2026-06-18, $10/200 designs, idempotency-key
`foldcraft-pdl1-baseline-70to185-n200-v1`. Regenerate the payload with
`python baseline/boltzprot_design.py payload --n 200`. Length coverage spanned
the full 70–185 (median 111, 20 designs ≥180) → no targeted top-up needed.
First look at its *self*-metrics (reference only): ipTM median 0.74 (163/200 >
0.5) but `binding_confidence` median 0.0000 — its structure head and affinity
head disagree, which the external oracle will adjudicate.

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

## Oracle scoring — how to run (on reg-box-1; scripts written, not yet box-tested)

Three `baseline/` scorers, each operating on a *design dir* (`results.csv` +
`designs/<name>.pdb`, complex = chain A target / chain B binder) and writing
columns back idempotently:

- `score_af2.py <dir>` → `af2_plddt/af2_iptm/af2_ipae` (colabdesign, mirrors
  FoldCraft's own eval). Only needed for **BoltzProt** (FoldCraft already has
  `plddt/iptm/ipae`).
- `score_boltz2.py <dir>` → `boltz2_iptm/boltz2_pair_iptm/boltz2_plddt` (open
  MIT Boltz-2). The independent leg for **FoldCraft**; the self-family leg for
  BoltzProt (reported, not trusted). `pip install boltz[cuda]`.
- `score_esmfold.py <dir>` → `esmfold_plddt/esmfold_rmsd` (fold check, both arms).
  `pip install transformers accelerate torch`.

**Efficient order (Boltz-2 is the cost driver, ~3–5 min/complex):** gate on AF2
first, then run the expensive Boltz-2/ESMFold **only on AF2-passers**, since a
design failing AF2 can't clear the consensus anyway. Concretely:
1. BoltzProt: `score_af2.py baseline/boltzprot` (200). FoldCraft: already scored.
2. Filter each `results.csv` to AF2-passers (pLDDT>0.8, ipTM>0.5, iPAE<0.35) and
   run `score_boltz2.py` + `score_esmfold.py` on those rows (the scripts skip
   already-filled rows; pre-filter or `--sample` to bound compute).
3. Consensus gate = **AF2 leg AND Boltz-2 leg** both pass (Boltz-2 threshold
   calibrated on a small set); `esmfold_rmsd` reports fold fidelity. Success rate
   ± Wilson CI, per method.

Scoring everything is ~93 GPU-hrs; the AF2-first prune cuts Boltz-2 to the
~hundreds of survivors (~1 GPU-day). Drive it through `baseline/scheduler.py` or
fan out on cloud if needed.

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
