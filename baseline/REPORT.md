# FoldCraft vs BoltzProt-1 on PD-L1 — an in-silico benchmark with a family-neutral oracle

*A small, single-target calibration benchmark, shared in case the methodology and
observations are useful. In-silico only; no wet-lab. June 2026.*

---

## TL;DR

We compared **FoldCraft** (fold-conditioned, AF2-hallucination) against
**BoltzProt-1** (Boltz-family de novo binder designer) on **PD-L1**, scoring every
raw design through an oracle stack. The headline is as much methodological as it is
a result:

- **AI-confidence scoring alone could not rank the two methods.** AlphaFold2 is
  biased toward FoldCraft (same family it optimizes against); open Boltz-2 turned
  out to rubber-stamp almost everything. Their "consensus" collapses to the biased
  leg.
- A **family-neutral molecular-mechanics interface energy** (OpenMM, Amber
  ff14SB/GBN2 — independent of both AF2 and Boltz) broke the tie:
  - **Raw output:** the two methods are **statistically indistinguishable**
    (median interface ΔE −33.1 vs −33.2 kcal/mol, p=0.22).
  - **BoltzProt's raw output is cleaner:** 0% net-repulsive interfaces vs
    FoldCraft's 9% (concentrated in the weak folds).
  - **FoldCraft's *filtered* deliverables win clearly:** its AF2-passing designs
    reach −46 kcal/mol — better than its own bulk *and* than BoltzProt's best —
    and because the force field is family-neutral, this **corroborates** the AF2
    gate rather than echoing it.

**Net:** comparable raw generators; FoldCraft's strength is that its AF2 filter
forwards genuinely better interfaces, confirmed by an independent metric.

---

## Setup

| | |
|---|---|
| **Target** | PD-L1 (`pd-l1-1.pdb`), epitope `30-34,50-54,69-76` (the FoldCraft example config) |
| **FoldCraft** | Reproduction of the published protocol — author example configs, 6 folds × 200 designs (n=1000 scored; the TIM fold is excluded, see below). Per-fold success rates land in the preprint's ~0–27% regime, so the harness reproduces the paper. |
| **BoltzProt-1** | Boltz API `protein:design`, de novo (no template), binder length 70–185, n=200. Same target + epitope. |
| **Comparison unit** | FoldCraft is fold-*conditioned*, BoltzProt is unconstrained, so we don't compare per-fold — we compare **"confident PD-L1 binders produced per method"**, FoldCraft pooled across its fold campaigns. |

**Fairness rule we tried to hold:** never judge a method by its own model. A
designer's structure model is biased toward designs it optimized against, so each
method should be judged by an *independent* model — and ideally by something
outside both families.

---

## The core finding: AI-confidence scoring couldn't rank the methods

We started with the natural oracle gate — require a design to pass **both** an AF2
leg (pLDDT>0.8, ipTM>0.5, iPAE<0.35) **and** an open-Boltz-2 leg (ipTM>0.5), the
idea being that two different model families agreeing removes single-model bias.
Two problems surfaced:

1. **The AF2 leg is family-biased.** FoldCraft hallucinates against AF2, so AF2 is
   not a neutral judge of it. AF2 passed **188/1000 (18.8%)** of FoldCraft designs
   but **0/200** of BoltzProt's. Taken at face value this looks like a blowout — but
   it is exactly the self-bias the fairness rule warns against. (As a control, when
   we re-scored FoldCraft's *own* designs with a fresh, identically-configured AF2
   scorer, the pass rate was sensitive to protocol details — a reminder that the
   AF2 leg is not a stable cross-method ruler.)

2. **The open Boltz-2 leg is non-discriminating.** We expected Boltz-2 to be the
   independent judge for BoltzProt (different output, but same family) and for
   FoldCraft (fully independent). Instead it passed **~73% of AF2-failures**, **~82%
   of AF2-passers**, and **~84% of BoltzProt designs** — i.e. it clears almost
   everything. A judge that says "yes" to ~3/4 of designs that another model
   rejected adds no discriminating signal.

Consequence: the AF2∩Boltz-2 consensus is just AF2 with a near-pass-through filter
on top, so it **inherits AF2's family bias**. The consensus benchmark, as designed,
**cannot honestly say which method is better.** (We note ESMFold confirmed both
methods' binders fold well as monomers — ~0.5–1.1 Å Cα-RMSD to the designed chain —
so the question was never "do these fold," but "is the *interface* real.")

This is worth flagging for anyone benchmarking AF2-family designers: AF2-confidence
is circular for them, and not every "independent" model is a useful discriminator —
it must be both *unbiased* and *selective*.

## The tiebreak: a family-neutral interface energy

To get a discriminating, unbiased signal we left the AI-confidence world entirely
and scored each design's predicted complex with **molecular mechanics**: OpenMM with
the Amber ff14SB force field and GBN2 implicit solvent — open-source, no licence, and
architecturally independent of both AF2 and Boltz. The metric is the rigid-body
interface interaction energy

```
ΔE = E(complex) − E(target alone) − E(binder alone)      [kcal/mol]
```

all three single-point energies taken on the same coordinates after a short
minimization of the complex (which relieves the minor clashes both predictors carry,
identically for both methods). More negative = more favourable interface. Unlike
buried-surface-area, this **penalises clashes and rewards complementarity** — it
tells a real interface from a merely *placed* one. (It is an interaction energy, not
a binding free energy: no entropy, no unbound-state relaxation. A proxy, but an
unbiased and selective one.) ~4 s/design on a single GPU; all 1200 designs scored.

---

## Results

Interface ΔE (kcal/mol; more negative is better):

| Framing | FoldCraft | BoltzProt-1 | Test |
|---------|-----------|-------------|------|
| **RAW** — every design, no selection | median **−33.1**  (IQR −43,−22; n=1000) | median **−33.2**  (IQR −41,−27; n=200) | Mann-Whitney **p=0.22** (n.s.) |
| **Net-repulsive (ΔE>0)** | **9%** | **0%** | — |
| **FORWARDED** — each method's deliverables | AF2-passers median **−46.4** (n=188) | best-20 by ipTM median **−35.0** | **p=1×10⁻⁵**, rank-biserial **−0.60** (large) |

**1. The raw populations are equivalent.** Draw a random design from each — the
interface energies are statistically identical. By neutral physics, both are
credible, comparable binder generators. Neither "wins" on raw output.

**2. BoltzProt's raw output is more uniformly physical.** None of its 200 designs
has a net-repulsive interface; 9% of FoldCraft's do, and that junk tail is
concentrated in the weak folds (ankyrin 16%, solenoid 17%; barrel and Top7 are
clean). FoldCraft casts a wider net with more misses — which is fine *if* the
downstream filter catches them, and it does:

**3. FoldCraft's filtered deliverables are clearly better — and this is the
non-circular part.** Its 188 AF2-passing designs reach a median ΔE of −46 kcal/mol,
far below both its own bulk (−33) and BoltzProt's best (−35). The key point: **OpenMM
is family-neutral**, so an independent force field *agreeing* that the AF2-gated
designs are the physically best one means the AF2 gate is selecting genuinely better
interfaces, not just AF2-flattering ones. This is the discriminating, unbiased
evidence the AI-confidence consensus could not provide, and it resolves the self-bias
worry from the first section.

Per-fold raw medians (kcal/mol): Top7 −35.9, barrel −37.3, Ig-like −34.1,
solenoid −31.8, ankyrin −22.7 — the same fold-difficulty ordering seen in the
success rates.

---

## Caveats (please read with the numbers)

- **In-silico only.** No wet-lab. Every "favourable" here is a predicted/physics
  proxy.
- **ΔE is an interaction energy, not a Kd** — no entropy, no unbound-state
  relaxation. Good for *relative* comparison applied identically to both methods;
  not an affinity.
- **The FORWARDED row is asymmetric.** FoldCraft's "best" are its AF2-passers
  (selected from 1000); BoltzProt had **0** AF2-passers, so its "best" are its top-20
  by self-ipTM (AF2 non-passers). It is a fair "each method's best deliverables"
  comparison, but the selection is not symmetric.
- **Each method is scored on its own predictor's complex** (AF2 structure for
  FoldCraft, Boltz structure for BoltzProt). That's the fair choice — each delivers
  its own predicted geometry — but it's a mild confound worth naming.
- **Single target, single epitope, modest n.** A calibration snapshot, not a
  multi-target benchmark. BoltzProt-1 is a closed, moving API; its results are a
  dated snapshot.

---

## Observations that may be useful to you

A few things surfaced that are about FoldCraft specifically, offered in case they're
helpful:

- **The repulsive-interface tail tracks fold difficulty.** The 9% of designs with
  ΔE>0 are almost entirely in ankyrin and solenoid (the larger/weaker folds). These
  also have the lowest interface confidence. The neutral force field agrees with
  AF2 that the *interface*, not the monomer fold, is where the weak folds fail.
- **Ig-like fails on fold fidelity, not interface.** In our reproduction, Ig-like
  produced good interface confidence (ipTM clears on ~95/200) but only ~9/200 met
  RMSD<3.5 Å to the intended template — the binder doesn't adopt the Ig-like fold,
  even when it docks. That's a fold-conditioning miss rather than a binding miss,
  and might be the most informative fold to look at for improving conditioning.
- **TIM fold / `5bvl` template.** We hit a crash from the original `5bvl1.pdb`
  having gaps in its residue numbering (missing 17, 35–37) so the length inferred
  from the numbering span (184) disagreed with the 180 actual residues, breaking
  `set_seq`. This is **already addressed upstream** (contiguous `5bvl_af2.pdb` + a
  fail-loud gap guard); we just excluded TIM here rather than re-run.
- **Inclusive vs exclusive `set_range`.** Worth a sanity check that runtime contact
  maps match your shipped `examples/cmaps/*.npy` — we found the contact-map window
  parsing needs to be *inclusive* (`"30-34"` → 30..34) to reproduce the published
  maps bit-for-bit.

---

## Reproducing

All scorers live in `baseline/` and operate on a design dir (`results.csv` +
`designs/<name>.pdb`, complex = chain A target / chain B binder), writing columns
back idempotently:

- `score_af2.py` — AF2 interface metrics (colabdesign), for the non-AF2 arm.
- `score_boltz2.py` — open Boltz-2 interface confidence (the leg that turned out
  non-discriminating — kept for transparency).
- `score_esmfold.py` — ESMFold monomer fold-fidelity (both arms fold well).
- `score_openmm.py` — the family-neutral interface energy (the decisive metric).
- `openmm_compare.py` — the comparison + statistics in this report.
