# RU1303 late-alignment diagnosis and per-patient expression-scale audit

> Historical diagnostic record. The current future HTAN test input is the
> eight-patient RU942-updated file
> `/scratch/chanj3/wangm10/HTAN/results/aligned_scdebussy_preprocessed_v2_ru942_deg_all.h5ad`.
> The nine-patient numbers in this document were not recomputed when RU1646 was
> removed; Addendum B documents the available eight-patient RU942 comparison.

Input: `/scratch/chanj3/wangm10/HTAN/results/aligned_scdebussy_preprocessed_v3.h5ad`
(34,553 cells x 2,370 genes, 9 patients). Model: `scdebussy.align` at
`scripts/v2/real_data_config.json` (fixed landmarks, `n_landmarks=100`,
`consensus_resolution=100`, `locality_radius=0.1`, `beta=1`, `kappa=3`).
Scripts in this directory reproduce every number below; each fit takes ~1 min.

## Headline

RU1303's SCLC-A tail is aligned to the consensus region occupied by other
patients' SCLC-N **because the fit performed no warping at all**, not because of
expression similarity and not because of a trade-off against the cost of leaving
its tail unmatched. The alignment for RU1303 is the exact identity map.

## 1. The HTAN v2 fit is a no-op for six of nine patients

Posterior match marginals, baseline fit:

| patient | matched mass | insertion | mean displacement | max abs displacement |
|---|---:|---:|---:|---:|
| RU1083 | 1.00 | 0.00 | 0.0000 | **0.0000** |
| RU263  | 1.00 | 0.00 | 0.0000 | **0.0000** |
| RU581  | 1.00 | 0.00 | 0.0000 | **0.0000** |
| RU1444 | 1.00 | 0.00 | 0.0000 | **0.0000** |
| RU151  | 1.00 | 0.00 | 0.0000 | **0.0000** |
| RU1303 | 1.00 | 0.00 | 0.0000 | **0.0000** |
| RU831  | 0.74 | 0.26 | -0.0171 | 0.1919 |
| RU1646 | 0.38 | 0.62 | +0.0606 | 0.0606 |
| RU942  | 0.71 | 0.29 | -0.1568 | 0.2929 |

Every patient whose 100 landmarks are all matchable gets `A = I` to machine
precision. The three that move do so only because unmatchable landmarks make the
diagonal inadmissible.

Cell level: `mean |v2_position - s_local| = 0.0157` over all cells, and **94.0%
of cells have `|v2_position - s_local| < 0.01`**. For the six identity patients
the correlation is exactly 1.000000 and the maximum difference is 0.0000.
The published coverage 0.799 / purity 0.631 for `v2.1 fixed` are therefore
almost entirely properties of the input `s_local`, not of the alignment.

## 2. Why: gap costs exceed the entire alignment energy by ~10x

Calibrated `eta_i = eta_d = 9172`, `lambda_i = lambda_d = 27516`. The cheapest
departure from the diagonal in a one-to-one 100x100 lattice is one insertion plus
one deletion:

```
min warp cost = (lambda_i + eta_i) + (lambda_d + eta_d) = 73,376
total match energy of the identity path for RU1303 = 6,328
best achievable match energy ignoring all gap costs  = 6,000
```

The largest possible gain from *any* rearrangement of RU1303 (328) is 0.4% of the
price of the cheapest single warp step. The posterior is not making a trade-off;
the non-identity region of path space is unreachable at any temperature near
`beta=1`.

Calibration is the proximate cause. `eta` is the 90th percentile of band-limited
row-minimum costs **against the initial interpolated consensus**, computed once.
After EM the converged per-landmark match costs for RU1303 are ~63 on average —
`eta` is ~145x the cost scale of the model it is actually regularizing.

Gap-cost sweep (all other settings fixed) shows there is no useful regime:

| eta | RU1303 matched mass | RU1303 tail (landmarks 90-99) maps to | RU1303 SCLC-A median position |
|---:|---:|---|---:|
| 9172 (calibrated) | 1.00 | 0.909 - 1.000 | 0.820 |
| 1000 | 1.00 | 0.707 - 0.798 | 0.618 |
| 200  | 0.00 | unmatched | unprojected |
| 50   | 0.00 | unmatched | unprojected |

The model goes from rigid identity straight to total collapse. It never chooses
"match the early trajectory, gap the tail", which is the biologically sensible
alignment for a patient lacking the terminal state.

## 3. The expression evidence points the other way

For each RU1303 landmark, the best-correlated landmark position in each other
log-scale patient (2,370 genes, Pearson):

| RU1303 own t | RU1083 | RU581 | RU1444 | RU151 |
|---:|---:|---:|---:|---:|
| 0.81 | 0.57 | 0.69 | 0.54 | 0.54 |
| 0.86 | 0.57 | 0.67 | 0.53 | 0.53 |
| 0.91 | 0.57 | 0.45 | 0.52 | 0.52 |
| 0.96 | 0.57 | 0.44 | 0.51 | 0.52 |

RU1303's terminal cells correspond to other patients' **mid-trajectory SCLC-A**
region (0.44-0.69), not their SCLC-N region (0.78-0.93). Markers agree — mean
over RU1303's last five landmarks vs the consensus at the same coordinates:

| | ASCL1 | NEUROD1 | INSM1 | CHGA |
|---|---:|---:|---:|---:|
| RU1303 late | **2.61** | **0.04** | 1.43 | 1.29 |
| consensus late | 0.06 | 0.55 | 0.97 | 0.00 |
| RU581 late | 0.05 | 3.00 | 1.94 | 0.66 |
| RU1083 late | 0.02 | 1.62 | 0.54 | 0.00 |

RU1303's tail is ASCL1-high / NEUROD1-negative and is being matched against an
NEUROD1-driven consensus. The match is not supported by expression; it survives
only because no alternative path is affordable.

Control: with `consensus_resolution=150` against 100 landmarks the diagonal
becomes inadmissible, and RU1303's tail moves to 0.60-0.66 with its SCLC-A median
dropping 0.820 -> 0.545 — i.e. into the other patients' SCLC-A region, as the
expression evidence predicts. This is a diagnostic, not a fix: deletion mass
simply piles up at both consensus ends (0.70 / 0.89 in the first/last decile) and
cross-patient SCLC-A spread gets worse (0.398 -> 0.472).

## 4. Preprocessing: RU1646 is not log-transformed

Per-patient `X` summary:

| patient | X max | frac nonzero | nonzero mean | median source total counts |
|---|---:|---:|---:|---:|
| RU1083 | 9.62 | 0.486 | 1.54 | 10069 |
| RU263 | 9.51 | 0.607 | 1.68 | 13695 |
| RU581 | 9.66 | 0.548 | 1.51 | 13352 |
| RU831 | 8.98 | 0.586 | 1.49 | 12759 |
| RU1444 | 9.38 | 0.551 | 1.56 | 12239 |
| RU151 | 10.02 | 0.546 | 1.74 | 15950 |
| RU1303 | 10.65 | 0.460 | 1.84 | 9952 |
| RU942 | 10.52 | 0.561 | 1.37 | 18764 |
| **RU1646** | **458.17** | **0.128** | **8.94** | **1105** |

Decisive test (`transform_probe.py`): within a single cell, the implied scale
factor `X / counts` has coefficient of variation **exactly 0.0000** for RU1646 and
0.25-0.32 for every other patient, while `expm1(X) / counts` has CV 33.8 for
RU1646 and 0.22-0.38 for the others. RU1646's `X` is size-factor-normalized
counts with **no `log1p`**; every other patient is on a log scale. RU1646 is also
the shallowest library in the cohort (median 1105 counts) with only 173 cells.

Consequences measured:
- RU1646 alone raises the calibrated `eta` from 5475 to 9172 (+68%).
- The initial barycenter reaches 30.4 where no log-scale patient exceeds 8.7.

**But this is not what causes the RU1303 artifact.** Refitting with RU1646
rescaled and `log1p`-transformed gives `eta = 8762` and leaves all six identity
maps bit-identical; RU1303's tail still maps to 0.909-1.000 and its SCLC-A median
is still 0.820. Only RU831 / RU942 / RU1646 placements change. Fix it because it
is wrong, not because it will fix RU1303.

## 5. Second preprocessing defect: local-linear extrapolation

`_landmarks.estimate` fits a local linear regression per landmark. Where the
kernel holds 2-3 cells it extrapolates far outside the data range:

| patient | observed X max | landmark estimate max | landmark estimate min |
|---|---:|---:|---:|
| RU831 | 8.98 | 120.80 | -103.59 |
| RU942 | 10.52 | 647.61 | -691.28 |
| RU1646 | 458.17 | 4603.25 | -2995.22 |

The worst rows are `matchable=False` and are excluded from the posterior, but
several are matchable — e.g. RU942 t=0.42 (max 49.4, `n_eff=8.9`), RU942 t=0.41
(max 19.1, `n_eff=30.5`), RU831 t=0.70 (max 13.3, `n_eff=21.6`). These enter both
the cost matrix and the `eta` calibration. Mild negative values (~-0.1) appear at
the endpoints of every patient, which is the same effect at low amplitude. The
`n_min=5` count/ESS gate does not bound the slope term, so it does not catch this.

## Recommendations

1. **Do not report the current HTAN v2 numbers as alignment results.** For six of
   nine patients the returned alignment is the identity; coverage and purity
   reflect `s_local`.
2. **Recalibrate gaps against the converged cost scale, not the initial
   consensus**, or make `eta` scale-free (e.g. a quantile of the realized match
   costs, recomputed across EM). The current one-shot rule is off by ~2 orders of
   magnitude and is itself contaminated by items 4 and 5.
3. **The one-to-one lattice with `T_p == T` makes identity the unique
   gap-free path.** Decoupling the landmark budget from `consensus_resolution`
   only converts this into forced end deletions. Partial matching ("align the
   shared portion, delete the consensus tail") needs a terminal-deletion mode
   that is cheaper than interior gaps, or a many-to-one relaxation.
4. **Re-run RU1646 through the cohort preprocessing** (`normalize_total` then
   `log1p`) and add an assertion on per-patient `X` range across the cohort.
5. **Bound the local linear estimate** to the observed range of the contributing
   cells, or fall back to local-constant whenever the fitted slope implies an
   out-of-range value, not only when the design is singular.

## Open question this does not settle

Whether RU1303 truly lacks a late NE-high compartment or is undersampled (658
cells) is not answerable from the alignment. The evidence here shows only that
its observed terminal profile is ASCL1-high / NEUROD1-negative and resembles
other patients' mid SCLC-A region, and that the fit never tested that hypothesis.

---

# Addendum A: why the alignment does nothing

Three layers. Only the first is a tuning problem; the second is structural and the
third explains why EM reports convergence anyway. Reproduce with `why.py`,
`why2.py`, `why3.py`.

## A1. Gap costs dwarf the entire alignment energy (proximate cause)

`eta = 9172`, `kappa = 3`. A length-1 gap run costs `(kappa+1)*eta = 36,688`, and
because the lattice is one-to-one with `T_p = T = 100`, any departure from the
diagonal needs one insertion *and* one deletion: `2(kappa+1)*eta = 73,376`.

| patient | identity match energy | cheapest warp |
|---|---:|---:|
| RU1303 | 6,328 | 73,376 |
| RU1444 | 12,851 | 73,376 |
| RU1083 | 15,527 | 73,376 |
| RU151 | 24,178 | 73,376 |
| RU263 | 28,241 | 73,376 |
| RU581 | 33,161 | 73,376 |

One warp step costs 2-12x the total cost of aligning the whole trajectory. The
non-identity region of path space is unreachable.

`eta` is the 90th percentile of band-limited row minima taken **once, against the
initial interpolated consensus**, where costs are ~100x their converged values.

## A2. Retuning `eta` cannot fix it — the window is structurally empty

Two thresholds compete, per patient:

```
warp beats identity   requires  eta < best_warp_saving / (2*(kappa + k))
matching beats collapse requires eta > sum(D_identity) / (2*T)
```

Warping is only ever chosen in the window between them. Scanning every monotone
shift `k` and every window (`why2.py`):

| patient | kappa=3 warp / collapse | kappa=1 | kappa=0 |
|---|---|---|---|
| RU1083 | 61.7 / 77.6 — empty | 82.9 / 77.6 — exists | 119.8 / 77.6 — exists |
| RU263 | 105.4 / 141.2 — empty | 151.8 / 141.2 — exists | 241.1 / 141.2 — exists |
| RU581 | 120.4 / 165.8 — empty | 200.6 / 165.8 — exists | 376.0 / 165.8 — exists |
| RU1444 | 53.6 / 64.3 — empty | 67.1 / 64.3 — exists | 86.6 / 64.3 — exists |
| RU151 | 220.9 / 120.9 — exists | 308.3 / 120.9 — exists | 467.5 / 120.9 — exists |
| **RU1303** | **5.0 / 31.6 — empty** | **7.1 / 31.6 — empty** | **11.1 / 31.6 — empty** |

At the shipped `kappa=3` the window is empty for five of six patients. Dropping
`kappa` to 0 — no opening cost at all — opens it for five of six but **never for
RU1303**, which misses by a factor of 3. The patient that most needs a warp is
the one for which no gap parameterization will produce one.

The reason is a ratio, not a magnitude. Abandoning the whole alignment costs
`2*T*eta = 200*eta`; one warp costs `2(kappa+1)*eta = 8*eta` at `kappa=3`. So a
single warp spends 4% of the entire "give up" budget, and to be worth it must
save more than 4% of the total match energy — roughly four landmarks' worth of
cost, bought by a shift that moves smoothed landmarks past each other by one
position. Adjacent landmarks are 0.01 apart while `locality_radius = 0.1`, so
they share ~95% of their kernel weight; for RU581, RU1444 and RU263 a one-landmark
shift changes the cost by only 1.6-5%. The saving can never reach the threshold.

## A3. EM then confirms its own initialization

`initialize()` averages patients at equal consensus coordinates — that is, it
*assumes* the alignment is the identity. A1 and A2 prevent the E-step from ever
leaving it, and the M-step then refits the barycenter under the identity
alignment, which makes identity a genuine local optimum of the match cost.

Mean distance from each landmark to its own cost-minimizing consensus row
(`why3.py`), before any EM and after convergence:

| patient | at initialization | after EM |
|---|---:|---:|
| RU1303 | 31.8 | **3.7** |
| RU151 | 10.1 | **7.2** |
| RU1083 | 10.1 | 13.9 |
| RU581 | 17.7 | 19.2 |
| RU263 | 24.7 | **33.4** |

RU1303's cost surface is pulled onto the diagonal by the barycenter update —
agreement is manufactured, not found. RU263's is not: after convergence its
expression evidence still says "shift me 33 landmarks", and the returned
alignment is the identity anyway. The fit converges in 5 iterations without
having left its starting point, and `converged=True` says nothing about this.

## A4. What would actually have to change

- Recalibrate `eta` against the converged cost scale, not the initialization.
  Necessary, not sufficient — A2 shows tuning alone does not reach RU1303.
- Decouple the gap unit from the collapse unit. Interior warps and total
  non-matching are currently priced in the same currency (`eta`), which is what
  makes the window empty. A terminal/boundary deletion mode priced below interior
  gaps would let a patient align its shared portion and delete the consensus tail
  — exactly the alignment RU1303 needs.
- Stop placing landmarks 10x denser than `locality_radius`. At `n_landmarks=100`
  with `locality_radius=0.1` consecutive landmarks are near-duplicates, so the
  cost surface carries almost no gradient at the scale the lattice can act on.
- Do not initialize with the identity alignment and then report the fixed point
  as a result. At minimum, report displacement from `s_local`; a fit where
  `max|pos - s_local| = 0` for most patients should not be presented as an
  alignment.

---

# Addendum B: eight-patient cohort with the DEG-PCA RU942 pseudotime

Input: `/scratch/chanj3/wangm10/HTAN/results/aligned_scdebussy_preprocessed_v2_ru942_deg_all.h5ad`
(34,380 cells x 2,758 genes; RU1083, RU1303, RU1444, RU151, RU263, RU581, RU831,
RU942 — RU1646 dropped). Only RU942's pseudotime was recomputed from DEG-selected
PCA. Same config as the baseline fit.

## B1. The input pseudotime is better

Median `s_local` by cell type, RU942:

| version | LUAD | SCLC-A | SCLC-N |
|---|---:|---:|---:|
| v3 (current) | 0.103 | 0.804 | 0.784 (inverted) |
| DEG-PCA rerun | 0.203 | 0.730 | **0.822** |

RU942's SCLC-N now follows its SCLC-A, as expected. Cross-patient spread of the
median SCLC-N position falls from 0.422 (v3, nine patients) to 0.194. RU151 and
RU263 remain inverted (SCLC-A after SCLC-N) in both versions — a separate issue.

## B2. The alignment is still a no-op

`eta = 4339` (halved by dropping RU1646, as predicted in section 4).

| patient | matched mass | mean displacement | max abs displacement | mean \|pos - s_local\| |
|---|---:|---:|---:|---:|
| RU1083 | 1.00 | 0.0000 | **0.0000** | 0.0000 |
| RU263 | 1.00 | 0.0000 | **0.0000** | 0.0000 |
| RU581 | 1.00 | 0.0000 | **0.0000** | 0.0000 |
| RU1444 | 1.00 | 0.0000 | **0.0000** | 0.0000 |
| RU151 | 1.00 | 0.0000 | **0.0000** | 0.0000 |
| RU1303 | 1.00 | 0.0000 | **0.0000** | 0.0000 |
| RU942 | 0.91 | -0.0080 | 0.0909 | 0.0031 |
| RU831 | 0.74 | +0.1739 | 0.2626 | 0.1824 |

**91.1% of cells have `|v2_position - s_local| < 0.01`**; six of eight patients are
the exact identity. Every improvement in B1 comes from the input, and the
alignment adds nothing on top of it.

RU1303 is unchanged in kind: tail landmarks still map to 0.909-1.000, SCLC-A
median 0.750 against other patients' SCLC-N q10 of 0.744, so 60.5% of its SCLC-A
still sits inside the SCLC-N range (was 99.8% on v3 — the drop is a change in
`s_local`, not in the alignment). The warp/collapse window (A2) is again empty at
every `kappa` for RU1303: warp needs `eta < 1.87`, collapse occurs below 28.10.

## B3. Conclusion

Fixing RU942's pseudotime and dropping RU1646 both improve the inputs and are
worth keeping. Neither touches the RU1303 artifact or the no-op behaviour,
because both are properties of the gap parameterization and the lattice (A1-A3),
not of the data.

## B4. Robustness across all four available cohorts

The same six patients receive the exact identity map in every version of this
dataset on disk, despite `eta` varying by 2.1x:

| input | eta | identity patients | frac cells with \|pos - s_local\| < 0.01 |
|---|---:|---|---:|
| `preprocessed_v3` (9 patients) | 9172 | 6/9 | 0.940 |
| `preprocessed_v2` (8 patients) | 4902 | 6/8 | 0.877 |
| `preprocessed_v2_deg_all` | 4676 | 6/8 | 0.907 |
| `preprocessed_v2_ru942_deg_all` | 4339 | 6/8 | 0.911 |

Identity set in all four: RU1083, RU1303, RU1444, RU151, RU263, RU581 — exactly
the patients whose 100 landmarks are all matchable. The behaviour is a property
of the model, not of any particular preprocessing or pseudotime version.

---

# Addendum C: what an actual fix requires

Everything below is measured on the eight-patient DEG cohort unless stated.
Metrics are the repo's own `compute_patient_coverage` and
`cross_batch_knn_purity` (k=50), i.e. the numbers already used in
`validation/v21/README.md`. Scripts: `metrics.py`, `genes.py`, `fitnorm.py`,
`window.py`, `an7.py`.

## C0. The missing control: the unaligned input beats every published v2 result

No report in `validation/v2` or `validation/v21` scores the input pseudotime
itself. It should, because that is the bar an alignment has to clear:

| | coverage | purity |
|---|---:|---:|
| **v3, raw `s_local`, no alignment** | **0.8144** | **0.7356** |
| v3, v2.1 fixed landmarks (published) | 0.7989 | 0.6310 |
| v3, v2.0 (published) | 0.7433 | 0.7021 |
| v3, refit v1 (published) | 0.7178 | 0.5526 |
| **DEG cohort, raw `s_local`, no alignment** | **0.8650** | **0.7715** |
| DEG cohort, v2 as shipped | 0.8575 | 0.7285 |

Every published number is below doing nothing. On v3 the shipped v2.1 result is
0.105 worse in purity than its own input. This is consistent with the identity
finding: the six identity patients contribute exactly the input, and the three
patients the model does move get moved the wrong way.

**Add this control before anything else.** Without it, "v2.1 improved coverage
from 0.743 to 0.799" is not evidence of alignment.

## C1. The objective keys on patient identity, not on the trajectory

Decomposing `Delta[i,j] = mean_g precision[i,g]*(x[i,g]-B[j,g])**2` by gene:

| gene | share of total cost | what it is |
|---|---:|---|
| HLA-A | 5.36% | MHC-I, patient haplotype |
| HLA-B | 4.79% | MHC-I, patient haplotype |
| IGFBP2 | 1.96% | |
| HLA-C | 1.93% | MHC-I, patient haplotype |
| COX7A2 | 1.86% | housekeeping |
| IFI27 | 1.83% | interferon response |
| WFDC2, BEX3, TMA7, ECH1 | ~1.3% each | housekeeping / secretory |

Top 10 genes carry **23.0%** of the cost, top 100 carry **60.1%**. The three HLA
genes alone carry 12.1%. Meanwhile the markers that define the LUAD -> SCLC-A ->
SCLC-N axis are nowhere: ASCL1 rank 423 (0.028%), NEUROD1 rank 865 (0.007%),
CHGA 630, NCAM1 1093, DLL3 792 — together about 0.2% of the cost.

This is a batch-effect problem, not a weighting problem. The model has no
per-patient intercept, so any constant offset between a patient and the consensus
— HLA haplotype, interferon tone, library chemistry, ambient RNA — is charged as
alignment error at every position. It was not visible before because the model
was never allowed to act on it (Addendum A).

The precision floor is **not** the culprit: only 66 of 2758 genes sit at the
`1/epsilon` cap and they carry 0.1% of the cost.

**Removing patient-level offsets works.** Centering each gene within each patient
before fitting moves the lineage genes to the top of the cost decomposition:

| gene | rank before | rank after |
|---|---:|---:|
| ASCL1 | 423 | **12** |
| NEUROD1 | 865 | **58** |
| NCAM1 | 1093 | **63** |
| INSM1 | 174 | **83** |
| HLA-A share | 5.36% | 1.89% |

Pre-centering is the crude version; the clean fix is an explicit per-patient
intercept estimated in the M-step, so the offset is a model parameter rather than
a preprocessing step.

## C2. Freeing the warp — validated, and dangerous on its own

Addendum A2 showed no `eta`/`kappa` opens a warp window, because warping and
total non-correspondence are priced in the same unit. Two ways out, both tested:

**(a) Config-only, no code change: `consensus_resolution >> n_landmarks` with
`kappa_d = 0`.** Every path then deletes exactly `T - T_p` consensus rows, and
with zero opening cost the deletion total is constant across paths — so the
model is free to choose *where* the deletions go, which is a monotone warp.
This works. At `consensus_resolution=300, kappa_d=0`, `mean |pos - s_local|`
rises from 0.013 to 0.134, only 3.3% of cells stay put, all eight patients warp
(displacements to +/-0.45), EM takes 23 iterations instead of 6, and RU1303's map
becomes genuinely nonlinear, topping out at 0.776 instead of 1.000.

**(b) Code change: add DTW steps.** `_build_graph` in `_model.py` emits only
M=(1,1), I=(1,0), D=(0,1). Adding (1,2)/(2,1) steps at a separate small penalty
`omega` prices reparameterization separately from non-correspondence, which is
the principled fix. `_dp.py` consumes the graph generically, so inference needs
no change; but a landmark could then match more than one consensus row, so the
conservation laws documented in `docs/v2.md`
(`sum_j A[i,j] + insertion[i] = 1`) would have to be restated.

**Do not do C2 before C1.** Freeing the warp while the objective still tracks HLA
makes the result far worse:

| configuration | coverage | purity |
|---|---:|---:|
| raw `s_local` (do nothing) | 0.8650 | **0.7715** |
| v2 as shipped (identity) | 0.8575 | 0.7285 |
| warp freed, **no** centering | 0.4600 | **0.3822** |
| warp freed, **with** centering | 0.4062 | 0.6685 |

Centering recovers most of the damage (0.382 -> 0.668), which confirms C1 is the
dominant term. But the identity map was acting as an accidental safety net: it
was hiding a misspecified objective behind a no-op.

## C3. What is still missing

Nothing tested clears the do-nothing bar. Centering plus corrected gap costs at
the default geometry degrades monotonically as gaps get cheaper:

| centered, `kappa=1` | coverage | purity | frac cells unmoved |
|---|---:|---:|---:|
| `eta = 400` | 0.6700 | 0.6637 | 0.705 |
| `eta = 150` | 0.5813 | 0.5926 | — |
| `eta = 60` | 0.2950 | 0.3491 | — |

Every configuration that warps loses coverage, because in a one-to-one lattice
the only way to warp is to consume landmarks, which shortens the projected span.
This is the same structural fact as A2 seen from the other side, and it is why
(b) — real DTW steps that warp without consuming budget — is likely necessary
rather than optional.

## C4. Recommended order of work

1. **Add the unaligned-input control to every report.** Non-negotiable, and it
   invalidates the current replacement narrative on its own.
2. **Give the model a per-patient intercept** (or center per patient as an
   interim). Largest single effect measured; without it nothing else helps.
3. **Reconsider the gene panel.** Recurrent-HVG selection preferentially picks
   genes that vary *between* patients, which is exactly the confound. Select on
   variation along pseudotime *within* patient, reproducibly across patients.
4. **Then** fix the gap parameterization — (a) as an immediate config-level
   experiment, (b) as the real fix.
5. **Recalibrate `eta` against the converged cost scale**, not the initial
   consensus. Necessary, but note the window condition in A2 is scale-invariant,
   so this alone changes nothing.
6. **Preprocessing:** `log1p` RU1646; bound the local-linear landmark estimate to
   the observed range (sections 4 and 5).
7. Re-evaluate only after 1-3 are in place, against the C0 control, with the
   contract frozen first.
