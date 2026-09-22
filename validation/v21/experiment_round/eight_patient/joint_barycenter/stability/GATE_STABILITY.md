# Stability/generalizability test of the label-free largest-gap coverage gate

Follow-up to [AUTO_GATE.md](../AUTO_GATE.md), which showed the largest-gap split of the pre-fit
coverage score exactly reproduces the manually selected RU1303-only gate on the full 8-patient
cohort. This pilot tests only the **stability** of that rule — no alignment refit, no gate-rule
retuning, no cell-type label, held-out marker, or alignment outcome used anywhere. Pure analysis
of the coverage scores and their underlying peer-position data already on disk
(`../auto_gate/frozen_gate.json`). Code: [gate_stability.py](gate_stability.py); full numbers:
[stability_results.json](stability_results.json).

Baseline (all 8 patients, no perturbation): threshold **.515**, largest gap **.222**, second
gap **.172**, gap ratio **1.29**, gate on = {RU1303}.

## 1. Leave-one-patient-out (cohort-composition robustness)

| Dropped patient | Resulting gate | Threshold | Gap ratio | RU1303 isolated? |
|---|---|---:|---:|---|
| RU1083 | {RU1303} | .515 | 1.29 | yes |
| RU1444 | {RU1303} | .515 | 1.29 | yes |
| RU151 | {RU1303} | .515 | **1.05** (near-tie) | yes, but marginally |
| RU263 | {RU1303} | .515 | 1.29 | yes |
| **RU581** | **{RU1303, RU151, RU263, RU831, RU942}** | .823 | **1.05** (near-tie) | **no — flips to a 5-patient gate** |
| RU831 | {RU1303} | .530 | 1.47 | yes |
| RU942 | {RU1303} | .515 | 1.29 | yes |
| RU1303 (dropped itself) | {RU151, RU263, RU831, RU942} | .793 | 2.83 (very decisive) | n/a |

**6 of 7** non-RU1303 removals reproduce the exact RU1303-only split, most at the same decisive
1.29 gap ratio. **Dropping RU581 breaks it**: the second-largest gap in the original 8-patient
score list is specifically the RU151–RU581 boundary (.172); removing RU581 removes one side of
that gap, and the *next* largest gap in the remaining 7 scores (RU151–RU1444, .232) becomes only
marginally larger than the RU1303–RU831 gap (.222, ratio 1.05 — essentially a coin flip) — so the
rule switches to gating a 5-patient block instead of RU1303 alone. Dropping RU151 doesn't flip
the assignment but does shrink the winning margin to the same near-tie ratio (1.05), for the same
structural reason (RU151 sits right at the edge of the second-largest gap). Removing RU1303
itself yields a *different*, still-decisive split (ratio 2.83) between the mid-coverage and
high-coverage clusters — reassuring that the mechanism finds real structure, but not the
structure that isolates any one "RU1303-like" patient, since there isn't one left.

## 2. Nonparametric bootstrap of the coverage score itself

Each patient's score is the median of 7 peer-position values (already stored); resampling those
7 values with replacement (2000 replicates) and recomputing the score directly tests estimation
noise in the score, using only data already computed for the original score.

| | Frequency gated on |
|---|---:|
| RU1303 | **99.5%** |
| RU263 | 47.3% |
| RU831 | 45.4% |
| RU151 | 41.9% |
| RU942 | 32.6% |
| RU1444 | 18.0% |
| RU1083 | 15.9% |
| RU581 | 15.0% |

**Exact RU1303-only split frequency: 46.9%.** RU1303 itself is almost never excluded, but the
*other* seven patients are frequently swept in alongside it — most often the mid-coverage cluster
(RU263/RU831/RU151/RU942, each gated on 33–47% of replicates), consistent with the leave-one-out
finding that the second-largest gap (separating the mid-cluster from RU581/RU1444/RU1083) is only
modestly smaller than the largest gap and can flip under resampling. Threshold ranges from .404
(essentially isolating RU1303 alone at the boundary) up to .884 (sweeping in nearly everyone) —
mean .659, std .152, a wide spread relative to the baseline .515.

## 3. Additive score-noise perturbation sweep

| Noise σ | RU1303-only frequency | Threshold mean ± sd |
|---:|---:|---|
| .02 (≈ grid resolution) | 89.6% | .542 ± .087 |
| .05 | 72.1% | .582 ± .137 |
| .10 (≈ mid-cluster spacing) | 47.9% | .618 ± .171 |

At noise on the order of the underlying 100-bin grid resolution (σ≈.01–.02), the split is fairly
robust (90% exact recovery). Once noise reaches the scale of the actual gaps *within* the
mid-coverage cluster (σ≈.05–.10, comparable to the .01–.07 spacing between RU263/RU942/RU151),
robustness drops to 72% and then to a coin flip at 48% — matching the bootstrap's ~47% almost
exactly, which is reassuring cross-validation between the two perturbation methods rather than an
inconsistency.

## Visualization

[gate_stability.png](gate_stability.png): left panel — sorted coverage scores with the baseline
threshold (solid line), the bootstrap threshold's 95% range (shaded band, visibly wide, spanning
.48–.86 and reaching well into the mid-coverage cluster), and the three noise-sigma
threshold mean±sd markers. Right panel — patient × bootstrap-replicate gate-selection heatmap
(black = gated on): RU1303's row is almost solid black; RU263/RU831/RU151/RU942's rows show
substantial, visually obvious mixed black/white banding; RU1444/RU1083/RU581's rows are mostly
white with occasional black.

## Gate-selection frequency summary (bootstrap, all perturbation methods agree on direction)

Stable near 0% or 100%: RU1303 (~100%), RU1083/RU1444/RU581 (~15–18%, i.e. stably *off*).
Unstable, swinging widely with perturbation method and magnitude: RU151, RU263, RU831, RU942 (all
in the 33–47% range under the peer-position bootstrap) — this is exactly the mid-coverage cluster
that the *automatic* gate correctly left off in the point-estimate run, but whose exclusion is not
robustly separated from RU1303's cluster once realistic estimation noise is considered.

## Main question: is the largest-gap gate robust, or is the RU1303-only split fragile?

**Mixed, leaning fragile.** The rule is robust to:
- dropping any single non-RU1303, non-RU581 patient (6/7 leave-one-out cases unchanged),
- small, grid-resolution-scale score noise (90% recovery at σ=.02).

But it is fragile to:
- dropping RU581 specifically, which flips the gate from 1 patient to 5 outright,
- realistic estimation noise in the score itself: only **47% exact RU1303-only recovery** under
  a bootstrap that resamples from the actual observed peer-position spread — not a synthetic
  assumption, but a direct nonparametric estimate of the score's own sampling variability,
- noise at the scale of the real spacing between the mid-coverage-cluster patients (σ≈.05–.10),
  which is not an unreasonably large perturbation given the mid-cluster's own .010–.070 gaps.

The RU1303-only decision is correct and decisive *on the observed point estimates for this exact
8-patient cohort*, but the margin it wins by (gap ratio 1.29) is not large relative to what
resampling shows the score's own noise floor to be — roughly half of plausible resampled cohorts
would gate in some or all of RU151/RU263/RU831/RU942 alongside RU1303, and one specific,
plausible one-patient removal (RU581) flips the decision outright.

## Recommendation

**Do not freeze this threshold rule for external validation as-is.** The evidence here does not
support treating "largest gap in the coverage score" as a reliable, generalizable decision
procedure — it currently works for this cohort's point estimate, but the instability under
resampling and leave-one-out indicates the rule is picking up a real but narrow-margin signal
that a different (even modestly different) patient composition could invert. Consistent with the
task's stated fallback: **recommend replacing the hard largest-gap threshold with a more
principled latent coverage model** — e.g., a probabilistic/generative model of the coverage score
that reports gate membership with uncertainty (a soft/continuous cheap-price weighting, or a
posterior probability of "low coverage" per patient) rather than a single point-estimate 1D
split — rather than tuning the threshold further on this same HTAN cohort. Any such replacement
should itself be checked with the same leave-one-out/bootstrap protocol used here before being
adopted.

Stopping after this stability test, per the task's scope.
