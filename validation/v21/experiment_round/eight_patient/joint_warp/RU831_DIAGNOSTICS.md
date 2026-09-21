# RU831 diagnostics: why does one patient look different in the joint-warp experiment?

This document is a diagnostic writeup, not a model-selection recommendation. The
joint posterior-warp graph (`warp="on"`) analyzed here is an experimental tool
used to understand RU831's alignment geometry; a separate audit of this branch
found it does **not** outperform the paired restricted-250-gene/unit-precision
`warp="off"` baseline (purity 0.850/0.785, `mean_abs_displacement` 0.009) on
purity or held-out concordance, so nothing below should be read as "the joint
model worked" in the Phase 4 sense. All findings here concern what the fitted
posteriors and cost surfaces look like for RU831 versus comparison patients,
not whether `warp="on"` should replace v2.

## Provenance

- Branch: `exp/ru831-diagnostics`, worktree `/tmp/scdebussy-ru831-diagnostics`.
- Commits produced by this task, in order:
  - `b3e9526` -- Part 1, cost-surface diagnostics for RU831/RU1303/RU1444 (pre-existing, from before this session's cutoff).
  - `9c1975f` -- Part 2 first pass: `posterior_diagnostics.py` (row entropy, multimodality, MAP-vs-mean path) and `displacement_underflow_check.py` (traced the landmark-level displacement gap to a numerical artifact), for RU831/RU1303/RU1444.
  - `7da7bac` -- Part 2 extension: added RU581 as a fourth comparison patient, plus `extended_diagnostics.py` (occupancy-thresholded displacement, leave-one-out refit, perturbed-initialization probe), written in response to a coordinator review.
  - `ad86e38` -- Part 3: `terminal_perturbation.py`, the one permitted targeted perturbation (`lambda_terminal_gap=15.5`, ~10x the "teq" reference of 1.5508).
  - This commit -- the present document.
- HEAD at time of writing: `ad86e38518f6fa60a7c8ff6fadda4bc4e3ee5794`.
- Input h5ad: `/scratch/chanj3/wangm10/HTAN/results/aligned_scdebussy_preprocessed_v2_ru942_deg_all.h5ad`, SHA256 `4f26e98a530bbda380121929d543b367884d0fb17beef58061de948ad09f7dc3` (re-verified during this session). 34,380 cells, 8 patients (RU1083, RU1303, RU1444, RU151, RU263, RU581, RU831, RU942); RU1646 is excluded from this cohort as in the earlier eight-patient checkpoint.
- Gene panel: the frozen top-250 trajectory panel, `validation/v21/experiment_round/eight_patient/feature_selection/top_250.json`.
- Exact commands (env: `/data1/chanj3/wangm10/conda_envs/scdebussy-py311`, run with `PYTHONPATH=src` from the worktree root):

```bash
python validation/v21/experiment_round/eight_patient/joint_warp/ru831_diagnostics/posterior_diagnostics.py \
  --adata /scratch/chanj3/wangm10/HTAN/results/aligned_scdebussy_preprocessed_v2_ru942_deg_all.h5ad \
  --genes-file validation/v21/experiment_round/eight_patient/feature_selection/top_250.json \
  --moderate-report validation/v21/experiment_round/eight_patient/joint_warp/joint_warp_o0.1_teq.json \
  --severe-report validation/v21/experiment_round/eight_patient/joint_warp/joint_warp_o0.02_teq.json \
  --patients RU831 RU1303 RU1444 RU581 \
  --output-dir validation/v21/experiment_round/eight_patient/joint_warp/ru831_diagnostics

python validation/v21/experiment_round/eight_patient/joint_warp/ru831_diagnostics/extended_diagnostics.py

python validation/v21/experiment_round/eight_patient/joint_warp/ru831_diagnostics/terminal_perturbation.py \
  --adata /scratch/chanj3/wangm10/HTAN/results/aligned_scdebussy_preprocessed_v2_ru942_deg_all.h5ad \
  --genes-file validation/v21/experiment_round/eight_patient/feature_selection/top_250.json \
  --base-report validation/v21/experiment_round/eight_patient/joint_warp/joint_warp_o0.1_teq.json \
  --patients RU831 RU1303 RU1444 RU581 \
  --output-dir validation/v21/experiment_round/eight_patient/joint_warp/ru831_diagnostics \
  --cache-copy-dir /data1/chanj3/wangm10/scDeBussy-v2/.htan_experiment_cache/exp-htan-eight-patient/joint_warp/ru831_diagnostics
```

- `extended_diagnostics.py`'s three refits (occupancy-threshold check, leave-one-out, perturbed-init) use the same adata/genes constants hardcoded at the top of the script -- see `validation/v21/experiment_round/eight_patient/joint_warp/ru831_diagnostics/extended_diagnostics.py:38-43`.
- The `.npz` cache from the terminal perturbation is at `/data1/chanj3/wangm10/scDeBussy-v2/.htan_experiment_cache/exp-htan-eight-patient/joint_warp/ru831_diagnostics/ru831_diag_high_lambda_terminal_gap.npz`, SHA256 `3a67f95c45e0c8693fb60e6f440548a851e9dd90c06a639469638fdea86dd314`, recorded in that directory's `MANIFEST.tsv` (copy-not-commit, matching the Phase 4 precedent).
- Biome's Node runtime is broken in this environment (too old a system `node`) for reasons unrelated to this task; every commit here used `SKIP=biome-format git commit ...` rather than `--no-verify`, so ruff/ruff-format and the other structural pre-commit hooks still ran and passed. This is noted once here rather than in every commit message.
- No architecture changes were made to `_barycenter.py`, `_model.py`, `_dp.py`, `_api.py`, or `_project.py`. The one runtime monkeypatch (`extended_diagnostics.py`'s perturbed-initialization probe, which replaces `scdebussy.align._api.initialize` for the duration of one refit and restores it immediately after) is a diagnostic-script-only substitution, not a library edit.

## Control-patient selection

**Rule, stated before the result:** among the six non-target patients (RU1083,
RU1444, RU151, RU263, RU581, RU942 -- excluding RU831, the patient under
investigation, and RU1303, the second comparison patient), compute the mean
and coefficient of variation (CV = std/mean) of each patient's cell-level
`per_patient[i]['mean_abs_displacement']` across the six existing joint-warp
candidate regimes (`o{0.02,0.05,0.1}_{teq,tlow}`). Rank each patient by mean
(ascending, lower is better) and by CV (ascending, more stable is better),
sum the two ranks with equal weight, and take the patient with the lowest
combined rank as the control.

| patient | mean | std | CV | rank(mean) | rank(CV) | combined rank |
|---|---|---|---|---|---|---|
| RU1444 | 0.0043 | 0.0001 | 0.0217 | 1 | 3 | **4** |
| RU581  | 0.0239 | 0.0003 | 0.0124 | 4 | 1 | 5 |
| RU263  | 0.0067 | 0.0002 | 0.0327 | 2 | 5 | 7 |
| RU942  | 0.0350 | 0.0007 | 0.0188 | 6 | 2 | 8 |
| RU1083 | 0.0071 | 0.0003 | 0.0368 | 3 | 6 | 9 |
| RU151  | 0.0252 | 0.0006 | 0.0224 | 5 | 4 | 9 |

For reference, RU831 and RU1303 (excluded from ranking, shown for scale):

| patient | mean | std | CV |
|---|---|---|---|
| RU1303 | 0.0090 | 0.0002 | 0.0188 |
| RU831  | 0.1504 | 0.0003 | 0.0023 |

**Conclusion: RU1444** is the control patient (lowest combined rank, low mean
displacement, moderate CV). RU581 is a close second (rank sum 5, best CV
of all six) and is used throughout this document as a second comparison
patient per a later review request, alongside RU1303. Note that RU831's own
CV (0.0023) is the lowest of any patient in the cohort at the cell level --
its cell-level displacement is unusually *large* but *stable* across regimes,
which is itself relevant to Part 2 below.

## Two displacement statistics that answer different questions

Two different quantities in this codebase are both informally called "RU831's
displacement," and they disagree:

1. **Cell-level, post-`project()`**: `per_patient[i]['mean_abs_displacement']`
   (`_api.py`, computed from `project()`'s per-cell conditional positions).
   For RU831 this is flat across all six joint-warp regimes: ~0.150-0.155
   with the widest split being `o0.1_teq`=0.1503 vs the no-warp
   `trajectory_250_unit` baseline's 0.1553 -- essentially the same number
   with or without any warp capability at all, CV=0.0023 (the lowest CV of
   any patient in the cohort). **Not bistable.**

2. **Landmark-level**, `diagnostics[patient]['mean_displacement']`
   (`_api.py:390`, computed via `_project.py:summarize()`'s `kinetic_curve`,
   normalized by `JointPosterior.patient_occupancy`). For RU831 this jumps
   between regimes: 0.115 (`o0.1_teq`) vs 0.246 (`o0.02_teq`). **Initially
   looked bistable** -- but see Part 2, where this gap is traced to a
   numerical artifact, not a real shift in posterior mass.

These are genuinely different quantities (one integrates over per-cell
conditional distributions with unprojected mass handled explicitly; the
other is a per-landmark occupancy-weighted average that, as shown below, is
vulnerable to underflow at landmarks with near-zero but nonzero occupancy).
**Only the landmark-level statistic showed apparent bistability, and Part 2
shows that even that apparent bistability is not real.** Anyone reading
`JOINT_WARP_RESULTS.md`'s displacement claims going forward should treat
`mean_displacement` (landmark-level) as unreliable at landmarks with very low
occupancy, and prefer the cell-level `mean_abs_displacement` (or an
occupancy-thresholded version of the landmark-level statistic) as the more
trustworthy summary.

## Part 1: cost-surface geometry (already committed, `b3e9526`)

Committed before this session's cutoff; summarized here for completeness.

- **Cost matrices shift substantially between the pre-joint fit and the
  joint-candidate fit** because the shared barycenter evolves during joint
  EM: RU831's cost matrix changes by 64% (max abs diff relative to median
  cost = 0.637), versus 4.8% for RU1303 and 7.9% for RU1444
  (`cost_matrix_identity_check.json`).
- **RU831 has only 80/100 matchable landmarks**, with a near-contiguous
  unmatchable block at local pseudotime ~0.707-0.889 (19 consecutive
  landmarks). RU1303 and RU1444 have 100/100 matchable landmarks.
- **RU831's row-argmin path has more abrupt jumps** (7, threshold 0.0303)
  than RU1303 (3) and is similar to RU1444 (5-6 depending on source:
  pre-joint vs joint-candidate; `cost_geometry_summary.csv`).
- **RU831's tail-residual-from-linear-fit is higher** (0.33-0.36) than
  RU1303 (0.26-0.28) or RU1444 (0.22-0.23) -- the cost-minimizing path
  deviates more from a straight line near RU831's tail.
- **Competing-minima fraction at the raw cost-surface level is low and
  similar across all three patients** (0-8% of rows have a second local
  minimum within the stated gap threshold; `cost_geometry_summary.csv`).
  This is **not** elevated for RU831 relative to RU1303/RU1444 -- the raw
  cost surface itself does not show unusual local ambiguity for RU831.

## Part 2: posterior diagnostics, corrected

### Row entropy, multimodality, MAP-vs-mean path (RU831, RU1303, RU1444, RU581)

Run with the corrected normalization (`log_a` normalized by
`JointPosterior.patient_occupancy`, matching `_project.py:summarize()`'s
actual reference implementation, not `logsumexp(log_a, axis=1)`; see
`posterior_diagnostics.py:145-188`'s docstring for the derivation). Both
regimes reproduce their cached `path_entropy`/`expected_match_11/12/21`
bit-exactly (`reproduction_check.json`, `abs_diff: 0.0` in every check).

| patient | regime | mean row entropy (nats) | n matchable | n multimodal rows | MAP-vs-mean mean |max diff| |
|---|---|---|---|---|---|---|
| RU831  | moderate (o0.1_teq)  | 2.752 | 80  | 0 | 0.0049 / 0.0141 |
| RU831  | severe (o0.02_teq)   | 2.743 | 80  | 0 | 0.0048 / 0.0139 |
| RU1303 | moderate             | 2.639 | 100 | 0 | 0.0035 / 0.0132 |
| RU1303 | severe               | 2.630 | 100 | 0 | 0.0033 / 0.0131 |
| RU1444 | moderate             | 2.620 | 100 | 0 | 0.0035 / 0.0127 |
| RU1444 | severe               | 2.614 | 100 | 0 | 0.0035 / 0.0126 |
| RU581  | moderate             | 2.615 | 100 | 0 | 0.0038 / 0.0119 |
| RU581  | severe               | (see posterior_summary.json) | 100 | 0 | -- |

Zero multimodal rows are detected for any patient in any regime under the
stated threshold (`gap_nats=1.0`, `min_component_mass=0.05`;
`multimodal_rows.csv`, `posterior_diagnostics.py:91-134`). Row entropy and
MAP-vs-mean disagreement are close across all four patients and both
regimes -- **RU831 shows no local (single-landmark) correspondence
multimodality distinguishing it from the three comparison patients.**

### The landmark-level displacement gap is a numerical artifact, not bistability

The initial draft of this document treated the moderate-vs-severe
`mean_displacement` gap (0.115 vs 0.246) as possible evidence of a bistable
correspondence. A coordinator review of a separate, independent audit found
this reading was very likely wrong and asked that it be checked directly
before being presented as a finding. It was checked, and the audit's
hypothesis is confirmed:

`displacement_underflow_check.json` (commit `9c1975f`) and this session's
independent re-derivation both trace the gap to the same mechanism. In the
severe regime, 19 of RU831's 100 landmarks (local pseudotime 0.7071-0.8889 --
the exact unmatchable block from Part 1) have `patient_occupancy` values of
`6.888341283549197e-15`: numerically nonzero (so `_project.py:summarize()`'s
`rows = np.isfinite(log_m)` mask includes them as "matched") but with **zero
finite entries in `log_a`** for those rows -- i.e. no real posterior support
whatsoever. `summarize()`'s conditional-mean position for such a row is
`weights @ consensus` where `weights` collapses to all-zero (since every
`log_a` entry in the row is `-inf`, `exp(-inf - anything) = 0`), so the
"mean position" evaluates to exactly `0`, and the resulting "displacement"
for that landmark is `0 - local_position`, i.e. exactly minus the landmark's
own coordinate (verified elementwise to a max abs diff of `1.1e-16` against
`kinetic_curve` computed via the real `summarize()`). Averaged over these 19
landmarks alone, this produces a spurious displacement of 0.798 -- almost
entirely responsible for the moderate-to-severe jump.

This session's `extended_diagnostics.py` directly tested the fix the audit
proposed:

- **Occupancy-thresholded recomputation** (`occupancy_thresholded_displacement.json`,
  threshold `patient_occupancy > 0.1`): restricting to well-supported
  landmarks removes the gap for RU831 almost entirely -- moderate 0.117 vs
  severe 0.115 (versus 0.115/0.246 unrestricted). Only 3 landmarks are
  dropped by the threshold in the moderate regime and 21 in the severe
  regime, and RU1303/RU1444/RU581 are unaffected by the threshold at all
  (0 landmarks dropped in either regime) because they have no near-zero-
  occupancy rows to begin with.
- The severe regime's 21 dropped rows include the same 19 landmarks (plus 2
  more with occupancy 3.4e-6 / 0.033, near but not below the underflow
  floor) identified independently by `displacement_underflow_check.py`.

**Conclusion: the landmark-level `mean_displacement` gap does not reflect a
genuine shift in posterior mass between regimes.** It is a display-statistic
artifact of averaging `kinetic_curve` (a quantity defined by
`_project.py:summarize()`, unchanged here) over landmarks with numerically
nonzero but physically meaningless occupancy. **Do not present the original
landmark-displacement statistic as evidence for multiple attractors.**

### Width-weighted reconciliation (the originally-requested check)

Before the occupancy-threshold check above, this session verified the
un-thresholded numbers exactly reproduce the cached scalars via direct
reconstruction from `map_vs_mean_path.csv`'s `mean_position`/`local_position`
columns (unweighted mean of `|mean_position - local_position|` over finite
rows: 0.1150 moderate / 0.1148 [computed over only 80 finite rows] vs cached
0.1150/0.2459). The apparent non-reconciliation for the severe regime traced
not to `_project.py:summarize()`'s width-weighting of `average[p]` (a
*different*, width-and-occupancy-weighted quantity that is not what
`diagnostics[patient]['mean_displacement']` actually reports -- see
`_api.py:390`, which uses the simple unweighted mean of `|kinetic_curve|`
over finite rows, not `average[p]`) but to `posterior_diagnostics.py`'s
`map_vs_mean_path()` using a stricter finiteness mask
(`np.isfinite(log_a).any(axis=1)`, 80 rows) than `summarize()`'s actual mask
(`np.isfinite(log_m)`, 99 rows) -- the 19-row difference between these two
masks *is* the artifact described above. So the width-weighting hypothesis
was checked and ruled out; the real mechanism is the occupancy-underflow one.

### Internal vs. terminal unmatched mass

RU831's `o0.1_teq` fit has substantial **internal** unmatched mass but
negligible **terminal** unmatched mass:

| patient | patient_unmatched_internal_mass | patient_unmatched_terminal_mass |
|---|---|---|
| RU831  | 37.27 | 0.245 |
| RU1303 | 19.99 | 0.709 |
| RU1444 | 16.01 | 0.518 |
| RU581  | 16.63 | 0.562 |

RU831's internal-unmatched mass is roughly double RU1303's and more than
double RU1444's/RU581's, while its terminal-unmatched mass is the *smallest*
of the four. Combined with the raw h5ad population structure (below), this
means RU831's story is **not** "lacks a terminal SCLC-N state" -- it has a
real, if small, terminal SCLC-N population (n=140, median s_local=0.9992,
range 0.44-1.00) beyond an internal density gap, not a hard truncation. The
correct framing is an **internal/intermittent support gap**, not a
terminal-truncation problem.

## Part 3: terminal-gap perturbation (10x teq)

`terminal_perturbation.py` refit all 8 patients once with `omega_warp=0.1`,
`lambda_terminal_gap=15.5` (~10x the teq reference 1.5508), and compared
against the cached `o0.1_teq`/`o0.1_tlow` baselines. Because RU831's terminal
unmatched mass was already small at teq (0.245, see above), the informative
question is not "does raising the terminal penalty change RU831's tail" but
whether it changes anything about **internal** correspondence, given the
internal gap is the larger structural issue.

Result: for RU831 specifically (compared against RU1303/RU1444/RU581):

- Terminal-unmatched mass (patient and consensus) drops to ~0 for **all
  four patients**, not RU831-specifically -- this is the generic effect of
  a much larger terminal penalty, and `endpoint_pileup` saturates at 2.0 for
  every patient.
- **Internal correspondence structure barely moves for any patient**:
  `expected_match_12`, `expected_match_21`, `path_entropy`, and
  `patient_unmatched_internal_mass` all shift by <1% between teq, tlow, and
  the high-lambda perturbation, for RU831 and for all three comparison
  patients alike (`terminal_perturbation_comparison.json`).
- The `terminal_perturbation_interior_summary.json` interior-region check
  (10th-90th percentile of each patient's matchable landmarks) shows RU831's
  interior correspondence mass sum (49.5) and mean row-max probability
  (0.091) are close to, if slightly lower than, RU1303/RU1444/RU581 (~63-67
  and ~0.103-0.104) -- consistent with RU831 simply having fewer/sparser
  matchable interior landmarks (64 vs 80 for the others), not a
  qualitatively different interior correspondence pattern.
- RU831's landmark-level `mean_displacement` is ~0.115 at teq but ~0.245-0.247
  at **both** tlow (*lower* lambda) and high-lambda (*10x higher* lambda) --
  the same non-monotonic, U-shaped pattern documented in Part 2's
  displacement-underflow analysis, not a real terminal-perturbation effect.
  (tlow's very low terminal penalty and high-lambda's very high terminal
  penalty both apparently push a few landmarks' occupancy through the
  near-zero-but-nonzero underflow regime that produces the artifact; teq
  sits in between and avoids it. This U-shape is itself a useful diagnostic
  of how fragile the raw landmark-level statistic is, and is a further
  reason not to trust it uncorrected.)

**Interpretation:** raising the terminal-gap penalty 10x changes only the
tail (terminal-unmatched mass, endpoint pileup), not RU831's internal
SCLC-A-adjacent correspondence pattern -- and this holds equally for the
comparison patients. This argues against a pure endpoint/open-mode artifact
being responsible for RU831's elevated cell-level displacement; it is
consistent with the internal/intermittent support-gap explanation instead.

## Part 3b: leave-one-out and perturbed-initialization probes

Two further checks, run in response to the coordinator review
(`extended_diagnostics.py`):

**Leave-one-out (exclude RU831), moderate regime:**

- Excluding RU831 shifts the shared barycenter by 5.7% (relative Frobenius
  norm) overall, and 6.3% restricted to the consensus region overlapping
  RU831's unmatchable block (0.65-0.92) -- a ratio of 1.09, i.e. only mildly
  concentrated there, not sharply localized.
- RU1303's landmark-level `mean_displacement` roughly halves without RU831
  present (0.0131 -> 0.0064); RU1444's and RU581's are essentially unchanged
  or slightly higher (0.0036->0.0045, 0.0170->0.0193).
- **Interpretation:** RU831 does measurably pull on the shared barycenter
  (consistent with Part 1's 64% cost-matrix shift for RU831 between
  pre-joint and joint-candidate fits, much larger than RU1303's 4.8% or
  RU1444's 7.9%), and that pull disproportionately affects RU1303 rather
  than RU1444/RU581 -- but the effect is not sharply concentrated in RU831's
  own problem region, so this is evidence of a real but modest
  barycenter-compromise effect, not a dramatic one.

**Perturbed initial barycenter (moderate regime only, +/-5% deterministic
sinusoidal offset to the initial barycenter, monkeypatched at the script
level, not in any library file):**

- The perturbed fit converges to the same free energy as the baseline to
  ~1e-9 relative difference (`-501.4805056...` vs `-501.4805056...`), the
  same iteration count (7), and the same `mean_displacement` to ~1.7e-8
  absolute difference for every one of RU831/RU1303/RU1444/RU581.
- **Interpretation:** no evidence of a second attractor basin reachable from
  this single, modest perturbation. This is a limited, single-perturbation
  test -- `initialize()` in `_barycenter.py` is deterministic given
  landmarks/coordinates, so exhaustive multi-start testing is not available
  without a much larger battery of differently-shaped perturbations, which
  was out of scope here. **This axis (optimization/local-minimum behavior)
  remains only partially tested; treat this single negative result as
  suggestive, not conclusive.**

## Cell-population cross-check from the raw h5ad

Directly from `adata.obs` (`cell_type_final2`, `s_local`), independent of any
fitted model:

| patient | LUAD n (s_local range) | SCLC-A n (median, range) | SCLC-N n (median, range) |
|---|---|---|---|
| RU831 (target) | 43 (0.00-0.19) | 2,261 (0.42, 0.25-0.61) | 140 (0.999, 0.44-1.00) |
| RU1303 (comparison) | 210 (0.00-0.71) | 448 (0.75, 0.28-1.00) | **0 -- no SCLC-N cells at all** |
| RU1444 (control) | 258 (0.08-0.69) | 883 (0.49, 0.01-1.00) | 1,482 (0.81, 0.43-1.00), 56.5% of cohort |
| RU581 (2nd comparison) | 393 (0.04-0.41) | 13,581 (0.48, 0.00-0.96) | 795 (0.92, 0.51-1.00) |

RU831's SCLC-A population ends around s_local ~0.61 and its SCLC-N
population is small and pinned near the end (median 0.9992) -- this
corroborates that the unmatchable-landmark gap at local pseudotime 0.71-0.89
found in Part 1 is a genuine density gap between real cell populations, not
a model-manufactured artifact of the fit. RU1444 (the selected control) has
a large, well-distributed SCLC-N population across more than half its
s_local range. **RU1303 is a structurally different problem: it has zero
SCLC-N cells at all**, so its unmatched/warp behavior (Part 1: 8% competing-
minima rows, the highest of the three original patients, and unmatchable
rows concentrated near its LUAD/SCLC-A-adjacent range) reflects a different
underlying biology than RU831's internal gap, not a comparable case of the
same phenomenon.

## Part 4: comparison table

| patient | cost geometry | support truncation | multimodality evidence | posterior uncertainty | endpoint sensitivity | likely issue |
|---|---|---|---|---|---|---|
| **RU831** | Cost matrix shifts 64% between pre-joint and joint-candidate fits (largest of the three); 7 abrupt row-argmin jumps; tail-residual-from-linear-fit 0.33-0.36 (highest of the three) | 80/100 matchable landmarks; near-contiguous unmatchable block at local pseudotime 0.71-0.89, corroborated by a real gap in raw SCLC-A (ends ~0.61) vs. small terminal SCLC-N (median 0.9992) populations | 0 multimodal rows in either regime; MAP-vs-mean disagreement small (<0.014); landmark-level `mean_displacement`'s apparent moderate-vs-severe gap (0.115/0.246) is a numerical-occupancy-underflow artifact, not real bistability (removed by occupancy-thresholding) | Row entropy ~2.74-2.75 nats, similar to controls; cell-level displacement flat and low-CV (0.150-0.155, CV=0.0023) across all regimes and even the no-warp baseline | 10x terminal-gap perturbation changes only the tail (terminal mass -> ~0), not internal correspondence (<1% shift in match_12/21, entropy) | Internal/intermittent support gap (37.3 internal-unmatched mass, highest of the four) interacting with landmark-level display-statistic instability; a real but modest barycenter-compromise effect (5.7-6.3% shift on leave-one-out) |
| **RU1303** | Cost matrix shifts only 4.8%; 3 abrupt jumps (fewest); tail-residual 0.26-0.28 | 100/100 matchable landmarks, but **zero SCLC-N cells in the raw data at all** -- a structurally different kind of gap from RU831's | 0 multimodal rows; MAP-vs-mean disagreement smallest of the four (~0.003-0.0035); highest competing-minima-row fraction of the three original patients (8%) at the raw cost-surface level, though still low in absolute terms | Row entropy ~2.63-2.64 nats; cell-level displacement low and stable (0.009-0.0134); barycenter-compromise effect: RU831's presence roughly doubles RU1303's landmark displacement (0.0064 -> 0.0131 with RU831 present) | Not separately probed with the terminal perturbation beyond the shared 8-patient refit; internal match structure stable | Missing a real biological state (SCLC-N) altogether, not an ambiguous-correspondence problem; some sensitivity to RU831's presence in the shared barycenter |
| **RU1444 (control)** | Cost matrix shifts 7.9%; 5-6 abrupt jumps; tail-residual 0.22-0.23 (lowest of the three) | 100/100 matchable landmarks; large, well-distributed SCLC-N population (56.5% of cohort, median 0.81) | 0 multimodal rows; MAP-vs-mean disagreement ~0.0035; competing-minima fraction 2-5% | Row entropy ~2.61-2.62 nats (lowest of the four); cell-level displacement lowest and most stable (0.0036-0.0042) | Not RU831-comparable in scale; internal match structure stable under the perturbation | Ordinary smooth local warp; no evidence of a distinct pathology |

RU581 (second comparison patient) tracks RU1444 closely on every Part 2/3
metric gathered (0 multimodal rows, MAP-vs-mean disagreement ~0.0038-0.012,
row entropy ~2.61-2.62 nats, interior correspondence mass/row-max-probability
within range of RU1303/RU1444) and was not run through Part 1's raw
cost-geometry figures (which remain the original 3-patient set committed in
`b3e9526`); its cell-level displacement (mean 0.0239, CV 0.0124) places it
between RU1303 and RU151/RU151 in the control-selection table above. It
behaves like a second, independent control rather than resembling RU831 on
any axis checked here.

## Part 5: evidence for and against each explanation

**Multimodal correspondence ambiguity (two or more well-separated
low-cost path families at the same parameters).** AGAINST, and now stronger
than in the original draft: zero multimodal rows detected under the stated
threshold in any regime for any of the four patients; low competing-minima
fraction (0-8%) at the raw cost-surface level, not elevated for RU831
specifically; small MAP-vs-mean path disagreement (<0.014, comparable
across all four patients); a perturbed-initialization probe converges back
to the same free energy and displacement to ~1e-8/1e-9 precision. The one
piece of data that originally looked like evidence FOR this (the
moderate-vs-severe landmark-level displacement gap) has been traced to a
numerical artifact and no longer counts as evidence either way.

**Missing terminal support.** AGAINST as a *complete* explanation, though
partially true: RU831 does have a small, late-pinned SCLC-N population
(n=140, median s_local 0.9992) rather than none at all, and its terminal-
unmatched mass (0.245) is the *smallest* of the four patients at the
selected `o0.1_teq` fit -- not the largest, as a pure "missing terminus"
story would predict. The 10x terminal-gap perturbation changed only the
tail and left internal correspondence essentially untouched for every
patient. The real density gap is internal (0.71-0.89 unmatchable block, sat
between a SCLC-A population ending ~0.61 and a small terminal SCLC-N
population), not terminal.

**Internal/intermittent support gap.** FOR, and the best-supported single
explanation here: RU831 has the highest `patient_unmatched_internal_mass`
of the four patients (37.3 vs 16-20), 80/100 matchable landmarks with a
19-landmark contiguous unmatchable block corroborated directly from raw
cell-type/s_local populations (not model-manufactured), and this pattern is
present and essentially unchanged across every regime and perturbation
tested.

**Broad/diffuse low-cost geometry and weak identifiability.** Partially FOR:
RU831's row-min-cost distribution is somewhat higher and its tail-residual
higher than the two original comparison patients (Part 1), consistent with
a flatter, less sharply-identified cost surface in its problematic region --
but this coexists with, and may simply be a restatement of, the internal
support gap rather than a separate phenomenon.

**Barycenter compromise (the shared consensus is unusually consequential
for RU831 specifically).** FOR, modestly: RU831's cost matrix shifts 64%
between the pre-joint and joint-candidate fits (vs 4.8%/7.9% for
RU1303/RU1444), and removing RU831 from the fit shifts the shared
barycenter 5.7% overall / 6.3% near RU831's own unmatchable region (a mild,
not sharp, concentration there) and roughly halves RU1303's own landmark
displacement. This is real but not dramatic evidence -- the effect is more
"RU831 measurably influences the shared consensus, and this affects RU1303
more than RU1444/RU581" than "the consensus is dominated by or badly
compromised because of RU831."

**Endpoint/open-mode artifact.** AGAINST: the 10x terminal-gap perturbation
changed only the tail (terminal-unmatched mass, endpoint pileup) and left
internal correspondence structure (<1% change in match_12/21, path entropy,
internal-unmatched mass) essentially untouched, for RU831 and for every
comparison patient. If RU831's elevated displacement were primarily an
artifact of how open/terminal modes are priced, this perturbation should
have moved internal correspondence more than it did.

**Optimization/local-minimum behavior.** Only partially testable with what
is available here, and this should be stated explicitly rather than
overclaimed: `initialize()` in `_barycenter.py` is deterministic given
landmarks and coordinates, so there is no built-in multi-start mechanism.
The one perturbed-initialization probe run here (a modest, single,
deterministic sinusoidal offset) converged back to the same solution to
very high precision -- a negative result, but from a single perturbation
type/magnitude, not an exhaustive search. **This axis remains genuinely
under-tested; do not treat the single negative result as ruling out
optimization-related explanations in general.**

**Ordinary smooth local warp.** Probably true and coexisting with the
support-gap explanation, though not cleanly separable from it with what was
measured here: RU831's cell-level displacement (~0.15) is elevated but of
the same order with or without any warp capability at all (0.150 with
`warp="on"`/`o0.1_teq` vs 0.155 with `warp="off"`/`trajectory_250_unit`),
which suggests most of RU831's elevated displacement is not specifically a
consequence of the joint-warp machinery but a more basic feature of how
poorly RU831's cells align to the shared barycenter in its problem region,
regardless of alignment method.

## Part 6: what would be needed before a branched/mixture consensus could be justified

This section is descriptive only; no architecture recommendation is made or
authorized here.

Given what has been found -- no multimodal rows, low competing-minima
fraction not elevated for RU831, a landmark-level displacement statistic
that appeared bistable but was traced to a numerical artifact, a real but
modest (not dramatic) barycenter-compromise effect, and a well-corroborated
internal/intermittent support-gap explanation that accounts for most of the
observed pattern without invoking multiple attractors -- the current
evidence does not support a branched or mixture-consensus architecture for
RU831. Before that kind of architecture change would be justified, at
minimum the following would need to be produced (none of which has been
produced in this task):

1. **Complete constrained path-family energies**, not just single-row
   argmin/competing-minima checks -- i.e. an actual enumeration or
   approximation of the total posterior mass assigned to two or more
   genuinely distinct *complete* paths through disjoint subsets of
   landmarks (not just locally-competing minima within one row), at the
   same fixed parameters. This was flagged as high-value by the coordinator
   review and was not completed in this task; the row-level multimodality
   check here is a necessary but not sufficient substitute for it.
2. **A same-graph `warp="off"` control with matched, comparably-calibrated
   terminal pricing**, to establish what RU831's cost geometry looks like
   with zero warp capability at all, as a cleaner baseline than the
   existing `trajectory_250_unit` (unit-precision, uncalibrated) run used
   opportunistically here.
3. **A genuine multi-initialization sweep** (several different
   perturbation types/magnitudes to the initial barycenter, not the single
   deterministic offset tested here), to more conclusively rule in or out
   multiple EM optima.
4. **Direct evidence that two distinct complete paths each explain a
   comparable, non-trivial share of held-out or independently-measured
   signal** (e.g. two different subsets of cells whose downstream biology
   -- not just alignment score -- is better explained by two different
   correspondence hypotheses) -- something no diagnostic run here was
   designed to detect, since all of them operate on the fitted posterior
   or cost surface itself, not on an independent readout.

Absent (1) and (4) in particular, the honest current reading is: RU831's
apparent pathology is most plausibly explained by an internal/intermittent
support gap (corroborated directly from the raw h5ad's cell-type
populations) interacting with a landmark-level display-statistic artifact,
plus a real-but-modest barycenter-compromise effect -- not by genuine
multimodal correspondence ambiguity. This agrees with the coordinator
review's framing; where this document's own diagnostics add detail beyond
that framing is mainly in narrowing the barycenter-compromise effect to
"real but mild, and more consequential for RU1303 than for RU831's own
problem region" and in confirming the internal/terminal mass split directly
against the raw cell populations.
