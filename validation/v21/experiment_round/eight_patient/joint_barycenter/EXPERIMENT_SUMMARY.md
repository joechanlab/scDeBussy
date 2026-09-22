# HTAN 8-patient alignment: experiment journey (debugging log, in order)

Top-level summary of the full debugging arc from the original v2 result to the current best
8-patient fit. Each earlier claim links to the report that established it; later reports in
`joint_barycenter/` supersede earlier numbers where they overlap. This is a chronological trace,
not a fresh analysis — see the linked documents for methods and full metrics.

## Legend

- ✅ **Validated model improvement** — a mechanic change that measurably helped and is kept.
- 🔧 **Correctness fix** — code was wrong relative to its own stated contract; fixing it is not
  optional and does not need re-justifying.
- ⚠️ **Provisional heuristic** — works on this cohort's point estimate, not yet shown to
  generalize; treat as a placeholder, not a conclusion.
- ❓ **Unresolved** — known issue, not addressed.

## The story, in order

**1. Original v2 mostly behaved like identity** ✅-problem-identified. The all-gene v2 fit scored
purity **.728**, *below* the raw `s_local` baseline's .772, with near-zero displacement — its
affine gap-only warp/gap mechanics (no dedicated speed-change state) were too restrictive to move
cells meaningfully away from their unaligned pseudotime.
(`validation/v21/experiment_round/eight_patient/SUMMARY.md`)

**2. Trajectory-250 feature selection improved the matching objective** ✅. A label-free
250-gene panel (no cell-type labels, no held-out markers) raised purity to **.785** at coverage
**.850**, and gave RU1303 real off-diagonal expression signal (best-match cost favors position
.404, not the near-identity .869 the all-gene cost matrix favored).
(`SUMMARY.md`, `feature_selection/`)

**3. Pairwise RU1303→RU581 showed the old open DTW could recover the biologically plausible
direction** ✅-signal-found. The v2 lattice's own posterior drifted RU1303's late region toward
RU581's *terminal* SCLC-N block (wrong direction); the older open-begin/open-end DTW, given the
same cost matrix, could reach RU581's *mid* SCLC-A region instead — proof the expression cost
supported the right answer even when the current graph mechanics didn't find it.
(`ru1303_ru581_pairwise/RU1303_RU581_PAIRWISE.md`)

**4. Separating speed-change from biological-gap costs** ✅, but insufficient alone. Adding a
dedicated soft run-length penalty for consecutive speed-change moves (distinct from the affine
biological-gap price) reduced plateau/compression pathology, but no `ρ` value alone reached the
target late-region interval — regularizing speed changes helps but doesn't fix directionality by
itself. (`ru1303_ru581_pairwise/SLOPE_REGULARIZATION.md`, `PATH_COST_DECOMPOSITION.md`)

**5. Cheap terminal pricing helped RU1303 but did not generalize** ⚠️→rejected. A uniformly cheap
reference-terminal price recovered RU1303's useful truncation (41% tail skip, held-out MSE
1.429 vs 1.429 fixed), but applied *uniformly* it also truncated full-trajectory controls that
shouldn't be truncated (RU581→RU1303 reverse direction got *worse*, MSE 1.802 vs 1.737) and did
nothing for RU831→RU581's 26-landmark internal gap (already visible at this stage, before its
root cause was understood). (`ru1303_ru581_pairwise/TERMINAL_PRICING.md`,
`TERMINAL_GENERALIZATION.md`)

**6. Coverage gating recovered the useful directionality** ✅. Instead of a uniform cheap price,
gate the cheap price *only* on patients whose terminal-centroid coverage score is well below the
cohort's — a label-free, pre-fit rule (no cell-type label, no held-out marker). On five
prespecified pairs this preserved every useful RU1303 truncation and avoided the harmful reverse
truncation. (`ru1303_ru581_pairwise/COVERAGE_GATE.md`)

**7. Joint barycenter initially became jagged** ❓→diagnosed→✅-fixed. Extending from pairwise to
a 3-patient joint posterior barycenter (RU1303/RU581/RU1083), the unconstrained
correspondence-weighted M-step produced a barycenter whose RMS second-difference roughness rose
~70x, because adjacent consensus landmarks alternated which single patient dominated their match
mass (bin 44: 99% RU1303; bin 45: flips to RU581/RU1083; bin 46: back to RU1303) — not a
low-support-bin artifact (support never dropped below 1.89). (`ru1303_ru581_pairwise/
THREE_PATIENT_JOINT.md`, `BARYCENTER_SMOOTHING.md`)

**8. Curvature regularization (γ=1) stabilized the M-step** ✅. Replacing the plain
correspondence-weighted mean with `min_B Σw_j‖B_j−B̂_j‖² + γΣ‖B_{j+1}−2B_j+B_{j-1}‖²` at γ=1 cut
barycenter roughness 70x→3.9x (3-patient) and 22x (4-patient with RU1444 added), turned EM from
oscillating to monotone convergence, preserved RU1303's truncation and the full patients' full
length, and *improved* held-out marker MSE. Validated at 3-, 4-, and 8-patient scale.
(`BARYCENTER_SMOOTHING.md`, `FOUR_PATIENT_JOINT.md`, `EIGHT_PATIENT_JOINT.md`)

At full 8-patient scale, two new problems appeared that the smaller pilots could not have
revealed: the coverage gate fired on 5/8 patients instead of just RU1303 (diagnosed in
`GATE_ABLATION.md`, fixed by reverting to RU1303-only gating — see step 10), and:

**9. RU831 pathology was traced to ignored `landmark_matchable=False` values and fixed as a
pipeline correctness issue** 🔧. RU831 has 20/100 landmarks flagged unmatchable by the frozen
preprocessing (real cell-density gap, not a modeling choice), but `match_cost` fed every script
in this round used their raw, unreliable expression anyway — the DP graph could only avoid them
if their imputed values *happened* to look expensive, which worked for the RU581 reference by
luck but not for RU1083 (~16 landmarks were silently matched at deceptively low cost).
`compile_graph(cost, tail_price, matchable=None)` now bakes a permanent penalty onto
ordinary match/warp edges sourced from unmatchable landmarks; `matchable=None` (every prior
caller) is a verified no-op. This is a bug fix, not a modeling choice — it makes the graph behave
according to its own stated data-support contract. (`ru831_diagnostic/RU831_DIAGNOSIS.md`,
`ru831_diagnostic/MATCHABILITY_FIX.md`, commit `95567df`)

**10. Post-fix RU1303-only gating produced the best full-cohort result** ✅. With the
matchability fix wired into the full 8-patient fit, manually restricting the coverage gate to
RU1303 alone (matching what the smaller pilots had validated, before the automatic rule
over-triggered) beat trajectory-250 restricted on *both* coverage and purity simultaneously — the
first candidate in this round to do so — while keeping RU1303/SCLC-N overlap near zero.
(`GATE_ABLATION_POSTFIX.md`, commit `9f7d2bc`)

**11. The label-free largest-gap gate reproduced RU1303-only gating on the point estimate, but
stability testing showed it is fragile** ⚠️→flagged-provisional. Sorting the existing,
already-label-free coverage score and splitting at the single largest gap between consecutive
values exactly reproduced the manual RU1303-only assignment with no patient identity named
(`AUTO_GATE.md`, commit `813b6da`). But leave-one-out and bootstrap stability testing
(`stability/GATE_STABILITY.md`, commit `b28e09d`) found: dropping one specific patient (RU581)
flips the gate to a 5-patient set outright, and a nonparametric bootstrap of the score's own
peer-position data recovers the exact RU1303-only split only **46.9%** of the time. The rule is
not recommended for freezing (see `NEXT_STEPS.md`).

## Key final numbers

| Method | Coverage | Purity | Held-out marker concordance | RU1303/SCLC-N overlap |
|---|---:|---:|---:|---:|
| raw `s_local` | .865 | .772 | .421 | .531 |
| trajectory-250 restricted | .850 | .785 | .419 | .132 |
| **best: post-fix joint fit, RU1303-only gate** | **.855** | **.786** | **.521** | **.000** |

(Full comparison including the automatic-gate and pre-fix variants: `GATE_ABLATION_POSTFIX.md`,
`MATCHABILITY_FIX_EIGHT_PATIENT.md`.)

## Gate-stability summary (step 11 detail)

- RU1303 itself is almost always identified as the low-coverage patient: gated on in 99.5% of
  bootstrap replicates and 7/7 leave-one-out variants where it wasn't the one dropped.
- The **exact** RU1303-only split (i.e., nobody else gated on too) is unstable: only 46.9% under
  bootstrap resampling of the score's own peer-position data, and drops to 47.9% under additive
  noise at the scale of the real gaps between mid-coverage patients (σ≈.10).
- Dropping RU581 from the cohort flips the automatic gate to a 5-patient set outright (near-tie
  gap ratio 1.05, vs the full-cohort 1.29).
- **Recommendation: do not freeze the largest-gap threshold rule for external validation** — see
  `NEXT_STEPS.md` for the proposed replacement direction.

## Reproduction: cache paths not tracked in git

All reports/figures/configs/source changes referenced above are committed. The following large,
fully-reproducible caches are intentionally gitignored (`validation/v21/experiment_round/.gitignore`:
`*.npz`) and must exist on disk to rerun any script in this round without a full refit:

- `/data1/chanj3/wangm10/scDeBussy-v2/.htan_experiment_cache/exp-htan-eight-patient/eight_patient/results/`
  — the frozen candidate caches (`original_v2.npz`, `trajectory_250.npz`, `trajectory_250_unit.npz`,
  etc., ~3–14 MB each) that every comparison and pairwise script in this round loads via the
  `CACHE` constant in `ru1303_ru581_pairwise/run_pairwise.py`.
- `validation/v21/experiment_round/eight_patient/joint_barycenter/*/{*.npz}` and
  `ru1303_ru581_pairwise/{three,four}_patient_joint/pilot_state.npz` — this session's own EM
  state/candidate caches, all regeneratable deterministically (no RNG anywhere in this pipeline)
  by rerunning the corresponding `.py` script in the same directory.
- Raw input: `/scratch/chanj3/wangm10/HTAN/results/aligned_scdebussy_preprocessed_v2_ru942_deg_all.h5ad`
  (~1.2 GB; SHA256 in `PROVENANCE.json`).
- Conda env: `/data1/chanj3/wangm10/conda_envs/scdebussy-py311` (has a modern `scdebussy.align`
  install; the system Python's editable install resolves to a different, older path — always
  invoke scripts in this round with this env's `python` and `PYTHONPATH=<repo>/src`).

## Classification recap

| # | Item | Class |
|---|---|---|
| 2 | Trajectory-250 feature selection | ✅ validated improvement |
| 4 | Speed/gap cost separation | ✅ validated improvement (partial, needed more) |
| 6 | Coverage gating (mechanism) | ✅ validated improvement |
| 8 | γ=1 curvature regularization | ✅ validated improvement |
| 9 | `landmark_matchable` propagation | 🔧 correctness fix |
| 10 | RU1303-only gate (manually set) | ✅ validated, best current result |
| 11 | Largest-gap label-free gate rule | ⚠️ provisional, not for freezing |
| — | RU831/RU942's residual purity/coverage cost even post-fix | ❓ unresolved (real, not a bug) |
| — | Latent coverage model for the gate | ❓ unresolved, next direction |
