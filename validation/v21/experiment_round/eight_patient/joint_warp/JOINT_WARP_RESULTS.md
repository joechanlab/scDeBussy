# Joint posterior-warp: Phase 3/4 results

## Mathematical contract

See `validation/v21/experiment_round/JOINT_WARP_CONTRACT.md` (Phase 1) for the
full derivation. Not duplicated here; the load-bearing facts used below are:
`A[i,j]` is a correspondence *weight*, not a match probability, once
`(1,2)`/`(2,1)` warp edges exist; occupancy (`M_i`, `O_j`) is a separate,
directly-aggregated quantity that remains a genuine probability on both axes
(`sum_j A[i,j] + patient_unmatched[i] = 1`, `O_j + consensus_unmatched[j] = 1`);
and the barycenter M-step consumes `A`/`log_a` unchanged regardless of source
posterior type.

## Implementation summary (Phase 3)

Commit `76ac08d` on `exp/joint-posterior-warp`. `fit()` gained
`warp="off"|"on"`, `omega_warp`, `lambda_terminal_gap` (mirroring
`LatticeModel`'s constructor names), `src/scdebussy/align/_api.py`. `warp="off"`
(default) is untouched and bit-identical to prior behavior, verified in
`tests/test_v2_joint_fit.py::test_warp_off_matches_default_exactly`.
`warp="on"` builds `LatticeModel(..., warp="on", ...)` every E-step and calls
`infer_joint`/`JointPosterior` instead of `infer`/`Posterior`;
`sufficient_statistics`/`expected_match_energy` (`_barycenter.py`) consume
`JointPosterior.log_a`/`.a` with **no code change**, confirming the Phase 1
contract's Q4 claim was a real drop-in swap, not an assumption.

New `AlignmentResult` content in joint mode (no new dataclass fields; carried
in the existing `diagnostics: dict` and `history: list[dict]`, so old-mode
shapes/fields are untouched):

- Per-iteration `history` rows gain `barycenter_change`, `e11_mass`,
  `warp_mass` (`e12+e21`), `internal_unmatched_mass`, `terminal_unmatched_mass`.
- Per-patient `diagnostics[patient]` gains `expected_match_11/12/21`,
  `patient_unmatched_internal/terminal_mass`, `consensus_unmatched_internal/
  terminal_mass`, `endpoint_pileup`, `consensus_share`,
  `degenerate_all_gap_risk`, `mean/max/p95_displacement`. Old-mode-only fields
  (`no_match_probability`, `low_match_probability`) are deliberately **not**
  extended (contract Q5: what "low support" should mean is left open by the
  plan).
- `_dp.py:check_joint_invariants` is a runtime safeguard (reusing the
  conservation identities from `tests/test_v2_joint_warp.py::assert_conservation`)
  called every E-step in joint mode; it raises `RuntimeError` on NaN/inf,
  occupancy-conservation violation, or count-identity violation. It ran on
  every iteration of all 6 real HTAN fits below without raising — i.e. the
  posterior invariants were sanity-checked on real data, not only synthetic
  cases.
- `_project.py`'s `project()`/`summarize()` now branch on `JointPosterior` via
  `occupancy_support()`/new `consensus_occupancy_support()`, using `M_i`/`O_j`
  instead of `sum_j A[i,j]`/`sum_i A[i,j]` per contract Q7/Q8 (numerically
  identical in this implementation, but occupancy is the semantic source of
  truth wired in).
- `_types.py:AlignmentResult.write()` exports `JointPosterior`'s actual fields
  instead of assuming old `Posterior` field names.

## Fit-level test results (Phase 3)

`tests/test_v2_joint_fit.py`, 7 new tests, all passing:

- (a) `test_warp_off_matches_default_exactly`: barycenter, `log_a`, `log_z`,
  history, and diagnostics keys are exactly identical between omitting `warp`
  and passing `warp="off"` explicitly.
- (a) `test_warp_on_rejects_global_mode_and_nonpositive_terminal_gap`:
  `fit()`-level validation mirrors `LatticeModel`'s own.
- (b) `test_joint_mode_uses_warp_edges_and_improves_fit`: on a synthetic
  monotone stretch/compression case (one of four patients follows a warped
  reparametrization of pseudotime), the joint fit's warp mass
  (`e12+e21 summed`) is strictly positive; a same-graph control with
  `omega_warp=1e6` uses zero warp mass; and enabling warp does not increase
  final free energy versus that same-structure no-warp control (a strict
  superset of paths cannot be worse at the achieved EM fixed point).
- (b) `test_joint_mode_diagnostics_and_history_fields`: all new diagnostic/
  history keys are present and finite; old-mode-only keys are absent.
- (c) `test_joint_mode_descent_and_invariants_hold`: the existing
  before/conditional/after free-energy descent check holds every iteration;
  `check_joint_invariants` does not raise on the final posteriors.
- (c) `test_joint_mode_project_and_summarize_use_occupancy`: `project()`/
  `summarize()` run without error on a `JointPosterior`-based
  `AlignmentResult` and return finite, correctly shaped output.
- (d) `test_joint_mode_is_deterministic`: two independent `fit()` calls with
  identical inputs produce bit-identical `barycenter`, `log_a`, and `history`.

Full suite: **605 passed** (598 pre-existing, unmodified, + 7 new) via
`PYTHONPATH=src <env>/bin/python -m pytest tests/ -q` from
`/tmp/scdebussy-htan-eight`.

## HTAN configuration (Phase 4)

- Input: `/scratch/chanj3/wangm10/HTAN/results/aligned_scdebussy_preprocessed_v2_ru942_deg_all.h5ad`.
  SHA256 verified to match CHECKPOINT.md's recorded
  `4f26e98a530bbda380121929d543b367884d0fb17beef58061de948ad09f7dc3` exactly
  before any fit ran.
- Base config: `configs/trajectory_250_unit.json` (frozen top-250 label-free
  trajectory genes, unit match precision, `n_landmarks=100`,
  `consensus_resolution=100`, `mode=open`, `kappa_i=kappa_d=3.0`,
  `calibration_quantile=0.9`, `calibration_band=0.25`), unmodified except
  adding `warp="on"`, `omega_warp`, `lambda_terminal_gap` to the `fit` block.
  `eta_i=eta_d≈0.5169` is calibrated identically across all 6 candidates
  (calibration does not depend on `warp`), so `lambda_i=lambda_d≈1.5508`
  (`kappa_i·eta_i`) is the natural "terminal cost equal to internal cost"
  reference point.
- Grid (3×2=6, within NEXT_JOINT_WARP_PLAN.md's suggested starting scale):
  `omega_warp ∈ {0.02, 0.05, 0.1}` × `lambda_terminal_gap ∈ {1.5508 ("teq",
  equal to internal), 0.5 ("tlow", lower)}`.
- Held-out marker panel: `HELD_OUT_MARKERS.json` (ASCL1, NEUROD1, INSM1,
  CHGA, CHGB, NCAM1, DLL3), used only for evaluation, never in fitting.
- Configs: `configs/joint_warp_o{omega}_t{eq,low}.json`. Run command (each):
  `PYTHONPATH=src <env>/bin/python validation/v21/experiment_round/run_candidate.py --adata <input> --config validation/v21/experiment_round/configs/joint_warp_o<omega>_t<eq|low>.json --output-dir validation/v21/experiment_round/eight_patient/joint_warp`
- All 6 fits converged in exactly 7 iterations, ~42s each (`~1.16s/iteration`
  old mode vs `~8.4s/iteration` joint mode on this cohort — the joint graph
  has roughly 3x the edges per node).

## Results table

Full machine-readable table: `run_table.csv` / `run_table.json` (10 rows: 4
comparison baselines + 6 joint candidates). Posterior-state diagnostics per
candidate (summed over all 8 patients) and per-patient are in each
candidate's `*.json` report's `diagnostics` block and in `run_table.csv`'s
`e11_frac`/`warp_frac`/`internal_unmatched_frac`/`terminal_unmatched_frac`
columns (fractions of total expected patient-state count, i.e.
`e11+e12+2*e21+internal_unmatched+terminal_unmatched`).

| method | coverage | purity | RU1303 overlap | held-out concordance | mean disp. | RU831 mean/max disp. | e11 / warp / int.-unmatched / term.-unmatched frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw `s_local` | 0.865 | 0.7715 | 0.531 | 0.4211 | 0 | n/a | n/a |
| original v2 | 0.858 | 0.7285 | 0.531 | 0.4170 | 0.013 | n/a | n/a |
| trajectory-250 restricted | 0.850 | 0.7850 | 0.132 | 0.4187 | 0.009 | n/a | n/a |
| moderate DTW prototype (`omega=0.06`, term=1.0) | 0.870 | 0.7629 | 0.100 | 0.4587 | 0.058 | n/a (compression/plateau, see SUMMARY.md) | n/a |
| **joint_warp_o0.02_teq** | 0.8725 | 0.7679 | 0.400 | 0.4259 | 0.0279 | 0.246 / 0.889 | 0.296 / 0.341 / 0.212 / 0.005 |
| joint_warp_o0.02_tlow | 0.8763 | 0.7658 | 0.411 | 0.4260 | 0.0285 | 0.247 / 0.889 | 0.294 / 0.339 / 0.211 / 0.012 |
| joint_warp_o0.05_teq | 0.8725 | 0.7651 | 0.402 | 0.4258 | 0.0279 | 0.115 / 0.213 | 0.300 / 0.338 / 0.214 / 0.005 |
| joint_warp_o0.05_tlow | 0.8775 | 0.7655 | 0.417 | 0.4260 | 0.0286 | 0.116 / 0.213 | 0.298 / 0.335 / 0.212 / 0.012 |
| **joint_warp_o0.1_teq (selected, see below)** | 0.8750 | 0.7658 | 0.402 | 0.4258 | 0.0280 | 0.115 / 0.213 | 0.306 / 0.331 / 0.217 / 0.006 |
| joint_warp_o0.1_tlow | 0.8775 | 0.7635 | 0.422 | 0.4260 | 0.0286 | 0.247 / 0.889 | 0.304 / 0.329 / 0.216 / 0.012 |

Iterations = 7, converged = True for all 6 joint candidates. All 6 satisfy the
finite/invariant safeguards (`check_joint_invariants` never raised).
`endpoint_pileup` (sum of first+last consensus-index occupancy) is ~1.6–1.8
across all patients in all 6 candidates — i.e. both trajectory endpoints are
individually near-saturated (~0.8–0.9 occupancy each), a genuine pile-up
signal worth carrying forward, though not obviously worse than old mode's
open-mode terminal handling (not directly comparable since old mode has no
occupancy-vs-A distinction to report this against).

## Interpretation against the decision rule

**No candidate fully satisfies the decision rule.** All 6 clear the coverage
bar (0.872–0.878 ≥ 0.865) and the held-out concordance bar (0.4258–0.4260 ≥
0.421, a small but consistent improvement over raw), and all move RU1303's
SCLC-A/other-SCLC-N overlap clearly below raw's 0.531 (down to 0.40–0.42,
though far short of the moderate DTW prototype's 0.100). **None reach
purity ≥ 0.772**: the best is `joint_warp_o0.02_teq` at 0.7679, a ~0.004
shortfall, with the rest at 0.7635–0.7658.

Among the 6, `omega_warp=0.02` (both terminal settings) and
`omega_warp=0.1_tlow` show a severe **RU831 pathology**: mean displacement
0.246, max 0.889 for that one patient alone (vs. 0.004–0.038 for every other
patient, see each candidate's `*.json` `diagnostics` block) — an order of
magnitude larger than any other patient's movement in the same fit.
`omega_warp∈{0.05,0.1}_teq` and `0.05_tlow` avoid this: RU831 mean/max
displacement drops to ~0.115/0.213, close to the healthier end of the range.
This is not a smooth function of `omega_warp`: `0.1_teq` (moderate) and
`0.1_tlow` (severe) sit right next to each other in the grid but land in
different qualitative regimes, which itself is a sign of a **bistable, not
smoothly graded, RU831 fit** — the EM optimum for this one patient jumps
between two attractors depending on a small change in the terminal-cost
penalty rather than moving continuously. `joint_warp_o0.1_teq` is selected as
the representative "best" candidate for the final heatmap/figures because it
has the highest purity (0.7658, tied with `o0.05_teq`) among the candidates
that avoid the RU831 pathology, but it is reported here explicitly as **not
passing the decision rule** (purity shortfall), not as a recommended
replacement for v2.

## RU831 diagnosis

Per-patient warp-function plots (`figures/warp_functions.png`) show RU831
tracks close to the raw identity line for `s_local` up to ~0.25–0.3, then
either (moderate regime, `o0.1_teq`, gray) makes a bounded upward jump
plateauing near aligned position ~0.6–0.85, or (severe regime, moderate DTW
prototype shown for comparison, cyan) makes a much larger, still-bounded
jump. In both regimes RU831's occupied range never reaches 1.0 — consistent
with the heatmap (`final/htan_comparison.png`), where RU831 has essentially
no cells past pseudotime ~0.6 in *every* method including raw `s_local`, i.e.
this patient genuinely lacks SCLC-N-stage coverage; the joint model is not
manufacturing that gap. What the joint model *is* doing differently from raw
is concentrating a large, discontinuous jump in the SCLC-A-adjacent region
specifically for RU831, more so than any other patient, and doing so in a way
that is sensitive to a 2x change in `omega_warp`/`lambda_terminal_gap`
(bistable, not graded). This matches the plan's own predicted failure mode
("cheap terminal edges absorb whole patients") localized to one patient
rather than the whole cohort, and matches the prior fixed-consensus DTW
prototype's independent finding of RU831-specific pathology
(`eight_patient/SUMMARY.md`: "pronounced compression and a plateau for
RU831"), now recurring in a genuine posterior EM model rather than only the
earlier post-fit heuristic.

## A/B/C limitation diagnosis

Aggregate displacement is essentially flat across a 5x range of `omega_warp`
(0.0279–0.0286 mean absolute displacement for all 6 candidates) while the
posterior's own warp-edge mass is substantial and stable (`warp_frac`
0.329–0.341 of total expected patient states — i.e. roughly a third of all
patient-state transitions are genuinely warp-kind edges, not negligible).
Combined with the fact that 7 of 8 patients move very little (0.004–0.04)
while RU831 alone accounts for essentially all of the "large" displacement
observed, and that this displacement is bistable rather than graded:

- **(A) warp/posterior formulation**: not obviously broken. The invariants
  hold on every iteration of every real fit (not just synthetic tests), the
  descent check holds, warp edges are used at non-trivial mass, and the
  synthetic fit-level test (`test_joint_mode_uses_warp_edges_and_improves_fit`)
  shows the mechanism does improve fit when the underlying warp signal is
  present and isolated from confounds. This rules out a basic wiring bug.
- **(B) barycenter representation**: partially implicated. The barycenter is
  still a single length-100 linear sequence; a warp edge can move a patient
  landmark's *conditional* position to the midpoint of two adjacent consensus
  cells (contract Q6/Q7), but the *consensus itself* is still a simple
  per-cell weighted mean (`_barycenter.py`'s closed form, unchanged). If
  RU831's true expression trajectory is not well-approximated by any single
  point on the shared linear barycenter in its problematic region, the
  M-step's weighted-mean update has nowhere better to move the barycenter
  without compromising the other 7 patients, which would produce exactly the
  observed pattern: warp edges get used (mass is real) but achieve limited
  *aggregate* movement, concentrated pathologically in the one patient whose
  data least fits the shared curve.
- **(C) single shared linear trajectory assumption**: the most likely primary
  driver. RU831's bistable jump under a small parameter change, its early
  plateau matching a real coverage gap (not model-induced), and the prior,
  independent DTW-prototype finding of the same patient-specific pathology
  under a completely different fitting procedure, together suggest RU831's
  SCLC-A-to-SCLC-N-adjacent region maps ambiguously onto the single shared
  consensus — i.e. there may not be a well-defined single warp for this
  patient at all, and the model is picking one of (at least) two
  comparably-costed alternatives depending on a small terminal-cost
  perturbation. This is consistent with the plan's own predicted failure mode
  and is exactly the kind of finding a branched/mixture consensus (explicitly
  out of scope here) would be needed to resolve.

**Overall**: the joint posterior-warp model works largely as designed
(posterior invariants hold on real data, warp mass is genuine and
substantial, EM descent holds, and the mechanism demonstrably helps on a
controlled synthetic case) but on this specific eight-patient cohort it does
not achieve movement beyond a small, RU831-concentrated, bistable effect
without a purity cost comparable to (and structurally different from) the
older heuristic DTW prototype's. The honest reading is closest to "the model
works as intended but the data's single-shared-linear-trajectory assumption
is being pushed against a genuine (at least) two-patient-scale
incompatibility (RU831 chiefly, RU1303 secondarily) that this model's
consensus representation cannot resolve without either accepting a
purity-costly jump or leaving that patient mostly unwarped."

## Failure modes observed

- RU831 bistable jump (above): flagged, not silently smoothed.
- Endpoint pile-up (`endpoint_pileup` ~1.6–1.8 across all patients/candidates):
  both trajectory ends are individually near-saturated; visible directly in
  `figures/warp_functions.png` as near-vertical segments at the axis edges
  for several patients.
- No degenerate all-gap collapse: `degenerate_all_gap_risk` is `False` for
  every patient in every candidate (checked against the
  `patient_unmatched_terminal_mass > 0.9·n` threshold in `_api.py`).
- No single-patient consensus domination: `consensus_share` (each patient's
  mean share of consensus occupancy) stays close to `1/8=0.125` for all
  patients in all candidates (range ~0.11–0.13), so the
  `joint_safeguards()` dominance warning (threshold 0.6) never fired.
- No nonfinite posterior quantities or conservation-identity violations in
  any of the 6 real fits (`check_joint_invariants` never raised).

## Recommended next step

Do not widen this grid further on this cohort (per the task's own
instruction, and because displacement is already insensitive to a 5x
`omega_warp` range). Before any further HTAN experimentation: (1) inspect
RU831's raw cost surface directly (reusing `cost_diagnostics.py`'s pattern,
already used for RU1303 in the prior round) to check for genuine bimodality
in its match costs, which would be direct evidence for the (C) diagnosis
above; (2) if confirmed, treat a mixture/branched consensus (or a
per-patient-pair compatibility diagnostic run before fitting a shared
consensus at all) as the next model change, rather than further tuning
`omega_warp`/`lambda_terminal_gap` on the current single-linear-consensus
architecture.

## Artifacts

- This document: `validation/v21/experiment_round/eight_patient/joint_warp/JOINT_WARP_RESULTS.md`
- Run table: `run_table.csv`, `run_table.json`
- Per-candidate fit reports (posterior-state diagnostics, resolved config,
  provenance): `joint_warp_o{omega}_t{eq,low}.json`
- Per-patient warp-function plots: `figures/warp_functions.png`/`.pdf`
- Tradeoff plots (purity vs displacement, coverage vs displacement, purity vs
  coverage): `figures/tradeoffs.png`/`.pdf`
- Held-out marker comparison: `heldout/heldout_summary.json`,
  `heldout/heldout_metrics.csv`, `heldout/heldout_trajectories.png`
- Final comparison heatmap (raw, original v2, trajectory-250 restricted,
  `joint_warp_o0.1_teq`, moderate DTW prototype for context):
  `final/htan_comparison.png`/`.pdf`, `final/metrics.csv`,
  `final/configurations.json`
- Fit caches (`.npz`, not committed to git; mirrors the checkpoint commit's
  precedent of copy-not-commit): copied to
  `/data1/chanj3/wangm10/scDeBussy-v2/.htan_experiment_cache/exp-htan-eight-patient/joint_warp/`,
  manifest in `CACHE_MANIFEST.tsv` (also committed alongside this document).
