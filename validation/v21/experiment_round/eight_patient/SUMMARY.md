# Eight-patient HTAN alignment experiment

Input: `aligned_scdebussy_preprocessed_v2_ru942_deg_all.h5ad`, 34,380 cells,
2,758 genes, eight patients, with RU1646 removed and RU942's DEG-PCA
pseudotime. SHA256 and source details are in `PROVENANCE.json`. No old
nine-patient fit was reused as this cohort's baseline.

## Result

The raw `s_local` control scores **0.865 coverage and 0.772 purity**. Original
v2 scores 0.858/0.728. A label-free 250-gene panel under the original
restricted lattice reaches **0.850/0.785**, but its mean displacement is only
0.009. It is the strongest balanced measured candidate in this round, with a
small coverage loss and limited movement. A 500-gene panel reaches 0.789 purity
but moves only 0.005 on average and does not resolve RU1303. These exploratory
scores come from choosing among panels on this same cohort, so they are not
independent validation.

The more meaningful movement test is the fixed-consensus DTW prototype with
unit-weighted 250 genes. At `omega=0.06`, terminal cost 1.0, coverage is 0.870,
purity 0.763, mean displacement 0.058, and RU1303 SCLC-A overlap with other
patients' SCLC-N falls from 0.531 to 0.100. Held-out marker trajectory
concordance increases from 0.421 to 0.459. It does **not** beat raw purity or
the restricted 250-gene fit, and therefore fails the prespecified decision
rule. The apparent high-purity terminal setting (`omega=0.4`, terminal 0.1)
scores 0.860/0.785 but leaves RU1303 at identity and raises its overlap to
0.792; it is not a useful solution for that case.

| Method | Coverage | Purity | Rejected | Mean displacement | p95 displacement | RU1303 overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw `s_local` | 0.865 | 0.772 | 0% | 0 | 0 | 0.531 |
| original v2 | 0.858 | 0.728 | 0.017% | 0.013 | 0.192 | 0.531 |
| centered default | 0.849 | 0.710 | 0.017% | 0.016 | 0.192 | 0.612 |
| trajectory 250, restricted | 0.850 | 0.785 | 0.017% | 0.009 | 0.111 | 0.132 |
| trajectory 500, restricted | 0.846 | 0.789 | 0.017% | 0.005 | 0.051 | 0.522 |
| trajectory 250, unit precision | 0.863 | 0.769 | 0.017% | 0.033 | 0.155 | 0.536 |
| moderate DTW prototype | 0.870 | 0.763 | 0.017% | 0.058 | 0.212 | 0.100 |
| terminal DTW setting | 0.860 | 0.785 | 0.017% | 0.011 | 0.101 | 0.792 |

Coverage is the existing mean patient fraction across 100 bins; purity is
cross-patient 50-neighbor cell-type purity. Rejection is nonfinite position.
The full 52-run table, including per-run parameters and gap counts, is
`final/all_runs.csv`. Per-patient metrics, type medians and cross-patient spread
are in `final/metrics.csv`; the 250-gene fit moves RU1303 SCLC-A median from
0.750 to 0.719, while moderate DTW moves it to 0.709. Neither approaches the
0.40–0.51 match preferred by the revised late-landmark cost surface.

## Hypotheses and diagnostics

The original all-gene cost matrix favors the late consensus for RU1303's last
20 landmarks (median row minimum 0.869). The 250-gene estimated-precision
objective favors 0.505 and its unit-precision version favors 0.404. The latter
is a plausible off-diagonal signal for RU1303's observed SCLC-A tail, but the
first implementation does not turn it into a full, trustworthy alignment.
`diagnostics/unit_cost/cost_matrices.png` shows every patient's cost surface;
`ru1303_cost_matrix.png`, `ru1303_late_candidates.csv`, and
`ru1303_gene_contributions.csv` show the late alternatives. For landmark 90,
STMN1 and KRT18 favor the earlier match, while HLA-A favors the late match.

The label-free selector uses 25 pseudotime bins, requiring adequate cells,
expression and within-patient change in at least three patients. It rewards
correlated temporal profiles and penalizes between-patient offsets. The top
250 panel has median support in seven patients and median pairwise profile
correlation 0.571. All seven predeclared markers in `HELD_OUT_MARKERS.json` were
excluded from selection and new candidate fits. The ranked gene table and exact
selection settings are in `feature_selection/`.

Unit precision gives more RU1303 off-diagonal signal, but its restricted fit
has purity 0.769. Capping high contributors lowers purity to 0.726. Centering
the 250-gene panel gives 0.783 purity with coverage 0.833. Thus neither
centering nor generic contribution capping is supported as the next default.

The DTW prototype has `(1,1)`, `(1,2)`, and `(2,1)` match transitions, separate
internal gaps, and a separate terminal deletion cost. The 21 unit-precision
and 21 estimated-precision sweeps are in `warp_sweep_unit/` and
`warp_sweep_estimated_scaled/`. The estimated-precision sweep performs poorly
(maximum purity 0.529). The moderate unit-precision map uses 37 internal
patient gaps across 800 landmarks and no terminal consensus deletions; RU1303
itself uses six of each warp transition, no gaps, and maps its last landmark to
the end. The terminal setting uses 30 terminal consensus deletions overall,
but **none for RU1303**. `diagnostics/warp/` contains mapping, tradeoff, and
gap-count plots and the patient-level flags.
The mapping plots also show pronounced compression and a plateau for RU831,
which is another reason the moderate map should not be promoted from this
prototype.

The prototype is post-fit Viterbi alignment to a *fixed* consensus. It does not
refit the barycenter and is not a v2 posterior model. Its finite unsupported
flag is inherited from the unit-precision fit; the prototype itself has no
calibrated posterior rejection probability. Treat its coverage and purity as
comparative diagnostics only. It must not replace `scdebussy.align` v2.

## Held-out biological check

Across ASCL1, NEUROD1, INSM1, CHGA, CHGB, NCAM1 and DLL3, mean cross-patient
smoothed-trajectory correlation is 0.421 on raw `s_local`, 0.419 on restricted
trajectory 250, 0.433 on restricted unit precision, 0.459 on moderate DTW, and
0.420 on the terminal setting. Mean peak spread is respectively 0.535, 0.513,
0.513, 0.467 and 0.523. Effects vary by gene: moderate DTW improves ASCL1 and
NCAM1 but worsens NEUROD1. The marker plots and per-gene metrics are in
`heldout/`. This is supportive of some temporal correction, but it does not
rescue the loss in purity.

## Direct answers and next experiment

- **Beat raw purity?** Yes, restricted 250 and 500 panels by 0.013 and 0.017;
  neither supplies substantial movement. The moderate movement candidate does
  not beat raw purity.
- **Beat raw coverage?** Moderate DTW reaches 0.870 versus 0.865, while its
  purity falls. Restricted panels lose 0.015–0.019 coverage.
- **Improve purity without major coverage loss?** The 250 and 500 panels do,
  with small motion and exploratory selection on this cohort.
- **Move RU1303 SCLC-A away from SCLC-N?** The 250 panel and moderate DTW reduce
  the overlap, but RU1303's median moves only 0.030 and 0.040, respectively.
- **Supported by cost surface?** Yes in direction: late RU1303 row minima favor
  mid-consensus. The actual fitted movements are much smaller than those minima.
- **Held-out genes more coherent?** Moderate DTW improves the aggregate and
  several markers, but not every gene; the restricted 250 panel does not improve
  aggregate concordance.

The next experiment should fit a **joint posterior model** with separately
priced warp and terminal states, updating the consensus under those matches.
Require posterior match mass and projection support to have a consistent
interpretation before testing it on HTAN. Freeze this panel-selection rule and
use an independent cohort or patient-level holdout for candidate selection.

## Reproduction and artifacts

All fits use deterministic configuration and no random seed. Fit settings are
in `../configs/*.json`; feature selection settings are in
`feature_selection/selection_config.json`, and the warp grid is in
`warp_sweep_config.json`. Each fitted cache and JSON report in `results/`
stores the resolved config, source SHA256, input cell IDs, and diagnostics.
`PROVENANCE.json` records the input hash and branch/base commit. Existing
nine-patient reports were not overwritten.

Focused verification of the alignment options, feature selector and warp
prototype: `46 passed` using the six relevant pytest modules. The five
warnings are AnnData's implicit conversion of test fixture indices to strings.
Fit `.npz` caches remain in this worktree and are excluded from the branch
commit because they total about 108 MB; the source, configurations, reports,
tables and figures are committed.

Main figure: `final/htan_comparison.png` and `.pdf`. Gray marks bins with
fewer than three finite cells; the adjacent status strip separately records
nonfinite and finite unsupported cells. Every panel uses the same patient order,
cell-type colors, 100 bins and unscaled `[0,1]` axis.
