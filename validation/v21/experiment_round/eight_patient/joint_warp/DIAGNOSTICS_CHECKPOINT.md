# Joint-warp diagnostics checkpoint

- Branch: `fix/joint-warp-diagnostics`
- Frozen Phase 3–4 base: `ecf3134c561c61e3de006665b56f0c4603c3576d`
- Verified fixes commit: `14c2b530d553a7136b3363413e41b10739af8574`
- Worktree: `/tmp/scdebussy-joint-diagnostics-fix`

## Changes

`src/scdebussy/align/_project.py` now divides joint residual expression by
`sum_i A[i,j]`, with residual `NaN` below `1e-12` correspondence mass; it
continues to use `O_j` for matched support. `src/scdebussy/align/_api.py`
renames `endpoint_pileup` to `endpoint_occupancy`, the sum of first and last
matched consensus-state occupancy. It is not a projected-cell concentration.
Old-mode residual behavior remains unchanged.

`aggregate_joint_warp.py` now reports
`warp_edge_fraction_among_match_edges = (e12+e21)/(e11+e12+e21)` and
`warp_patient_state_fraction = (e12+2*e21)/(e11+e12+2*e21+internal+terminal)`.
The e11, warp, internal-unmatched, and terminal-unmatched patient-state
fractions form a partition. The table, heatmap, tradeoff plots, and figure
metrics were regenerated from frozen caches with the paired
`trajectory_250_unit` restricted baseline included. The six-candidate grid,
decision rule, objective, and fitting path were unchanged.

Changed source and tests: `_project.py`, `_api.py`,
`tests/test_v2_joint_diagnostics.py`, `tests/test_v2_joint_fit.py`,
`aggregate_joint_warp.py`, `regenerate_reporting.py`, and
`summarize_and_plot.py` (CSV line endings only). Updated artifacts are
`JOINT_WARP_RESULTS.md`, `run_table.csv/.json`, `final/*`, `figures/tradeoffs.*`,
and `verification/joint_warp_o0.1_teq_corrected.json`.

## Verification

- Focused tests: **13 passed** (`test_v2_joint_diagnostics.py` and
  `test_v2_joint_fit.py`). Full suite: **611 passed**, 6 existing warnings.
- Frozen deterministic synthetic joint fit, before versus after: SHA256 of
  barycenter, all posterior `log_a` arrays, and projected positions matched
  exactly; the complete history, including free energy, matched exactly.
- Regenerated HTAN `joint_warp_o0.1_teq` fit: barycenter and unprojected
  masses matched cached arrays exactly; positions matched elementwise with
  `equal_nan=True`. Coverage `0.875`, purity `0.7658225705792349`, mean
  displacement `0.028007415678087358`, iterations `7`, and convergence were
  unchanged. The regenerated JSON report is in `verification/`; its large
  `.npz` cache remains outside git in
  `/tmp/scdebussy-joint-diagnostics-fix-verification/`.
- The new 11-row table has 5 controls/baselines and 6 original joint fits;
  patient-state fractions sum to one for every joint fit (maximum error
  `2.22e-16`). The heatmap has raw, original v2, restricted estimated,
  restricted unit, selected joint, and moderate DTW panels in the original
  eight-patient order and fixed `[0,1]` scale.

| Corrected field, representative `joint_warp_o0.1_teq` | Before | After |
| --- | ---: | ---: |
| `warp_frac` (mixed edge/state units) | 0.3308216316 | removed |
| warp match-edge fraction | absent | 0.5192139149 |
| warp patient-state fraction | absent | 0.4708478403 |
| RU831 `endpoint_pileup` | 1.6144934369 | renamed `endpoint_occupancy`, same value |
| paired restricted-unit baseline in main table | absent | coverage 0.8625, purity 0.769181 |

In a deterministic `(2,1)` residual test, patient expression values 2 and 4
with `O_j=1` and barycenter 1 previously yielded residual 5; the corrected
weighted-mean residual is 2. In a `(1,2)` test with correspondence weights
0.5/0.5, expression 3, and barycenter values 1/2, the corrected residuals
are 2/1 (previously 0.5/-0.5). These residuals are not fields in the HTAN
fit JSON report; they are outputs of `align.summarize()`.

## Historical fields to avoid

Do not use the original Phase 3–4 `run_table` `warp_frac` as a fraction of
anything coherent; its edge numerator and patient-state denominator had
different units. The original six fit JSON files retain `endpoint_pileup`;
read those numbers only as matched endpoint occupancy. Any joint-mode
residual expression exported from the old `align.summarize()` is incorrectly
normalized when correspondence column sum differs from `O_j`. The original
ten-row main table and five-panel heatmap omitted the paired unit-precision
baseline. The original fit JSONs and caches remain untouched for provenance.

Reproduce reporting without refitting:

```bash
PYTHONPATH=src /data1/chanj3/wangm10/conda_envs/scdebussy-py311/bin/python validation/v21/experiment_round/eight_patient/joint_warp/regenerate_reporting.py --adata /scratch/chanj3/wangm10/HTAN/results/aligned_scdebussy_preprocessed_v2_ru942_deg_all.h5ad --cache-root /data1/chanj3/wangm10/scDeBussy-v2/.htan_experiment_cache/exp-htan-eight-patient
```

The commit hooks' Biome step cannot parse under the installed Node version;
Ruff flags pre-existing style in `summarize_and_plot.py`. Those formatter
hooks were skipped for the fixes commit. AST, private-key, whitespace,
line-ending, and conflict hooks passed; `git diff --check` was clean.
