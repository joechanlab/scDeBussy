# HTAN experiment checkpoint

## Git and storage

- Worktree at audit: `/tmp/scdebussy-htan-eight`.
- Branch: `exp/htan-eight-patient`.
- HEAD before this checkpoint commit: `7176f8d099bea000df240340f05dc02ddfe5a4bf`. For the current HEAD after this document is committed, run `git -C /tmp/scdebussy-htan-eight rev-parse HEAD`.
- Base branch: `refactor/v2`; base SHA: `a647d697c3d39d93fbce38158c2bd071448c988c`.
- Git remote: `origin`, `git@github.com:joechanlab/scDeBussy.git`. This checkpoint does not imply a remote push.
- At audit, tracked files had no modifications. The untracked historical `validation/v21/FINDINGS.md`, `experiment_round/results/*.json`, and `experiment_round/summary/` are included in this checkpoint commit; their three `.npz` caches are copied to persistent storage. The separate `refactor/v2` workspace has existing unrelated dirty files and was not reset or committed here.
- Large cache copy: `/data1/chanj3/wangm10/scDeBussy-v2/.htan_experiment_cache/exp-htan-eight-patient/`. Its layout mirrors `validation/v21/experiment_round/`. All 55 `.npz` files (112,107,481 bytes) were copied and verified with an `rsync -acn` dry run that reported no differences. See `CACHE_MANIFEST.tsv` for per-file sizes and SHA256.
- No `.h5ad` was generated in this experiment worktree. The input remains at the external path below. Neither the worktree nor caches should be deleted until the branch and persistent cache copy are verified at the next session.

## Input and exact run settings

- Eight-patient input: `/scratch/chanj3/wangm10/HTAN/results/aligned_scdebussy_preprocessed_v2_ru942_deg_all.h5ad`.
- Input SHA256: `4f26e98a530bbda380121929d543b367884d0fb17beef58061de948ad09f7dc3`.
- 34,380 cells, 2,758 genes; patients RU1083, RU1303, RU1444, RU151, RU263, RU581, RU831, RU942. RU1646 is excluded.
- Python: `/data1/chanj3/wangm10/conda_envs/scdebussy-py311/bin/python`; execute from the experiment worktree root with `PYTHONPATH=src`. No random seed or stochastic method was used.
- Baseline commands and frozen configs:

```bash
PYTHONPATH=src /data1/chanj3/wangm10/conda_envs/scdebussy-py311/bin/python validation/v21/experiment_round/run_candidate.py --adata /scratch/chanj3/wangm10/HTAN/results/aligned_scdebussy_preprocessed_v2_ru942_deg_all.h5ad --config validation/v21/experiment_round/configs/original_v2.json --output-dir validation/v21/experiment_round/eight_patient/results
PYTHONPATH=src /data1/chanj3/wangm10/conda_envs/scdebussy-py311/bin/python validation/v21/experiment_round/run_candidate.py --adata /scratch/chanj3/wangm10/HTAN/results/aligned_scdebussy_preprocessed_v2_ru942_deg_all.h5ad --config validation/v21/experiment_round/configs/centered_default.json --output-dir validation/v21/experiment_round/eight_patient/results
PYTHONPATH=src /data1/chanj3/wangm10/conda_envs/scdebussy-py311/bin/python validation/v21/experiment_round/select_trajectory_genes.py --adata /scratch/chanj3/wangm10/HTAN/results/aligned_scdebussy_preprocessed_v2_ru942_deg_all.h5ad --held-out validation/v21/experiment_round/HELD_OUT_MARKERS.json --output-dir validation/v21/experiment_round/eight_patient/feature_selection
PYTHONPATH=src /data1/chanj3/wangm10/conda_envs/scdebussy-py311/bin/python validation/v21/experiment_round/run_candidate.py --adata /scratch/chanj3/wangm10/HTAN/results/aligned_scdebussy_preprocessed_v2_ru942_deg_all.h5ad --config validation/v21/experiment_round/configs/trajectory_250.json --output-dir validation/v21/experiment_round/eight_patient/results
PYTHONPATH=src /data1/chanj3/wangm10/conda_envs/scdebussy-py311/bin/python validation/v21/experiment_round/run_candidate.py --adata /scratch/chanj3/wangm10/HTAN/results/aligned_scdebussy_preprocessed_v2_ru942_deg_all.h5ad --config validation/v21/experiment_round/configs/trajectory_250_unit.json --output-dir validation/v21/experiment_round/eight_patient/results
PYTHONPATH=src /data1/chanj3/wangm10/conda_envs/scdebussy-py311/bin/python validation/v21/experiment_round/warp_prototype.py --adata /scratch/chanj3/wangm10/HTAN/results/aligned_scdebussy_preprocessed_v2_ru942_deg_all.h5ad --fit-cache validation/v21/experiment_round/eight_patient/results/trajectory_250_unit.npz --output-dir validation/v21/experiment_round/eight_patient/warp_sweep_unit --penalty-scale fixed
PYTHONPATH=src /data1/chanj3/wangm10/conda_envs/scdebussy-py311/bin/python validation/v21/experiment_round/summarize_and_plot.py --adata /scratch/chanj3/wangm10/HTAN/results/aligned_scdebussy_preprocessed_v2_ru942_deg_all.h5ad --candidate-dir validation/v21/experiment_round/eight_patient/results --candidate validation/v21/experiment_round/eight_patient/results/original_v2.npz --candidate validation/v21/experiment_round/eight_patient/results/centered_default.npz --candidate validation/v21/experiment_round/eight_patient/results/trajectory_250.npz --candidate validation/v21/experiment_round/eight_patient/warp_sweep_unit/dtw_omega_0.06_terminal_1.npz --candidate validation/v21/experiment_round/eight_patient/warp_sweep_unit/dtw_omega_0.4_terminal_0.1.npz --output-dir validation/v21/experiment_round/eight_patient/final
```

These are the exact entry points and config paths used; each fit report in `eight_patient/results/*.json` records its resolved parameters and source SHA256. The script refuses to overwrite existing fit outputs, so use a new output directory for reproduction. Other panel/config runs are in `configs/`, and the warp grid is in `eight_patient/warp_sweep_config.json`. The earliest fits were generated before the final source snapshot was committed; their source hash is `95ab4c33a0c1d550dab2acaadc9986e034e948dbb6e7bbabaccc048faa0a01ee`. Later fits use source hash `2e5ddd5a82b10cb1eafa5a8862e63f2261174b9a14c92093b1821e7a0b56ecea`. `PROVENANCE.json` explains this limitation; do not claim that every fit ran from the final commit.

## Outputs and verification

- Main interpretation: `eight_patient/SUMMARY.md`; input/source record: `eight_patient/PROVENANCE.json`.
- Main heatmap: `eight_patient/final/htan_comparison.png` and `.pdf`; all-run table: `eight_patient/final/all_runs.csv`; per-patient table: `eight_patient/final/metrics.csv`; displayed configs: `eight_patient/final/configurations.json`.
- Baseline figure and table: `eight_patient/baseline/`.
- Cost surfaces and RU1303 details: `eight_patient/diagnostics/unit_cost/`; warp-function and tradeoff figures: `eight_patient/diagnostics/warp/`; held-out marker trajectories and per-gene scores: `eight_patient/heldout/`.
- Frozen panel and gene ranking: `eight_patient/feature_selection/`; seven excluded markers: `HELD_OUT_MARKERS.json`.
- Reusable fit caches: `eight_patient/results/*.npz`; original-v2 detailed diagnostic: `eight_patient/diagnostics/original_v2_diagnostic.npz`; 42 prototype maps: `eight_patient/warp_sweep_unit/*.npz` and `eight_patient/warp_sweep_estimated_scaled/*.npz`. Three earlier nine-patient caches are under `results/*.npz` and are historical.
- Focused tests: `46 passed, 5 AnnData fixture warnings` across `test_v2_api.py`, `test_v2_coordinates.py`, `test_v2_discovery.py`, `test_v2_precision_modes.py`, `test_htan_feature_selection.py`, and `test_htan_warp_prototype.py`.
- The pre-commit structural checks passed. Biome and Ruff style hooks were skipped for the experiment commit because the installed Biome Node runtime failed and Ruff rewriting would change the recorded fit-source fingerprint. This is a style-gate limitation, not evidence of model correctness.

## Next step

Read `NEXT_JOINT_WARP_PLAN.md`. Implement a joint posterior warp model on a fresh experiment branch only after reviewing its explicit correspondence, occupancy, and terminal-gap contracts. Keep the trajectory-250 rule frozen; test synthetic separation before a small HTAN matrix. The current prototype is fixed-consensus Viterbi and must not replace v2.
