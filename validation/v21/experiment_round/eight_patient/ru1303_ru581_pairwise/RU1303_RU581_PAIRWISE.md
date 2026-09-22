# RU1303 → RU581 focused pairwise alignment diagnostic

## Scope and methods

This diagnostic uses the existing eight-patient HTAN input and **only** the RU1303/RU581 landmark pair. No cohort fit, gene selection, or model code was rerun. The frozen `trajectory_250_unit.npz` cache supplies the two 100-landmark expression curves. The pairwise cost is the mean squared expression difference across exactly the frozen trajectory-250 genes, with unit precision. All 100 landmarks of each patient are matchable. The seven held-out markers (ASCL1, NEUROD1, INSM1, CHGA, CHGB, NCAM1, DLL3) are excluded from that panel and were read only after the paths were computed.

The older hard DTW call is in `src/scdebussy/tl/_alignment.py` (`_run_alignment`'s `dtw(...)`, around lines 381–388; defaults around lines 1017–1024). Its asymmetric recurrence advances one RU1303 query landmark per step while RU581 reference position advances by 0, 1, or 2. Open begin/end lets a fully consumed RU1303 query use a subspan of RU581. The older default Sakoe–Chiba band is 20% of 100 bins. In contrast, current v2's `src/scdebussy/align/_model.py` lattice normally matches `(1,1)` and handles a patient-only or reference-only advance as an insertion/deletion with affine costs; its open mode permits terminal skips, and `src/scdebussy/align/_dp.py` sums over paths to obtain a posterior. There is no dedicated speed-change move in the current v2 `warp="off"` graph.

Here **pairwise v2** means applying that unchanged v2 lattice directly to the frozen RU1303 × RU581 cost matrix, with the original eight-patient unit-fit parameters (`beta=1`, `eta_i=eta_d=0.516918`, `lambda_i=lambda_d=1.550754`, open mode). The v2 path shown on the cost plot is its MAP path; the cell mapping and heatmap use its *posterior conditional mean* at landmarks with match mass at least 0.1. This direct pairwise test does not refit a two-patient barycenter and therefore isolates path mechanics from consensus fitting. Raw `s_local` is the identity path. Older DTW uses its original asymmetric/open recurrence on **the same unit-precision cost matrix** instead of its older default cosine distance, so the paths reflect recurrence differences rather than different genes or metrics. Band 20 is the faithful older setting; band 60 is one reachability check because the late cost minima near RU581 index 40 lie outside band 20.

## Path-level results

The native total cost is the method's own objective and is **not directly comparable** between v2 and DTW: v2 adds gap penalties, while hard DTW charges visited expression cells and has no biological-gap state. The matched-expression column is the sum of the common cost matrix along each path; its span differs across methods.

| Method | Native total / matched expression cost | Matched span RU1303 → RU581 (indices) | Unmatched begin/end RU1303; RU581 | Warp stretch / compression | Internal biological gaps RU1303 / RU581; terminal skips | RU1303 mean / max cell displacement |
| --- | ---: | --- | --- | ---: | --- | ---: |
| Raw identity | 83.84 / 83.84 | 0–99 → 0–99 | 0/0; 0/0 | 0 / 0 | 0/0; 0 | 0 / 0 |
| v2 pairwise MAP; posterior mapping | 77.99 / 46.98 | 25–94 → 0–69 | .25/.05; 0/.30 | 0 / 0 | 0/0; 30 RU1303 + 30 RU581 | .038 / .067 |
| Open DTW, original band 20 | 74.86 / 74.86 | 0–99 → 20–80 | 0/0; .20/.19 | 53 / 14 | 0/0; 39 RU581 | .143 / .202 |
| Open DTW, widened band 60 | 67.94 / 67.94 | 0–99 → 20–41 | 0/0; .20/.58 | 85 / 7 | 0/0; 78 RU581 | .295 / .586 |

For DTW, `(1,0)` repeats a RU581 position (stretch) and `(1,2)` jumps over one RU581 position (compression); neither is counted as a biological gap. The v2 MAP path has no internal insertion/deletion in this pair. Its 30 skips on each axis occur at the ends. The posterior still has high uncertainty: expected unmatched counts are 51.3 on each axis, and the posterior matching mass of RU1303 landmarks 80–99 is only 0.28–0.47. Cell displacement and overlap use v2's posterior mapping, so they need not equal its MAP path coordinates.

## Late RU1303 region and held-out evaluation

RU581 SCLC-A occupies local-position q10–q90 **0.421–0.531** (median 0.480); its SCLC-N occupies **0.709–1.000** (median 0.920). RU1303 SCLC-A has median raw `s_local` 0.750. Values below are RU581 local positions; `—` means the v2 MAP path left that RU1303 landmark unmatched.

| RU1303 landmark | Pairwise cost minimum | v2 posterior mean | v2 MAP | DTW band 20 | DTW band 60 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 80 | .424 | .765 | .556 | .626 | .414 |
| 85 | .414 | .819 | .606 | .667 | .414 |
| 90 | .404 | .874 | .657 | .717 | .414 |
| 95 | .404 | .930 | — | .768 | .414 |
| 99 | .384 | .970 | — | .808 | .414 |

| Evaluation, RU1303 SCLC-A on RU581 axis | Raw | v2 posterior | DTW band 20 | DTW band 60 |
| --- | ---: | ---: | ---: | ---: |
| Median position | .750 | .710 | .626 | .414 |
| Fraction inside RU581 SCLC-N q10–q90 | .915 | .520 | .132 | 0 |
| Held-out marker standardized MSE, landmarks 80–99 | 3.066 | 2.565 | 1.210 | 1.114 |

Held-out marker MSE compares the seven smoothed marker values at RU1303 landmarks 80–99 with RU581 values at each method's mapped landmark, after scaling each marker by its pooled landmark standard deviation. It is descriptive; a flat mapping can lower this score while losing temporal resolution. In particular, band 60 maps **all 20 late landmarks** to RU581 position 0.414, and its path has an 85-step stretch count. That is a severe plateau, not a biologically satisfactory continuous alignment. Band 20 moves the SCLC-A *median* away from RU581 SCLC-N, but late landmarks 90–99 still map into its SCLC-N interval.

## Answer and next fix

The pairwise expression cost supports RU1303's late SCLC-A region near RU581's mid SCLC-A region. The older open DTW can reach it only when the band is widened; then it collapses the late region onto one RU581 position. With its original band it improves the SCLC-A/SCLC-N overlap but still sends the very late tail toward SCLC-N. Pairwise v2's MAP path stops before the tail, while its posterior mean drifts strongly toward RU581's terminal region; the current `(1,1)` plus gap path vocabulary and posterior uncertainty are therefore already limiting **before** a multi-patient barycenter is involved.

The next focused fix should target **pairwise warp mechanics and gap/open-end handling**: allow speed changes without labeling them unmatched, retain separate terminal and internal costs, and constrain long `(1,0)` plateaus or require minimum supported slope. Check the late-region posterior mass and marker trajectories before changing the shared barycenter or the frozen feature cost. This pair does not justify a consensus redesign on its own.

## Artifacts and reproduction

- `cost_matrix_paths.png` / `.pdf`: common pairwise cost, identity, v2 MAP, original-band DTW, and the one widened-band sensitivity path.
- `mapping_functions.png` / `.pdf`: v2 posterior mean and MAP versus both DTW paths.
- `pairwise_heatmap.png` / `.pdf`: side-by-side raw, v2 posterior, and original-band DTW, with RU1303 then RU581; 100 bins, `[0,1]` axes, original gold/red/cyan scale, and gray for empty or fewer than three finite cells.
- `landmark_matches.csv`: all 100 landmark positions, both v2 mappings, both DTW mappings, posterior match mass, and pairwise row minima.
- `pairwise_metrics.json`: unrounded metrics and exact evaluation definitions.
- `run_pairwise.py`: deterministic regeneration from the frozen cache and original H5AD; run `PYTHONPATH=src /data1/chanj3/wangm10/conda_envs/scdebussy-py311/bin/python validation/v21/experiment_round/eight_patient/ru1303_ru581_pairwise/run_pairwise.py` from this branch's root.

The input is `/scratch/chanj3/wangm10/HTAN/results/aligned_scdebussy_preprocessed_v2_ru942_deg_all.h5ad` (checkpoint SHA256 `4f26e98a530bbda380121929d543b367884d0fb17beef58061de948ad09f7dc3`). The frozen landmarks are from `.htan_experiment_cache/exp-htan-eight-patient/eight_patient/results/trajectory_250_unit.npz`; the script verifies that their cell IDs match the input and their 250 genes match the frozen panel. No cell-type labels or held-out markers enter cost construction or path selection.
