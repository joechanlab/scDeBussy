# RU1303→RU581: soft speed-run penalty

**Outcome: slope/run regularization alone is insufficient.** This experiment reused the frozen RU1303×RU581 trajectory-250, unit-precision cost matrix. RU1303 was fully consumed; RU581 had open begin/end. No cohort fit, gene selection, or barycenter code was run.

The pairwise graph retained ordinary `(1,1)` matches, matched-repeat `(1,0)` and reference-skip `(1,2)` speed moves, and separate affine biological gaps. Each speed move had the existing `omega=0.05` cost. For a consecutive run of `ell` repeat or skip moves, the added cost was `rho(ell-1)^2`, implemented by its incremental edge cost. The biological-gap price stayed at the previous **3× internal-gap** setting, which keeps the capped comparator's landmarks matched; `beta=5` was fixed for capped and regularized posteriors. A 12-move technical state bound was never active in the selected MAP path (largest repeat run 1, skip run 3). Only `rho ∈ {0.02, 0.1, 0.4}` was tested.

The figure candidate was selected **without held-out markers or cell-type labels**: among gap-free paths, choose the smallest mean distance of the posterior positions at landmarks 80–99 from RU581's target interval 0.4–0.6. This selects `rho=0.02`; no candidate passes the target.

| Method | RU581 positions at RU1303 landmarks 80 / 90 / 99 | Held-out marker MSE, 80–99 | RU1303 mean / max cell displacement | Longest exact plateau, whole / late | Longest repeat / skip run | Internal biological gaps |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Raw identity | .808 / .909 / 1.000 | 3.066 | 0 / 0 | 1 / 1 | 0 / 0 | 0 |
| Current v2 posterior, β=1 | .765 / .874 / .970 | 2.565 | .038 / .067 | 1 / 1 | 0 / 0 | 0¹ |
| Unconstrained open DTW, band 60 MAP | .414 / .414 / .414 | 1.114 | .295 / .586 | **39 / 20** | 38 / 6 | 0 |
| Capped-repeat DTW posterior, high gap cost | .723 / .850 / .990 | 2.211 | .067 / .089 | 4 / 1 | 3 / **21** | 0 |
| Run penalty ρ=.02 posterior (**figure choice**) | .729 / .855 / .988 | 2.247 | .064 / .084 | **2 / 1** | **1 / 3** | 0 |
| Run penalty ρ=.1 posterior | .739 / .860 / .986 | 2.335 | .057 / .075 | 2 / 1 | 1 / 2 | 0 |
| Run penalty ρ=.4 posterior | .747 / .865 / .984 | 2.412 | .050 / .066 | 2 / 1 | 1 / 1 | 0 |

¹ The v2 MAP path has **30 terminal** patient skips (25 leading, 5 trailing), but zero *internal biological* gaps. The regularized MAP paths have zero internal gaps and essentially full late posterior match mass. The compact `metrics.csv` gives unrounded values, terminal skips, posterior entropy, and move counts.

The soft penalty works mechanically: it reduces the unconstrained 20-landmark late plateau to one and the longest compression run from 6 to 3 at ρ=.02, without escaping through biological gaps. Yet the entire late posterior range remains 0.729–0.988, above RU581's SCLC-A q10–q90 interval (0.421–0.531) and approaching its SCLC-N region. Raising ρ to .1 or .4 shortens speed runs further but shifts the late mapping still closer to identity/terminal RU581. The selected regularized MAP and posterior follow the same broad upward trajectory, so posterior averaging is not the main remaining issue in this comparison.

**Stop here.** With the frozen pairwise costs and full RU1303 coverage, discouraging local speed extremes cannot reconcile the early-to-middle path with the late pointwise cost minimum near RU581 0.4. The next diagnostic, if pursued separately, should inspect full-path cost contributions or trajectory representation; this result does not support integrating the recurrence into barycenter EM.

Artifacts: `metrics.csv`, `mapping_functions.png`, `comparison_heatmap.png`, and `../slope_regularization.py`. The five-panel heatmap uses the previous RU1303/RU581 order, gold/red/cyan colors, 100 bins, `[0,1]` axes, and gray for empty or fewer than three finite cells. Its regularized panel is the prespecified geometry choice ρ=.02, **not** a successful model.
