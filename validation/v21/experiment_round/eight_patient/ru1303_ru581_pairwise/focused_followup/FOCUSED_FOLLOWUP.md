# RU1303→RU581: posterior, path vocabulary, and plateau check

This is a pairwise diagnostic on the **same frozen trajectory-250, unit-precision 100×100 cost matrix** used in `RU1303_RU581_PAIRWISE.md`. No HTAN cohort fit or barycenter update was run. The RU581 SCLC-A q10–q90 interval is 0.421–0.531; SCLC-N starts at q10 0.709. Held-out markers were used only after paths were computed.

## Small tests

- **v2 temperature:** the unchanged open affine-gap lattice was evaluated at `beta = 0.5, 1, 2, 5, 10`. Its MAP path is identical at every beta; beta changes only the path posterior.
- **Run-limited DTW:** a pairwise-only graph consumes every RU1303 landmark while leaving RU581's ends open. It has ordinary `(1,1)` matches, matched-repeat `(1,0)` and reference-skip `(1,2)` moves, each speed move priced separately at `omega=0.05`. Biological patient/reference gaps remain separate affine-gap states. A match run may use the same RU581 landmark at most **four** times. The path stays within a 60-bin band, wide enough to reach the observed late cost minimum near RU581 index 40. An unconstrained band-60 DTW MAP path from the prior diagnostic provides the plateau comparison.
- **Gap escape control:** the same constrained graph was checked at the original internal-gap price and one 3× internal-gap price; terminal pricing, cost matrix, warp cost, and repeat cap were fixed. This is a diagnostic, not a parameter search.

| Method | RU581 position at RU1303 landmarks 80 / 90 / 99 (posterior) | Mean late match mass | Late MAP–posterior mean difference¹ | Full path entropy | Held-out late marker MSE | Longest MAP plateau | Late MAP matched rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v2 β=.5 | .789 / .893 / .985 | .383 | .235 | 161.7 | 2.854 | 1 | 15/20 |
| v2 β=1 | .765 / .874 / .970 | .425 | .215 | 137.8 | 2.565 | 1 | 15/20 |
| v2 β=2 | .709 / .813 / .901 | .475 | .154 | 73.3 | 1.686 | 1 | 15/20 |
| v2 β=5 | .585 / .674 / .728 | .701 | .021 | 8.2 | 1.304 | 1 | 15/20 |
| v2 β=10 | .560 / .660 / .736 | .745 | .003 | 5.0 | 1.330 | 1 | 15/20 |
| constrained β=1, original gaps | .758 / .868 / .981 | .771 | .011¹ | 165.8 | 2.460 | 1 | **1/20** |
| constrained β=5, original gaps | .763 / .863 / .979 | .728 | .010¹ | 75.5 | 2.488 | 1 | **1/20** |
| constrained β=5, 3× internal gaps | .723 / .850 / .990 | **1.000** | .042 | 101.0 | 2.211 | **4** | **20/20** |

¹ Difference is computed only where the MAP path actually matches a late landmark. For the constrained original-gap rows it covers just **one** of 20 late landmarks and must not be read as broad MAP/posterior agreement. The compact machine-readable table (`metrics.csv`) also records mean late row-conditional entropy, minimum match mass, maximum MAP/posterior difference, mapped positions for both MAP and posterior, path cost, gap/warp moves, and SCLC-A/SCLC-N overlap.

The v2 MAP sends landmark 80 to 0.556 and 90 to 0.657, and leaves 95–99 unmatched. Increasing beta to 5 largely removes the posterior's drift away from that MAP path (mean late difference .215→.021), so **posterior averaging is one source of the original β=1 failure**. It does not make the v2 MAP path reach RU581's mid SCLC-A region for the full tail.

The unconstrained band-60 DTW path sends **all** landmarks 80–99 to RU581 position 0.414, a 20-landmark plateau. With a four-repeat cap, the original gap price instead yields a MAP path with 51 biological patient gaps and only the final late landmark matched. Raising the internal-gap cost 3× removes those gaps (55 ordinary, 22 repeat, 22 skip moves) and caps the longest plateau at four, but both MAP and posterior still map the late tail upward: MAP positions at landmarks 80/90/99 are 0.616/0.818/1.000, posterior positions 0.723/0.850/0.990. Late posterior match mass is essentially one, and MAP/posterior mean difference is .042, so this remaining failure is **not** explained by missing posterior support or a long flat run.

## Decision

**Success criteria are not met.** Sharpening v2 fixes much of its posterior averaging, while the constrained DTW vocabulary fixes the 20-landmark plateau only by accepting late alignment toward RU581's SCLC-N end (or, at the original gap price, by dropping half of RU1303 as biological gaps). The low pointwise costs near RU581 0.4 cannot be used by a full, monotone, short-run path without paying for conflicting earlier trajectory correspondence. The next small investigation should inspect **trajectory representation and full-path cost contributions** for this pair, especially the early-to-middle region that forces the later path upward. Do not integrate this recurrence into barycenter EM or tune a wider grid from these results.

Artifacts: `metrics.csv`, `cost_mapping.png`, `comparison_heatmap.png`, and `../focused_followup.py`. The heatmap preserves the prior pairwise RU1303/RU581 row order, gold/red/cyan scale, 100 bins, `[0,1]` axis, and gray for empty or fewer than three finite cells. Its constrained panel uses the 3×-gap, four-repeat, β=5 posterior because it is the only tested constrained setting whose MAP path matches all 20 late landmarks; it is **not** a passing winner.
