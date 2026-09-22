# RU1303→RU581: is the smooth late path reachable or disfavored?

**Answer:** it is reachable, but the current full-path objective disfavors it. This diagnostic reused the frozen trajectory-250/unit-precision pairwise cost and did not fit the cohort or change the barycenter.

Four complete, monotone, all-matched mappings were scored: the previous capped-repeat/high-biological-gap MAP path (**current**); a **smooth** candidate fixing RU1303 landmarks 80–99 to RU581 indices 40–59, with the earlier prefix optimized by the same soft-run graph; the earlier open DTW **plateau** path; and **identity**. All have zero internal biological gaps. RU1303 is fully consumed and RU581 has open ends.

The common decomposition is `C_total = C_expr + C_warp + C_gap + C_regularization`, where `C_expr` sums the frozen unit-precision landmark cost, `C_warp=0.05 × (repeat+skip2 moves)`, `C_gap=0.516918 × (unmatched RU581 prefix+suffix)` here, and `C_regularization=ρ Σ_runs(ell−1)²` for consecutive repeat or skip2 moves. The table uses **ρ=0.02**, the smallest previously tested soft-run penalty. The internal biological-gap multiplier remains 3×, although no candidate uses an internal gap. This scoring reproduces the current capped graph's MAP energy at ρ=0 and the forced smooth graph's energy at ρ=0.02 to numerical tolerance.

| Candidate | `C_expr` (late 80–99) | `C_warp` | `C_gap` | `C_reg` | `C_total` | Late RU581 positions 80 / 90 / 99 | Held-out marker MSE | Longest plateau whole / late | Biological gaps |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Current terminal-like | 77.95 (21.98) | 2.20 | 0 | 8.50 | **88.65** | .616 / .818 / 1.000 | 1.928 | 4 / 1 | 0 |
| **Smooth 0.4–0.6** | **70.75 (14.87)** | 2.10 | **20.68** | .28 | **93.80** | .404 / .505 / .596 | **.760** | 3 / 1 | 0 |
| Plateau near .414 | 67.94 (13.33) | 4.60 | 40.32 | 49.14 | 162.00 | .414 / .414 / .414 | 1.114 | 39 / 20 | 0 |
| Identity | 83.84 (24.32) | 0 | 0 | 0 | **83.84** | .808 / .909 / 1.000 | 3.066 | 1 / 1 | 0 |

The smooth candidate lowers whole-path expression cost by **7.20** relative to the current path and late expression cost by **7.11**; its held-out marker MSE also falls from 1.928 to .760. It loses because it leaves 40 RU581 terminal landmarks unmatched, costing **20.68** under the existing open-end extension rule. The plateau has the lowest expression cost, but pays 40.32 in open-end cost and 49.14 in run penalty; it wins only if these structural costs are ignored.

To rule out a poorly chosen smooth prefix or a recurrence bug, the soft-run graph was solved twice at each **already tested** ρ: unrestricted, and with all late matches fixed to indices 40–59 and biological gaps prohibited. These are exact MAP costs within the graph's 12-move technical state bound; the best smooth path uses much shorter runs, so that bound is inactive.

| ρ | Unrestricted soft-run MAP | Best forced-smooth MAP | Smooth disadvantage |
| ---: | ---: | ---: | ---: |
| .02 | 80.615 | 93.803 | 13.188 |
| .1 | 80.773 | 94.427 | 13.654 |
| .4 | 80.819 | 94.588 | 13.769 |

**Interpretation:** the smooth path is feasible and explicitly found by the same graph, so this is not a search/reachability failure. The pointwise expression objective and held-out markers favor its late placement, while the full objective rewards covering RU581's terminal span strongly enough to reject it. The immediate model issue is **terminal missingness/open-end pricing relative to correspondence cost**, possibly compounded by early-to-middle trajectory geometry; slope penalties alone cannot correct it. No further tuning was performed.

Artifacts: `cost_decomposition.csv`, `paths_and_costs.png`, `comparison_heatmap.png`, and `../path_cost_decomposition.py`. The heatmap keeps the prior RU1303/RU581 order, gold/red/cyan colors, 100 bins, `[0,1]` axes, and gray for empty or fewer than three finite cells. The historical current capped path has a 21-move skip run, and the plateau path a 38-move repeat run; both exceed the soft graph's technical run bound and are scored as explicit counterfactual paths under the common cost, not presented as admissible soft-run MAP paths.
