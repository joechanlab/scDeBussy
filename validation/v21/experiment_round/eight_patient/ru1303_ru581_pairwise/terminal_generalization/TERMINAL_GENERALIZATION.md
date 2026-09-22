# Fixed terminal-price pairwise checks

Five directed pairs used the frozen trajectory-250 landmarks and unit-precision expression cost. Both settings retain the same warp cost (0.05), quadratic run penalty (ρ=0.02; maximum technical run 12), internal biological gap cost (3× existing gap terms), leading reference price (0.516918), band (60), and posterior β (5). Only the **reference terminal-tail** price differs: previous/current = 0.516918, cheap = 0.10. All source landmarks are consumed. Held-out ASCL1, NEUROD1, INSM1, CHGA, CHGB, NCAM1, and DLL3 are used only for evaluation.

The table reports MAP reference span and gaps, then posterior displacement and full-trajectory marker MSE. “Δ” is mean absolute MAP–posterior mapping difference; “H/mass” is mean per-landmark conditional posterior entropy in nats / mean match mass. The [metrics CSV](metrics.csv) additionally includes path-cost decomposition, late-only disagreement/entropy/mass/MSE, and landmark 80/90/99 positions.

| Direction | Tail price | MAP ref span | Tail skipped | Internal gaps | Plateau / speed run | Mean displacement | Δ | H / mass | Held-out MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RU1303→RU1083 | 0.517 | 0–99 | 0% | 0 | 3 / 2 | .010 | .017 | 2.20 / 1.00 | 2.197 |
| RU1303→RU1083 | 0.10 | 0–58 | 41% | 0 | 3 / 2 | .258 | .019 | 2.07 / 1.00 | 1.429 |
| RU1303→RU1444 | 0.517 | 0–99 | 0% | 0 | 2 / 2 | .052 | .036 | 2.22 / 1.00 | 1.887 |
| RU1303→RU1444 | 0.10 | 0–74 | 25% | 0 | 3 / 2 | .146 | .037 | 2.39 / 1.00 | 1.697 |
| RU581→RU1083 | 0.517 | 0–99 | 0% | 0 | 3 / 2 | .044 | .013 | 2.05 / 1.00 | 1.707 |
| RU581→RU1083 | 0.10 | 0–92 | 7% | 0 | 3 / 2 | .042 | .016 | 2.12 / 1.00 | 1.677 |
| RU831→RU581 | 0.517 | 0–99 | 0% | 26 patient | 1 / 2 | .120 | .014 | 2.03 / .75 | 2.345 |
| RU831→RU581 | 0.10 | 0–92 | 7% | 26 patient | 1 / 2 | .096 | .022 | 2.24 / .75 | 2.163 |
| RU581→RU1303 | 0.517 | 0–99 | 0% | 0 | 5 / 4 | .075 | .031 | 1.99 / 1.00 | 1.737 |
| RU581→RU1303 | 0.10 | 0–74 | 25% | 0 | 6 / 5 | .071 | .041 | 2.18 / 1.00 | 1.802 |

The RU1303→RU1083 late MAP landmarks 80/90/99 move from .838/.949/1.000 to .414/.485/.586; the posterior moves to .464/.530/.607. RU1303→RU1444 shifts less completely: late MAP .535/.636/.747 and posterior .608/.702/.806 at 0.10. The full-trajectory RU581→RU1083 control remains mostly intact (7% tail skipped) and marker MSE changes little. RU831→RU581 still has 26 unmatched **source** landmarks and mean match mass .75, so cheap **reference** terminal pricing does not resolve that partial trajectory. In the reverse RU581→RU1303 direction, the cheap price drops 25% of the reference tail and worsens held-out MSE; its longest plateau/run also rises to 6/5.

Heatmaps, each with identical pair-specific row order across panels, cell-type colors, 100 bins, [0,1] axis, and gray for empty or <3 finite-cell bins:

- [RU1303→RU1083](RU1303_to_RU1083_heatmap.png)
- [RU1303→RU1444](RU1303_to_RU1444_heatmap.png)
- [RU581→RU1083](RU581_to_RU1083_heatmap.png)
- [RU831→RU581](RU831_to_RU581_heatmap.png)
- [RU581→RU1303](RU581_to_RU1303_heatmap.png)

**Decision: stop before barycenter fitting.** Price 0.10 improves the two RU1303 directions, but its directional 25% truncation with worse marker agreement in RU581→RU1303 is inconsistent with a robust universal terminal price. The recurrence prices only unmatched **reference** suffixes cheaply; it still requires every source landmark to be consumed, so it cannot turn RU831's unmatched source segment into terminal non-overlap. The next diagnostic should clarify which endpoint is allowed to be partial and when a full-trajectory reference suffix may be dropped. No per-pair tuning, cohort run, or barycenter change was made.
