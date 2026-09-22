# RU1303→RU581: terminal-tail pricing

The fixed 100×100 trajectory-250, unit-precision cost matrix was reused. RU1303 remains fully consumed; RU581's leading-end price (0.516918 per landmark), internal biological gap penalties (3× the existing gap terms), speed cost (0.05), quadratic run penalty (ρ=0.02), band (60), and posterior β (5) are unchanged. Only the RU581 **terminal** unmatched price changes. Held-out ASCL1, NEUROD1, INSM1, CHGA, CHGB, NCAM1, and DLL3 are used only for scoring.

## MAP objective decomposition

| Tail price | Expression | Warp | Internal gap | RU581 leading | RU581 tail | Run | Total | Tail length | Biological gaps | Longest plateau / speed run |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.516918 (current) | 79.395 | 1.100 | 0 | 0 | 0 | 0.120 | 80.615 | 0 | 0 | 2 / 3 |
| 0.25 | 75.955 | 1.850 | 0 | 0 | 2.250 | 0.180 | 80.235 | 9 | 0 | 3 / 3 |
| 0.10 (cheap) | 71.051 | 1.950 | 0 | 0 | 3.500 | 0.080 | 76.581 | 35 | 0 | 3 / 2 |
| 0 (diagnostic) | 69.788 | 2.600 | 0 | 0 | 0 | 0.360 | 72.748 | 50 | 0 | 3 / 2 |

The sum of the components equals the graph's MAP cost in each setting. The internal gap count includes both patient and reference gaps; both are zero throughout.

## RU1303 landmarks 80–99

| Tail price | MAP RU581 positions at 80 / 90 / 99 | MAP late range | MAP landmarks in 0.4–0.6 | Posterior positions at 80 / 90 / 99 | Posterior landmarks in 0.4–0.6 | Held-out MSE, MAP / posterior | Longest late plateau |
|---:|---|---|---:|---|---:|---:|---:|
| 0.516918 | .707 / .838 / 1.000 | .707–1.000 | 0/20 | .729 / .855 / .988 | 0/20 | 2.109 / 2.247 | 1 |
| 0.25 | .586 / .747 / .909 | .586–.909 | 1/20 | .699 / .815 / .936 | 0/20 | 1.518 / 1.769 | 1 |
| 0.10 | .465 / .545 / .646 | .465–.646 | 16/20 | .519 / .595 / .685 | 11/20 | .916 / 1.154 | 2 |
| 0 | .414 / .444 / .495 | .414–.495 | 20/20 | .497 / .565 / .636 | 15/20 | .897 / .944 | 3 |

All late MAP and posterior landmark mappings preserve order, and late posterior match mass is effectively 1. The [late-landmark CSV](late_landmark_mapping.csv) gives every 80–99 position and match mass; [metrics.csv](metrics.csv) gives the complete decomposition. The [four-panel heatmap](comparison_heatmap.png) uses the previous pairwise figure's two-row order, cell-type colors, 100 bins, common [0,1] axis, and gray rule for empty or insufficient-support bins.

**Conclusion.** Terminal non-overlap needs its own price, separate from internal missing biology: at 0.10 the MAP path moves mostly into RU581 SCLC-A without biological gaps or long plateaus, and the held-out marker MSE falls sharply. Zero cost pushes every late MAP landmark into 0.4–0.6, but compresses them into .414–.495 and leaves half of RU581 unmatched. The posterior still extends beyond .6 at both cheaper settings, so this diagnostic supports separate terminal pricing but does not by itself establish a fully satisfactory posterior alignment. No further tuning or cohort run was done.
