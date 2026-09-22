# Endpoint-compatibility terminal pricing: five fixed pairwise checks

This diagnostic reuses the frozen trajectory-250/unit-precision matrix, the existing speed-change/run recurrence (warp 0.05, ρ=0.02, β=5, band 60), and unchanged internal biological gap and leading-end prices. It uses one rule for all five directions, without held-out marker fitting or per-pair tuning.

For each direction, the already-computed fixed-0.10 MAP endpoint supplies a *candidate* terminal position (j). With (K=10), the score compares mean expression cost between the final query landmarks and the last 10 retained reference landmarks against their mean cost to **all** skipped reference landmarks:

\[
\Delta_{\rm suffix}=\overline D_{q[90:100],r[j-9:j]}-\overline D_{q[90:100],r[j+1:99]},\qquad
\lambda_{\rm terminal}=0.516918\,\sigma(\Delta_{\rm suffix}/0.25).
\]

The displayed index ranges are inclusive in concept; the code uses exact Python slices. A more negative score means the skipped suffix is less compatible and lowers the terminal price. This is a two-stage diagnostic based on a prior cheap-tail endpoint, not a ready joint-model update.

| Query→reference | Δ suffix | Adaptive price | MAP reference span | Tail skipped | Internal gaps | Plateau / speed run | Mean displacement | Held-out MSE | MAP–posterior Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RU1303→RU581 | −.340 | .105 | 0–64 | 35% | 0 | 3 / 2 | .214 | .928 | .027 |
| RU1303→RU1083 | −.368 | .096 | 0–58 | 41% | 0 | 3 / 2 | .258 | 1.427 | .019 |
| RU1303→RU1444 | −.132 | .192 | 0–94 | 5% | 0 | 2 / 2 | .086 | 1.808 | .031 |
| RU581→RU1083 | −.297 | .121 | 0–93 | 6% | 0 | 3 / 2 | .042 | 1.682 | .015 |
| RU581→RU1303 | −.357 | .100 | 0–74 | 25% | 0 | 6 / 5 | .071 | 1.802 | .041 |

The MSE column uses the seven held-out markers over the full matched query trajectory and the posterior mapping; MAP-only and late-only MSE, cost components, entropy, match mass, and landmark 80/90/99 positions are in [metrics.csv](metrics.csv). The [score/behavior plot](compatibility_vs_terminal_behavior.png) shows that the reverse control's score overlaps the positive cases. Heatmaps compare `Raw | previous/current | fixed 0.10 | adaptive` under the same two-row order, cell-type colors, 100 bins, [0,1] axis, and gray convention: [RU1303→RU581](RU1303_to_RU581_heatmap.png) and [RU581→RU1303](RU581_to_RU1303_heatmap.png).

**Decision: stop before joint fitting.** The rule retains the RU1303→RU581 and RU1303→RU1083 improvements, but RU581→RU1303 receives essentially the same cheap price and still truncates 25% of the reference tail while worsening held-out MSE versus the previous/current alignment (1.737→1.802). It also attenuates the RU1303→RU1444 gain versus fixed 0.10 (MSE 1.697→1.808). Endpoint-versus-suffix expression compatibility alone cannot distinguish these directions reliably; terminal pricing needs a richer model of trajectory coverage or directional biological state before a small joint-fit test. No barycenter or cohort run was performed.
