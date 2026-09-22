# Expression-only coverage gate: five fixed pairwise checks

The frozen trajectory-250/unit-precision costs, recurrence, warp/run penalty (ω=0.05, ρ=0.02), internal biological gap cost, band 60, and posterior β=5 are unchanged. For each of the four patients in these five pairs, I averaged its final 10 landmark expression vectors, found that centroid's nearest expression position on each of the other three patients' frozen trajectories, and took the median normalized position as its **coverage score**. No cell-type labels or held-out markers entered this score or the gate. [Patient scores and peer matches](patient_coverage.json) are saved separately.

| Patient | Coverage score | Nearest positions on the three peers |
|---|---:|---|
| RU1303 | .404 | RU581 .394; RU1083 .404; RU1444 .414 |
| RU581 | .778 | RU1303 .697; RU1083 .778; RU1444 .899 |
| RU1083 | .697 | RU1303 .697; RU581 .626; RU1444 .970 |
| RU1444 | .697 | RU1303 .697; RU581 .657; RU1083 .939 |

One shared rule was applied to every direction: use terminal price **0.10** only if `query coverage ≤ reference coverage − 0.10`; otherwise keep the original price **0.516918**. The 0.10 coverage margin is a fixed ten-bin separation, not fitted to labels or marker outcomes. All source landmarks remain consumed. The table compares posterior held-out-marker MSE over the full query trajectory; “Δ map” is mean absolute MAP–posterior mapping difference. The [metrics CSV](metrics.csv) also contains posterior entropy/mass, exact MAP path costs, plateau/run lengths, and late-only MSE.

| Query→reference | Coverage q / r (difference) | Gate | Tail skipped: fixed / adaptive / gated | Internal gaps | MSE: fixed / adaptive / gated | Δ map, gated |
|---|---|---|---|---:|---|---:|
| RU1303→RU581 | .404 / .778 (−.374) | on | 35% / 35% / 35% | 0 | .921 / .928 / .921 | .026 |
| RU1303→RU1083 | .404 / .697 (−.293) | on | 41% / 41% / 41% | 0 | 1.429 / 1.427 / 1.429 | .019 |
| RU1303→RU1444 | .404 / .697 (−.293) | on | 25% / 5% / 25% | 0 | 1.697 / 1.808 / 1.697 | .037 |
| RU581→RU1083 | .778 / .697 (+.081) | off | 7% / 6% / 0% | 0 | 1.677 / 1.682 / 1.707 | .013 |
| RU581→RU1303 | .778 / .404 (+.374) | off | 25% / 25% / 0% | 0 | 1.802 / 1.802 / 1.737 | .031 |

The gated RU1303→RU581 map retains the useful 35% reference-tail skip and moves its late region toward RU581's middle. The reverse direction retains the reference tail and restores the previous alignment's marker MSE (1.737 versus 1.802 under both cheap-tail alternatives). RU581→RU1083 also stays fully covered; its MSE rises slightly by .030 relative to fixed 0.10, so the gate is not uniformly best on the held-out score, but it avoids truncating this full-trajectory control.

Heatmaps compare `Raw | fixed 0.10 | endpoint adaptive | coverage gate` using the same pair-specific row order, colors, 100 bins, [0,1] axis, and gray empty/insufficient-support rule: [RU1303→RU581](RU1303_to_RU581_heatmap.png) and [RU581→RU1303](RU581_to_RU1303_heatmap.png).

**Decision.** In these five prespecified pairs, a simple coverage gate preserves all three useful RU1303 truncations and prevents the harmful reverse truncation without introducing internal gaps. It is reasonable to carry this fixed rule into the **next small joint-fit test**, while treating the four-patient coverage estimate and ten-bin margin as provisional rather than a validated cohort-wide solution. No barycenter or cohort run was performed here.
