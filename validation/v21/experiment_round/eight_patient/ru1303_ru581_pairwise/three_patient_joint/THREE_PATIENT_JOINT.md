# Three-patient gated posterior barycenter pilot

This single pilot uses RU1303, RU581, and RU1083 only. It reuses the frozen trajectory-250/unit-precision landmarks and the pairwise graph exactly: `(1,0)/(1,1)/(1,2)` speed moves, warp cost 0.05, quadratic run penalty ρ=0.02, unchanged internal biological gaps, β=5, and band 60. The terminal price is 0.10 when the coverage gate is on and 0.516918 otherwise. Held-out markers and cell-type labels are used only for evaluation.

At initialization, the final 10 expression landmarks of each patient were matched to the other two frozen trajectories. Their median nearest positions gave **frozen** coverage scores RU1303 **.399**, RU581 **.737**, and RU1083 **.662**. Relative to the maximum score .737, the same ten-bin gate (`score ≤ .637`) turned cheap terminal skipping **on only for RU1303**. Scores and gate states were never recomputed during EM; details are in [frozen_gate.json](frozen_gate.json).

The pairwise-informed initialization starts from the mean RU581/RU1083 expression trajectory, performs one gated posterior correspondence and barycenter update, and then enters posterior EM. The M-step is the correspondence-weighted mean of the three patients at each shared landmark; no new model parameter was tuned.

| Phase | Patient | Coverage | Purity | Mean displacement | Terminal skipped | Internal gaps | MAP–posterior Δ | Held-out marker MSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Initial | RU1303 | .56 | .868 | .263 | 40% | 0 | .008 | .628 |
| Final | RU1303 | .53 | .863 | .262 | 42% | 0 | .005 | .617 |
| Initial | RU581 | .99 | .694 | .041 | 0% | 0 | .014 | 1.580 |
| Final | RU581 | .99 | .726 | .068 | 0% | 0 | .011 | 1.598 |
| Initial | RU1083 | .96 | .882 | .015 | 0% | 0 | .006 | 1.552 |
| Final | RU1083 | .95 | .868 | .023 | 1% | 0 | .010 | 1.555 |

Across all three patients, coverage changes **.837→.823**, purity **.732→.755**, and mean displacement **.045→.067** from initialization to final. RU1303 landmarks 80/90/99 map by posterior to **.469/.530/.577** at the final iteration, still in the shared middle rather than the endpoint. The final median position of RU1303 SCLC-A cells is **.436**; RU581/RU1083 SCLC-A medians are **.556/.587**, while their SCLC-N medians are **.930/.928**. Full-patient terminal skips are 0% and 1%. The [three-panel heatmap](three_patient_heatmap.png) uses the same cell-type colors, 100 bins, [0,1] axis, and gray empty/insufficient-support convention. Full per-patient metrics are in [metrics.csv](metrics.csv).

**Convergence and smoothness:** The negative log partition decreased at every iteration, but the pilot did **not** converge by its 20-iteration cap; the final relative objective improvement is still **2.1%**. Barycenter RMS second difference rises from **.0135 to .9499** (about **70×**), with sharp adjacent-landmark expression jumps. Every shared bin retained substantial posterior support (minimum summed match weight across iterations **1.89**), so this is not merely an empty-bin fill artifact. [Convergence history](convergence.json) and the [saved barycenter/curves](pilot_state.npz) permit inspection.

**Decision: do not escalate to four patients yet.** The coverage gate interaction behaves as intended: RU1303 stays truncated, the full patients remain near full length, and MAP/posterior maps agree. The failure is in the unrestricted posterior barycenter update: it makes the shared expression trajectory highly jagged, has not converged, and slightly worsens both full patients' held-out marker MSE (RU581 1.580→1.598; RU1083 1.552→1.555). The next focused step is to stabilize or constrain the barycenter M-step while keeping the frozen gate and pairwise graph fixed. No eight-patient cohort or parameter sweep was run.
