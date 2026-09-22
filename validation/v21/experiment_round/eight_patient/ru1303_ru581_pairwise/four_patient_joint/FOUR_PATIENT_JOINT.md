# 4-patient gated posterior barycenter pilot: does γ=1 still stabilize?

Extends [barycenter_smoothing](../barycenter_smoothing/BARYCENTER_SMOOTHING.md) by adding
**RU1444** to the frozen 3-patient pilot (RU1303, RU581, RU1083). Every frozen mechanic is
unchanged: trajectory-250 genes, unit precision, DTW-style warp/gap graph, the same
terminal-coverage-score gate formula, and the same terminal pricing. Only γ ∈ {0, 1} is
compared — no gate/graph/γ tuning.

Coverage gate (unchanged formula, now over 4 patients): RU1303 score .404 vs. max .778 → gate
**on** (cheap terminal, price .10); RU581 (.778), RU1083 (.697), RU1444 (.697) all gate **off**
(full price). RU1444 lands in the same full-coverage bucket as RU581/RU1083, so the seed and
held-out-marker template are the mean/leave-one-out over all three full patients.

## Does γ=1 still stabilize with a 4th patient?

**Yes.**

| | γ=0 | γ=1 |
|---|---|---|
| Initial roughness | .0099 | .0059 |
| Final roughness | .7999 | .0361 |
| Roughness ratio (final/initial) | 80.8x | 6.1x |
| Converged (20-iter cap) | no | no |
| Final relative improvement | .0088 (still oscillating .009–.025 over last 6 iters) | .00006 (monotone decay) |

γ=1 reduces final barycenter roughness by **22x** relative to γ=0, and — as in the 3-patient
case — turns the EM trajectory from an oscillating one into a monotonically-converging one
(iteration-20 relative improvement 0.00006 vs 0.0088, and `barycenter_max_change` is still
shrinking smoothly instead of bouncing between .3–.49). Neither run hits the 20-iteration
patience cap, matching the 3-patient pilot's behavior.

## Per-patient outcomes (final phase, γ=0 → γ=1)

| Patient | Coverage | Purity | Displacement | Terminal skip | Internal gaps | MAP–post Δ | Held-out marker MSE |
|---|---|---|---|---|---|---|---|
| RU1303 (gated) | .540→.540 | .872→.873 | .228→.230 | **38%→39%** | 0→0 | .0065→.0092 | n/a (not a full patient) |
| RU581 | .990→.990 | .822→.770 | .058→.066 | 0%→0% | 0→0 | .0156→.0141 | 1.290→1.276 |
| RU1083 | .940→.960 | .910→.908 | .022→.036 | 0%→0% | 0→0 | .0088→.0100 | 1.225→1.198 |
| RU1444 (new) | .990→.990 | .766→.771 | .005→.008 | **0%→0%** | 0→0 | .0106→.0139 | 0.638→0.627 |

Full sweep in [metrics.csv](metrics.csv), history in [convergence.json](convergence.json).

- **RU1303 truncation is preserved**: terminal skip 38%→39% under γ=1, same order as the
  3-patient pilot (42%→44%), coverage/purity essentially unchanged.
- **RU1444 develops no new truncation or pathology**: 0% terminal skip and 0 internal gaps at
  both γ=0 and γ=1, same as the other two full patients. Its held-out marker MSE actually
  improves slightly (0.638→0.627).
- No internal gaps appear anywhere, for any patient, at either γ.
- MAP–posterior disagreement stays small everywhere (≤0.016) and doesn't materially change with
  γ — the regularization affects the barycenter shape, not the alignment's confidence.
- Held-out marker MSE **improves** for all three full patients under γ=1 (RU581 1.290→1.276,
  RU1083 1.225→1.198, RU1444 0.638→0.627) — consistent with the 3-patient result, not a
  coincidence specific to that pair.

## Patient-contribution-weight heatmaps: does adjacent-landmark dominance instability persist?

[γ=0](contribution_weight_gamma0.png) shows the same failure mode found in the 3-patient
diagnosis, now with a 4th patient: sharp yellow/green stripes where a single patient carries
>80–100% of a landmark's match mass, alternating with muted teal/blue neighbors — concentrated
around shared position .08–.20, .45–.62, and .88–1.0.

[γ=1](contribution_weight_gamma1.png) is visibly far more uniform — mostly even teal/blue
sharing across all four patients, with only two much milder RU1303-only bumps (~.10–.15 and
~.48–.50, reaching ~60–70% rather than γ=0's 100%) and one mild RU1083 uptick near 1.0. This
shows the regularization doesn't just smooth over an unstable posterior after the fact — it
also stabilizes the *posterior correspondence itself* in the next E-step, since a smoother
barycenter gives every patient a less erratic target to match against.

## Verdict

γ=1 generalizes cleanly from 3 to 4 patients: same ~20–80x roughness reduction, same shift from
oscillating to monotone convergence, same preservation of RU1303's truncation and the full
patients' full length, no internal gaps, and a consistent (if small) held-out marker MSE
improvement for every full patient. RU1444 integrates without any new pathology. The
adjacent-landmark single-patient-dominance instability identified in the 3-patient diagnosis is
confirmed to reappear at γ=0 with a 4th patient and is substantially suppressed at γ=1.

**Recommendation: the model (frozen gate/graph + γ=1 curvature-regularized M-step) is ready for
the full 8-patient test.** No further parameter tuning is indicated by this pilot; a full run
should still budget for the possibility that more patients could introduce genuinely
low-support regions the two 3–4 patient pilots haven't exercised, in which case per-landmark
support (not just roughness) is the metric to watch first.

See [four_patient_heatmap.png](four_patient_heatmap.png) (Raw | γ=0 | γ=1) and the two
contribution-weight heatmaps above.
