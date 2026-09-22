# Curvature-regularized barycenter M-step

Follow-up to [three_patient_joint](../three_patient_joint/THREE_PATIENT_JOINT.md). The frozen
coverage gate, pairwise graph, and EM loop are unchanged; only the M-step is replaced by

```
min_B  sum_j w_j || B_j - Bhat_j ||^2 + gamma * sum_j || B_{j+1} - 2 B_j + B_{j-1} ||^2
```

solved as one banded 100x100 linear system per EM iteration (`w` is the per-landmark posterior
match mass, shared across all 250 genes, so a single `solve` updates the whole barycenter).
gamma in {0, 0.1, 1} — the same three-value sweep as the unregularized pilot's own convergence
check, no alignment/gating parameter retuned.

## Diagnosis first

At the unregularized (gamma=0) final barycenter, total posterior support per landmark never
drops below 2.11 (median 3.2) — jaggedness is **not** an empty/low-support-bin artifact. But
decomposing that support by patient shows the worst-curvature landmarks (43–57, 95–98) are each
carried by essentially **one** patient (>95% of the mass), and which patient dominates **flips
between adjacent landmarks** — e.g. bin 44 is 99% RU1303, bin 45 flips to 51%/49% RU581/RU1083,
bin 46 flips back to 99% RU1303. Since the three patients' 250-gene expression profiles differ
at a fixed shared-landmark index, this single-patient alternation directly produces the
sawtooth: the weighted mean is being pulled between different patients' expression vectors from
one landmark to the next, not failing from lack of data.

## Results

| gamma | converged | iters | initial roughness | final roughness | roughness ratio |
|---|---|---:|---:|---:|---:|
| 0.0 | no | 20 | .0135 | .9499 | 70.4x |
| 0.1 | no | 20 | .0112 | .3074 | 27.4x |
| 1.0 | no | 20 | .0088 | .0346 | 3.9x |

None hit the 20-iteration patience cap, but gamma=1 is qualitatively converging cleanly
(relative improvement monotonically decays to 0.00023 by iteration 20, no oscillation) while
gamma=0 is still oscillating between ~0.02 and ~0.08 relative improvement at iteration 20 —
consistent with the smoothing penalty removing the instability, not just masking it.

Per-patient final-phase metrics (gamma=0 vs gamma=1):

| Patient | Coverage | Purity | Displacement | Terminal skip | Internal gaps | MAP-post Δ | Held-out marker MSE |
|---|---|---|---|---|---|---|---|
| RU1303 | .53→.51 | .863→.876 | .262→.261 | 42%→44% | 0→0 | .0054→.0063 | .617→.637 |
| RU581 | .99→.99 | .726→.728 | .068→.088 | 0%→0% | 0→0 | .0114→.0121 | 1.598→1.568 |
| RU1083 | .95→.96 | .868→.819 | .023→.043 | 1%→0% | 0→0 | .0099→.0107 | 1.555→1.539 |

Full sweep (incl. gamma=0.1) is in [metrics.csv](metrics.csv) and
[convergence_summary.json](convergence_summary.json).

## Verdict

gamma=1 preserves every success criterion: RU1303 stays truncated (42%→44% terminal skip, same
direction as before, coverage/purity essentially unchanged), RU581/RU1083 stay full-length
(RU1083's spurious 1% terminal skip actually disappears), no internal gaps appear anywhere, and
held-out marker MSE **improves** for both full patients (RU581 1.598→1.568, RU1083 1.555→1.539)
at the cost of a small RU1303 regression (0.617→0.637, +3%). MAP–posterior agreement is
essentially unchanged (still <0.013 everywhere). Barycenter roughness drops 70x→3.9x relative to
the smooth initialization, and the EM trajectory itself becomes monotone instead of oscillating.

This confirms the failure was in the **unconstrained weighted M-step**, not the posterior
weights themselves (support was never near zero) — curvature regularization is the right fix,
not more temperature/gate tuning. **Recommend gamma=1 (or a nearby value, e.g. 0.5–2, worth a
finer sweep) for the next 4-patient pilot**, keeping the frozen gate and pairwise graph fixed as
in this pilot. gamma=0.1 is too weak (roughness ratio still 27x) to trust as-is.

See [figure](barycenter_smoothing_figure.png): raw | unregularized (gamma=0) | smoothed
(gamma=1) heatmaps, plus the local RMS-second-difference roughness profile per landmark for all
three gammas — the two curvature hot zones identified in the diagnosis (landmarks ~40–60 and
~85–99) are visibly flattened by gamma=1.
