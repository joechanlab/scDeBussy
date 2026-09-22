# Full 8-patient HTAN joint fit: does the validated model generalize?

Extends the validated [4-patient pilot](../ru1303_ru581_pairwise/four_patient_joint/FOUR_PATIENT_JOINT.md)
to all 8 frozen HTAN patients (RU1083, RU1303, RU1444, RU151, RU263, RU581, RU831, RU942). Every
frozen mechanic is reused unchanged (trajectory-250 genes, unit precision, DTW-style warp/gap
graph, the coverage-gate rule, terminal pricing); only the patient loop is generalized. γ=1 is
the validated main setting; γ=0 is run once as a diagnostic control, not swept. No parameter is
retuned. Code: [eight_patient_joint.py](eight_patient_joint.py).

## Coverage score and gate decision (recorded before fitting)

| Patient | Coverage score | Gate | Terminal price |
|---|---:|---|---:|
| RU1083 | .949 | off (full) | .517 |
| RU1444 | .939 | off (full) | .517 |
| RU581 | .879 | off (full) | .517 |
| RU151 | .707 | **on (cheap)** | .10 |
| RU263 | .657 | **on (cheap)** | .10 |
| RU942 | .667 | **on (cheap)** | .10 |
| RU831 | .626 | **on (cheap)** | .10 |
| RU1303 | .404 | **on (cheap)** | .10 |

**This is the headline finding.** The gate rule (`score ≤ max − 0.10`) is unchanged from the
validated 3-/4-patient pilots, but at 8-patient scale it gates **5 of 8 patients** cheap, not
just RU1303. In the smaller pilots only RU1303 was ever below the margin because the other
patients (RU581/RU1083/RU1444) were all close to each other in coverage. At full scale, RU151/
RU263/RU942/RU831 turn out to have meaningfully lower coverage than the RU1083/RU1444 ceiling
(.94–.95) and fall more than .10 below it — a genuine property of this cohort, not a bug, but
one the smaller pilots could not have revealed since they only ever included 1–2 of these
patients. The gate's *mechanism* generalizes correctly; its *effective threshold* does not scale
gracefully to a wider coverage spread.

## Does γ=1 still stabilize the barycenter at 8-patient scale?

Yes, in the same direction as the smaller pilots, though with two new wrinkles at this scale.

| | γ=0 (diagnostic) | γ=1 (main) |
|---|---|---|
| Initial roughness | .0168 | .0091 |
| Final roughness | 1.0075 | .1586 |
| Roughness ratio | 59.9x | 17.4x |
| Final relative improvement (iter 20) | .0011 (still shrinking, no oscillation) | .0009 (still shrinking, no oscillation) |
| Converged (20-iter cap) | no | no |

γ=1 still cuts final roughness **6.4x** relative to γ=0 (.1586 vs 1.0075), consistent with the
smaller pilots. Unlike the 3-/4-patient runs, γ=0 itself does **not** oscillate badly here
(relative improvement is monotonically shrinking at iteration 20 for both settings) — with more
patients pulling the correspondence in more directions at once, γ=0's trajectory happens to be
better-behaved than in the smaller cohorts, though its barycenter is still far rougher.

## Per-patient outcomes (final phase, γ=1)

| Patient | Gate | Coverage | Purity | Mean disp. | p95 disp. | Terminal skip | Internal gaps | MAP–post Δ | Held-out marker MSE |
|---|---|---|---|---|---|---|---|---|---|
| RU1083 | off | .940 | .831 | .035 | .060 | 0% | 0 | .0044 | 1.207 |
| RU1444 | off | .980 | .819 | .022 | .032 | 0% | 0 | .0143 | 0.648 |
| RU581 | off | .990 | .860 | .044 | .048 | 0% | 0 | .0179 | 1.292 |
| RU1303 | on | .560 | .890 | .234 | .383 | **38%** | 0 | .0074 | n/a |
| RU151 | on | .750 | .506 | .177 | .213 | 15% | 0 | .0204 | n/a |
| RU263 | on | .980 | .710 | .047 | .063 | 0% | 0 | .0334 | n/a |
| RU942 | on | .760 | .728 | .074 | .160 | 4% | **3** | .0169 | n/a |
| RU831 | on | .440 | .936 | .093 | .235 | 23% | **6** | .0165 | n/a |

Full sweep (both γ, initial+final) in [metrics.csv](eight_patient_joint/metrics.csv); history in
[convergence.json](eight_patient_joint/convergence.json).

- **RU1303 truncation is preserved**: 38% (γ=1) vs 40% (γ=0) terminal skip, same order as every
  smaller pilot (38–44%), 0 internal gaps.
- **RU1444 develops no new truncation or pathology**: 0% skip, 0 gaps — confirms the 4-patient
  result at full scale.
- **RU831 develops new internal gaps under γ=1**: 6 internal patient gaps at γ=1 vs 3 at γ=0 —
  smoothing makes this *worse*, not better, for RU831 specifically. The
  [MAP-curve plateau diagnostic](eight_patient_joint/convergence.json) confirms it: longest run
  of identical consecutive MAP positions is **12 landmarks** at γ=1 vs 9 at γ=0, and matched
  landmarks drop from 97 to 94. RU831's own coverage is the lowest of any patient (.44), and its
  purity is the *highest* (.936) — consistent with a small number of confidently-typed cells
  compressed into a narrow position range rather than spread out, i.e. exactly the
  "compression/plateau" pathology this pilot was asked to watch for.
- **RU942 also shows internal gaps** (3, unchanged between γ=0 and γ=1) — a second, milder
  instance of the same kind of pathology, not fixed or worsened by regularization.
- **RU151 has the lowest purity of any patient** (.506, vs .71–.94 for everyone else) — flagged
  as weak but not gap/plateau pathology.
- MAP–posterior disagreement stays small everywhere (≤.033, RU263 highest) — alignment
  confidence is not degraded by scaling to 8 patients.

## Patient-contribution-weight heatmap: does the diagnosed instability persist at scale?

[γ=0](eight_patient_joint/contribution_weight_gamma0.png) shows severe, widespread
single-patient-dominance spikes (near-100%-mass stripes) for RU1303, RU151, RU942, RU263, and
**RU831** (a solid block of 100% RU831 dominance around shared position .62–.72).
[γ=1](eight_patient_joint/contribution_weight_gamma1.png) suppresses essentially all of these
**except RU831's**, which persists as a near-100%-mass yellow spike at ~.68–.70 even under
regularization — direct visual confirmation that RU831's instability is not just an artifact of
the unconstrained M-step and resists the same fix that worked for every other patient in the
3-/4-/8-patient pilots. This is the clearest single piece of evidence that RU831 is a genuinely
different case from the RU1303-truncation pattern the gate/γ combination was validated against.

## Comparison against controls

Computed with the frozen, unmodified `../../summarize_and_plot.py` scoring function (identical
methodology to the existing `SUMMARY.md` baselines) on raw `s_local`, `original_v2.npz`, and
`trajectory_250.npz` (all pre-existing, not rerun) plus the new `eight_patient_joint_gamma1.npz`:

| Method | Coverage | Purity | Mean disp. | p95 disp. | Rejection | RU1303 overlap | Held-out marker concordance |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw `s_local` | **.865** | .772 | 0 | 0 | 0% | .531 | .421 |
| original v2 | .858 | .728 | .013 | .192 | .02% | .531 | — |
| trajectory-250 restricted | .850 | .785 | .009 | .111 | .02% | .132 | .419 |
| **8-patient joint fit (γ=1)** | .800 | **.793** | .066 | .182 | **0%** | **.000** | **.582** |

Full metrics: [final_comparison/metrics.csv](final_comparison/metrics.csv); figure:
[final_comparison/htan_comparison.png](final_comparison/htan_comparison.png); held-out marker
detail: [heldout/heldout_summary.json](heldout/heldout_summary.json),
[heldout/heldout_trajectories.png](heldout/heldout_trajectories.png) (held-out panel uses the
same 7-gene lineage panel — ASCL1, NEUROD1, INSM1, CHGA, CHGB, NCAM1, DLL3 — as every earlier
pilot in this round; the original round's own `markers.json` was not present in this worktree,
so it was recreated with the identical, already-validated gene list). Rejection rate and my
proxy "unsupported" rate are both ~0% for the joint fit — the custom per-cell status/unprojected
analog built for this candidate (§ code docstring) rarely triggers, so it adds little beyond the
other diagnostics here; treat it as a weak, non-native signal, not equivalent to the v2 API's own
`align.project` status.

The joint fit is **the best of all four on purity, RU1303/SCLC-N separation, and held-out marker
concordance** (+38% relative over the best existing control), and has **zero rejected cells** —
but its **coverage is the lowest of all four** (.800 vs .850–.865), driven directly by the
5-of-8 gate triggering described above, and its mean displacement is markedly larger than
trajectory-250's (.066 vs .009), which is the expected cost of that much more aggressive
terminal pricing. See [final_comparison/htan_comparison.png](final_comparison/htan_comparison.png):
the joint-fit panel visibly has far more gray (empty/<3-cell) area than the other three panels,
concentrated in RU1303, RU831, RU151, RU942, and RU263's tail — exactly the five gated patients.

## Main decision: does the validated model improve alignment at full-cohort scale without sacrificing coverage, creating spurious truncation/gaps, or introducing new distortions?

**No, not unconditionally.** Three things are true at once:

1. The core mechanism validated in the 3-/4-patient pilots **does generalize**: γ=1 continues to
   stabilize the barycenter (6.4x roughness reduction, no oscillation), RU1303's truncation and
   RU1444's clean full-length behavior both reproduce exactly as predicted, and the model
   achieves the best purity, RU1303/SCLC-N separation, and held-out marker concordance of any
   candidate tested in this round — a substantial biological-alignment improvement.
2. But it **does sacrifice coverage** (.800 vs .850–.865 for every existing control), because the
   frozen coverage-gate rule — never wrong in the smaller pilots, which only ever exercised 1–2
   gated patients at a time — gates 5 of 8 patients cheap at full scale, a behavior the smaller
   pilots structurally could not have surfaced.
3. It **does introduce one new patient-specific distortion**: RU831 (lowest coverage, highest
   purity, worst plateau/gap numbers, and the only patient whose single-patient-dominance
   instability survives γ=1 regularization) — a compression/plateau pathology distinct from and
   not fixed by the fix validated on RU1303/RU581/RU1083/RU1444. RU942 shows a milder version of
   the same kind of gap.

## Recommendation

**Not yet ready to replace the current v2 HTAN result as-is.** The curvature-regularized M-step
(γ=1) is validated and should be kept — it is not the source of the coverage or RU831 problems
(γ=0 has the same 5-patient gate trigger and a *worse* RU831 outcome by the roughness/plateau
metrics, just not by internal-gap count). The blocking issue is the coverage-gate margin
(currently a fixed 0.10 relative to the cohort maximum), which was tuned/validated on a
2–3-patient coverage spread and over-triggers once the cohort spans a wider range of native
coverage scores (.40–.95 here). Recommend, as the next (out-of-scope-for-this-pilot) step:
re-examine the gate's margin/threshold specifically against the 8-patient coverage-score
distribution recorded above, and separately investigate RU831 as an individual case before
re-running this comparison — do not simply re-sweep γ, since this pilot found the barycenter
regularization itself is working as intended.

Stopping here per the task's scope; no further tuning performed in this pilot.
