# Gate ablation: is the coverage regression caused by the extra gate activations?

Follow-up to [EIGHT_PATIENT_JOINT.md](EIGHT_PATIENT_JOINT.md), which found the automatic
coverage gate fires on 5/8 patients at full scale (not just RU1303 as validated in the 3-/4-patient
pilots) and coverage drops to .800 vs .850–.865 for existing controls. This pilot manually
freezes the gate to **RU1303 on, all other 7 patients off** — exactly the outcome validated in
the smaller pilots — with every other mechanic (γ=1, graph, genes, precision, terminal price
values) unchanged. Code: [gate_ablation.py](gate_ablation.py), reusing
[eight_patient_joint.py](eight_patient_joint.py)'s functions directly; only which patients
receive the cheap vs full terminal price is overridden.

## Main question: does recovering the gate recover coverage while keeping the gains?

**Partially.** Coverage improves substantially and purity even improves slightly, but held-out
marker concordance gives some of its gain back — and RU831's pathology is **not** resolved by
the ablation.

| Method | Coverage | Purity | RU1303/SCLC-N overlap | Mean disp. | p95 disp. | Held-out marker concordance |
|---|---:|---:|---:|---:|---:|---:|
| raw `s_local` | .865 | .772 | .531 | 0 | 0 | .421 |
| trajectory-250 restricted | .850 | .785 | .132 | .009 | .111 | .419 |
| 8-patient joint, **automatic gate** (5/8 on) | .800 | .793 | .000 | .066 | .182 | **.582** |
| 8-patient joint, **RU1303-only gate** | **.839** | **.804** | .000 | .049 | .116 | .533 |

(All frozen controls loaded from disk, not rerun; scored with the unmodified
`../../summarize_and_plot.py`/`../../heldout_markers.py`. Figure:
[gate_ablation_comparison/htan_comparison.png](gate_ablation_comparison/htan_comparison.png);
full tables: [gate_ablation_comparison/metrics.csv](gate_ablation_comparison/metrics.csv),
[gate_ablation_heldout/heldout_summary.json](gate_ablation_heldout/heldout_summary.json).)

- **Coverage recovers most of the way**: .800 → .839, closing ~65% of the gap to
  trajectory-250's .850, though still short of it and of raw's .865.
- **Purity actually improves further**: .793 → .804 — the best of any candidate tested in this
  round.
- **RU1303/SCLC-N separation is fully preserved**: .000 in both gated configurations, still the
  best of all four methods.
- **Held-out marker concordance partially gives back its gain**: .582 → .533. Still far above
  both existing baselines (.419–.421, i.e. +27% relative instead of +38%), but the automatic
  5-patient gate was itself contributing real signal to that metric, not purely an artifact.
- Displacement drops a lot (mean .066→.049, p95 .182→.116) — expected, since 4 fewer patients
  are being pulled by the cheap terminal price.

## Per-patient effect of restoring full price to RU151/RU263/RU831/RU942

| Patient | Coverage (auto → RU1303-only) | Terminal skip (auto → RU1303-only) | Internal gaps (auto → RU1303-only) |
|---|---|---|---|
| RU1303 (still gated) | .560 → .590 | 38% → 34% | 0 → 0 |
| RU151 | .750 → **.920** | 15% → 0% | 0 → 0 |
| RU263 | .980 → .990 | 0% → 0% | 0 → 0 |
| RU942 | .760 → .790 | 4% → 0% | 3 → 3 |
| RU831 | .440 → .490 | 23% → 0% | **6 → 7** |

RU151 is the patient that benefits most from ungating (coverage +.17, terminal skip 15%→0%) —
it was likely the single biggest contributor to the automatic gate's coverage cost. RU263 was
barely affected either way (it was gated but already near-full coverage under the cheap price,
so the terminal-price change did little). RU831 and RU942's own coverage recovers only modestly
(.44→.49, .76→.79) even with full terminal pricing restored — both remain the two lowest-coverage
patients in the cohort regardless of gate status.

## RU831 plateau/gap pathology: gate-driven or intrinsic?

**Intrinsic, not gate-driven** — this is the clearest finding of the ablation. Restoring RU831 to
full terminal price does not fix its pathology and by one measure makes it slightly worse:

| | Automatic gate (RU831 cheap) | RU1303-only gate (RU831 full price) |
|---|---|---|
| Internal patient gaps | 6 | **7** |
| Longest MAP plateau (landmarks) | 12 | 11 |
| Distinct MAP positions | 61 | 72 |
| Matched landmarks | 94 | 93 |
| Barycenter final roughness (whole fit) | .1586 | .1447 |

The graph-derived internal-gap count gets marginally worse (6→7) under full pricing, while the
plateau-length/distinct-position numbers improve marginally (12→11 landmarks, 61→72 distinct
positions) — a mixed, not clearly-improving picture either way, and nowhere close to RU1444's or
RU581's clean 0-gap behavior under either gate setting. RU831 remains, in both configurations,
the patient with the lowest coverage (.44–.49) and the highest purity (.933–.936) in the cohort —
consistent with a small number of confidently-typed cells compressed into a narrow, broken-up
position range, a distortion this ablation confirms is **not** an artifact of the coverage-gate
over-triggering diagnosed in the previous pilot.

## Heatmap and contribution-weight check

[gate_ablation_comparison/htan_comparison.png](gate_ablation_comparison/htan_comparison.png) —
Raw | trajectory-250 restricted | automatic-gate joint fit | RU1303-only-gate joint fit, same
patient order/colors/bins/axis/status-strip convention as the previous pilot. RU151's row is
visibly far less gray in the RU1303-only-gate panel than the automatic-gate panel, matching its
large coverage recovery; RU831's row stays heavily gray in both gated panels, consistent with
its pathology being unaffected by the gate setting.
[gate_ablation/contribution_weight_ru1303_only_gate.png](gate_ablation/contribution_weight_ru1303_only_gate.png)
still shows RU831 as the one patient whose single-patient match-mass dominance persists near
shared position .65–.72, exactly as under the automatic gate — further confirming this is an
RU831-specific, gate-independent instability.

## Recommendation

The gate ablation isolates the coverage regression cleanly: it **is** caused by the automatic
gate over-triggering on RU151/RU263/RU942/RU831 (mostly RU151), and reverting to the
RU1303-only gate validated in the smaller pilots recovers most (not all) of the lost coverage
while *improving* purity and keeping RU1303's separation intact, at a partial (not full) cost to
held-out marker concordance. RU831's compression/plateau pathology is a **separate, intrinsic**
issue unrelated to the gate and unresolved by this ablation — it needs its own investigation
(likely in its own expression/cost-matrix structure) before any wider rollout.

As instructed, no gate threshold tuning was performed here — this pilot only tested the
binary RU1303-only override already validated in earlier pilots, not a revised margin. Stopping
after this ablation.
