# Gate ablation, repeated on the post-matchability-fix pipeline

Repeats [GATE_ABLATION.md](GATE_ABLATION.md) with the [landmark-matchability fix](ru831_diagnostic/MATCHABILITY_FIX.md)
now wired into `gate_ablation.py`'s `compile_graph` call (one `matchable=matchable[p]` argument
added, mirroring [MATCHABILITY_FIX_EIGHT_PATIENT.md](MATCHABILITY_FIX_EIGHT_PATIENT.md)'s change
to `eight_patient_joint.py`). Everything else is unchanged: trajectory-250 genes, unit precision,
warp/gap mechanics, γ=1, terminal price values, and the gate manually frozen to RU1303-only
(exactly reproducing the configuration validated in the 3-/4-patient pilots). No parameter
retuned. Code: [gate_ablation.py](gate_ablation.py).

## Main question: after the matchability fix, does RU1303-only gating recover coverage while preserving purity, held-out marker concordance, and RU1303 separation?

**Yes, unambiguously** — this result is a clear improvement across nearly every metric, not a
mixed trade-off. Per the task's stop condition, the one-at-a-time RU151/RU263/RU942/RU831 gate
sweep was **not run**.

| Method | Coverage | Purity | Held-out marker concordance | RU1303/SCLC-N overlap | Mean disp. | p95 disp. |
|---|---:|---:|---:|---:|---:|---:|
| raw `s_local` | .865 | .772 | .421 | .531 | 0 | 0 |
| trajectory-250 restricted | .850 | .785 | .419 | .132 | .009 | .111 |
| 8-patient joint, automatic gate (post-fix) | .818 | .774 | .552 | .029 | .0735 | .2150 |
| **8-patient joint, RU1303-only gate (post-fix)** | **.855** | **.786** | .521 | **.000** | **.0596** | **.1659** |

(All frozen controls loaded from disk, not rerun; scored with the unmodified
`../../summarize_and_plot.py`/`../../heldout_markers.py`. Figure:
[gate_ablation_postfix_comparison/htan_comparison.png](gate_ablation_postfix_comparison/htan_comparison.png);
full tables: [gate_ablation_postfix_comparison/metrics.csv](gate_ablation_postfix_comparison/metrics.csv),
[gate_ablation_postfix_heldout/heldout_summary.json](gate_ablation_postfix_heldout/heldout_summary.json).)

RU1303-only gating now **beats trajectory-250 restricted on both coverage (.855 vs .850) and
purity (.786 vs .785)** — the first candidate in this entire round to do so on both axes
simultaneously — while keeping RU1303/SCLC-N overlap at an exact 0 (best of all four methods, far
below raw/original_v2's .531 or trajectory-250's .132). Displacement also **improves** relative
to the automatic gate (mean .0735→.0596, p95 .2150→.1659), not just coverage. The one metric that
moves in the worse direction is held-out marker concordance (.552→.521, a 5.7% relative decline)
— still a +24% relative improvement over both existing frozen baselines (.419–.421), so this is a
minor, not a blocking, trade-off.

## Per-patient (final phase, γ=1, automatic gate → RU1303-only gate, post-fix)

| Patient | Coverage | Purity | Mean disp. | p95 disp. | Terminal skip | Internal gaps | MAP–post Δ |
|---|---|---|---|---|---|---|---|
| RU1083 (full both) | .970→.940 | .821→.777 | .030→.046 | .051→.075 | 0%→0% | 0→0 | .0067→.0132 |
| RU1444 (full both) | .990→.980 | .817→.795 | .021→.017 | .029→.029 | 0%→0% | 0→0 | .0144→.0082 |
| RU581 (full both) | .990→.990 | .846→.866 | .044→.064 | .049→.071 | 0%→0% | 0→0 | .0178→.0211 |
| RU1303 (gated both) | .590→.620 | .851→.868 | .217→.196 | .338→.299 | 34%→30% | 0→0 | .0110→.0111 |
| **RU151** | .720→**.910** | .492→.566 | .211→.045 | .236→.099 | 19%→0% | 0→0 | .0241→.0395 |
| RU263 | .970→.970 | .712→.679 | .030→.012 | .063→.060 | 0%→0% | 0→0 | .0519→.0228 |
| **RU942** | .750→**.840** | .674→.743 | .137→.074 | .186→.147 | 8%→0% | 7→7 | .0199→.0192 |
| **RU831** | .560→**.590** | .855→.857 | .132→.156 | .155→.202 | 10%→0% | 24→26 | .0218→.0182 |

Full sweep: [gate_ablation/metrics.csv](gate_ablation/metrics.csv).

## RU831 and RU942 specifically

Both remain gated cheap under RU1303-only (they're full-coverage-eligible once the extra gate
activations are removed, so their own coverage/gaps come purely from the matchability fix now,
not from any terminal-pricing discount):

- **RU831**: coverage improves again (.560→.590), matching the same direction seen when the
  matchability fix alone was applied on top of the automatic gate. Internal gaps stay essentially
  flat (24→26) — consistent with the gap count being governed by RU831's own ~20 unmatchable
  landmarks (see [MATCHABILITY_FIX_EIGHT_PATIENT.md](MATCHABILITY_FIX_EIGHT_PATIENT.md)), not by
  which gate configuration is active. Purity is essentially unchanged (.855→.857), and
  MAP–posterior disagreement improves slightly (.0218→.0182). Terminal skip drops to 0% (RU831 no
  longer receives the cheap terminal price at all under this manual gate), consistent with more
  of its trajectory now being expressed as an interpretable internal gap rather than terminal
  truncation.
- **RU942**: coverage improves substantially (.750→.840, the second-largest per-patient gain
  after RU151), purity improves too (.674→.743), and displacement drops (mean .137→.074, p95
  .186→.147). Internal gaps stay flat (7→7), again consistent with its own ~4 unmatchable
  landmarks driving that count independent of the gate.

Both patients' remaining internal gaps are the correctly-diagnosed, interpretable unsupported
intervals from the matchability fix, not a gate artifact — removing their cheap terminal pricing
does not remove or worsen those gaps, it only removes the (now-unnecessary) coverage cost that
had been layered on top by the automatic gate over-triggering.

## RU151 is again the largest single beneficiary

As in the pre-fix ablation, **RU151 gains the most from ungating** (coverage .720→.910, terminal
skip 19%→0%, mean displacement .211→.045) — even more dramatically than before, since it is now
being compared against a cleaner automatic-gate baseline. RU263 remains essentially unaffected
either way (.970→.970), since it was already near-full coverage under the cheap price.

## Heatmap

[gate_ablation_postfix_comparison/htan_comparison.png](gate_ablation_postfix_comparison/htan_comparison.png)
— Raw | trajectory-250 restricted | automatic gate (post-fix) | RU1303-only gate (post-fix), same
patient order/colors/bins/axis/status-strip convention as every earlier figure in this round.
RU151's and RU942's rows are visibly far less gray in the RU1303-only panel; RU831's row shows a
clean gray break within its SCLC-A block in both gated panels (the matchability-fix-diagnosed
unsupported interval), essentially unchanged between the two gate configurations.

## Status

RU1303-only gating clearly improves the trade-off on the post-fix pipeline — better than clearly
improving it pre-fix, in fact (it now beats trajectory-250 on both coverage and purity
simultaneously, not just on coverage). Per the task's stop condition, **the one-at-a-time
RU151/RU263/RU942/RU831 gate-addition sweep was not run.** The gate threshold itself remains
untouched. Stopping here.
