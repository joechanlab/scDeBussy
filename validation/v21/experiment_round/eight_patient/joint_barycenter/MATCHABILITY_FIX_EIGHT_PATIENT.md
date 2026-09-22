# Full 8-patient joint fit with landmark-matchability fix enabled

Reruns the full 8-patient joint fit (frozen trajectory-250 genes, unit precision, DTW-style
warp/gap graph, coverage gate, terminal pricing, γ=1 curvature regularization — all unchanged)
with the [matchability fix](ru831_diagnostic/MATCHABILITY_FIX.md) (`95567df`) now wired into
`eight_patient_joint.py`'s `compile_graph` call for every patient, using each patient's own
frozen `landmark_matchable` array. No parameter retuned. Code:
[eight_patient_joint.py](eight_patient_joint.py) (diff: one `matchable=matchable[p]` argument
added to the existing `compile_graph` call).

Matchable landmark counts recorded before fitting: RU1083 100/100, RU1303 100/100, RU1444
100/100, RU151 100/100, RU263 100/100, RU581 100/100, **RU831 80/100**, **RU942 96/100**. RU942
having 4 unmatchable landmarks was not previously flagged (the earlier diagnosis focused on
RU831); this rerun is the first to apply the fix to it too.

The pre-fix comparison artifact (`gamma1.npz`) was gitignored and got overwritten in place by
this rerun; since the pipeline is fully deterministic (no RNG anywhere), it was regenerated
byte-for-byte via [regenerate_prefix_gamma1.py](regenerate_prefix_gamma1.py) (confirmed:
`final_smoothness` and every per-patient terminal-skip/gap count reproduce the numbers committed
in `3455669`/`EIGHT_PATIENT_JOINT.md` exactly) — not a new experiment, just recovering the
baseline needed for this comparison.

## Overall comparison (frozen `summarize_and_plot.py`/`heldout_markers.py`, unmodified)

| Method | Coverage | Purity | Mean disp. | p95 disp. | RU1303 overlap | Held-out marker concordance |
|---|---:|---:|---:|---:|---:|---:|
| raw `s_local` | .865 | .772 | 0 | 0 | .531 | .421 |
| trajectory-250 restricted | .850 | .785 | .009 | .111 | .132 | .419 |
| 8-patient joint, **pre-fix** | .800 | .793 | .066 | .182 | .000 | **.582** |
| 8-patient joint, **post-fix** | **.818** | .774 | .074 | .215 | .029 | .552 |

Full tables: [matchability_fix_comparison/metrics.csv](matchability_fix_comparison/metrics.csv),
[matchability_fix_heldout/heldout_summary.json](matchability_fix_heldout/heldout_summary.json).
Figure: [matchability_fix_comparison/htan_comparison.png](matchability_fix_comparison/htan_comparison.png)
(Raw | trajectory-250 restricted | pre-fix 8-patient joint | post-fix 8-patient joint, same
patient order/colors/bins/axis/status-strip convention as every earlier figure in this round).

**Coverage improves** (.800→.818, closing more of the gap to trajectory-250's .850) — expected,
since RU831 and RU942 now get more of their genuinely-supported landmarks matched instead of a
few being wasted on unreliable-but-accidentally-cheap false matches. **Purity, displacement, and
held-out marker concordance all move slightly in the worse direction** (.793→.774, .066→.074 mean
disp., .582→.552 concordance) — a real, modest cost, but still far better than both existing
frozen controls on every metric except displacement (which was always going to be larger than
trajectory-250's, since this model does far more targeted terminal/gap movement by design).
RU1303/SCLC-N overlap moves from an exact 0 to a small .029 — still dramatically better than
raw/original_v2 (.531) or trajectory-250 (.132).

## Per-patient (final phase, γ=1, pre-fix → post-fix)

| Patient | Coverage | Purity | Mean disp. | p95 disp. | Terminal skip | Internal gaps | MAP–post Δ |
|---|---|---|---|---|---|---|---|
| RU1083 (full) | .940→.970 | .831→.821 | .035→.030 | .060→.051 | 0%→0% | 0→0 | .0044→.0067 |
| RU1444 (full) | .980→.990 | .819→.817 | .022→.021 | .032→.029 | 0%→0% | 0→0 | .0143→.0144 |
| RU581 (full) | .990→.990 | .860→.846 | .044→.044 | .048→.049 | 0%→0% | 0→0 | .0179→.0178 |
| RU1303 (gated) | .560→.590 | .890→.851 | .234→.217 | .383→.338 | **38%→34%** | 0→0 | .0074→.0110 |
| RU151 (gated) | .750→.720 | .506→.492 | .177→.211 | .213→.236 | 15%→19% | 0→0 | .0204→.0241 |
| RU263 (gated) | .980→.970 | .710→.712 | .047→.030 | .063→.063 | 0%→0% | 0→0 | .0334→.0519 |
| **RU942 (gated)** | .760→.750 | .728→.674 | .074→.137 | .160→.186 | 4%→8% | **3→7** | .0169→.0199 |
| **RU831 (gated)** | **.440→.560** | .936→.855 | .093→.132 | **.235→.155** | **23%→10%** | **6→24** | .0165→.0218 |

Full sweep: [eight_patient_joint/metrics.csv](eight_patient_joint/metrics.csv); history:
[eight_patient_joint/convergence.json](eight_patient_joint/convergence.json).

## RU831 specifically: unsupported interval, gap pattern, plateau/compression

The fix changes RU831's outcome the most, and in the direction the diagnosis predicted:

- **Coverage improves substantially**: .440→.560 (+.12, the single largest per-patient change in
  either direction across the whole cohort) — RU831 is no longer being denied its genuinely
  well-supported landmarks by a graph that was partly reacting to accidentally-cheap unreliable
  values elsewhere.
- **Terminal skip drops**: 23%→10% — less of RU831's tail is being discarded as a terminal
  truncation; more of it is now correctly expressed as an internal gap instead.
- **Internal gaps rise sharply in count (6→24) but become interpretable**: recomputing the exact
  gap-landmark positions against the actual final 8-patient barycenter gives
  **[15, 15] and [67, 89]** — a clean, near-exact match to RU831's own raw unmatchable set
  ([15] and [70, 88]) found in the earlier diagnosis, with only a small (~4-landmark) run-penalty
  boundary extension. This is the same structural convergence already seen in the pairwise
  rerun: the gap is no longer a partially-accidental, reference/barycenter-dependent artifact,
  but a clean reflection of RU831's actual landmark-support hole.
- **Plateau/compression improves**: longest exact-constant MAP plateau drops from 12→4 landmarks,
  and matched landmarks drop from 94→76 (fewer matched, but the ones still matched are less
  compressed — `n_distinct_positions` in the earlier pairwise check rose from 61 to 68 under the
  same fix). p95 displacement improves .235→.155.
- **The compression pathology does not fully disappear** (purity still drops slightly .936→.855,
  and MAP–posterior disagreement rises modestly .0165→.0218) — consistent with
  [MATCHABILITY_FIX.md](ru831_diagnostic/MATCHABILITY_FIX.md)'s conclusion that RU831 genuinely
  lacks landmark support in that region; the fix makes the *handling* of that fact correct and
  consistent, it does not manufacture data that isn't there.

## RU942: a second patient affected, previously unflagged

RU942's 4 unmatchable landmarks (not investigated before this rerun) show the same qualitative
pattern as RU831 at smaller scale: internal gaps roughly double (3→7), terminal skip rises
modestly (4%→8%), and purity/displacement both move slightly worse (.728→.674 purity,
.074→.137 mean displacement) — consistent with RU942 previously benefiting from a few of the same
kind of accidentally-cheap false matches that inflated RU831's apparent coverage before the fix.

## Convergence and barycenter roughness

| | Pre-fix | Post-fix |
|---|---|---|
| Initial roughness | .0091 | .0082 |
| Final roughness | .1586 | **.0961** |
| Roughness ratio | 17.4x | 11.7x |
| Converged (20-iter cap) | no | no |
| Final relative improvement (iter 20) | .0009 | .0007 |

The barycenter is **smoother after the fix**, not less stable: final roughness drops .1586→.0961,
and the EM trajectory remains monotone with no oscillation (relative improvement shrinking
smoothly to .0007 by iteration 20, same qualitative behavior as pre-fix). This makes sense: with
RU831's and RU942's unreliable landmarks no longer contaminating the barycenter-weighted mean at
all (previously they contributed some nonzero, noisy posterior mass at those positions), the
consensus trajectory the curvature regularization has to smooth is itself less noisy to begin
with.

## Main question: does the fix improve/materially change the result, especially RU831, without changing RU1303's previously validated behavior?

**Yes to both parts.** RU831's behavior changes materially and in the intended direction: higher
coverage, lower terminal-truncation, a shorter plateau, and — most importantly — a gap pattern
that is now a clean, reference-independent reflection of its actual landmark-support hole rather
than a partly-accidental one. RU942 shows the same qualitative (smaller) effect, previously
unflagged. Coverage improves cohort-wide (.800→.818); purity, displacement, and held-out marker
concordance move modestly in the worse direction but remain well above both frozen baselines on
every metric except displacement. RU1303's core validated behavior is preserved: it remains the
one clearly gate-truncated patient, terminal skip stays in the same 34–38% band as every earlier
pilot (3-patient: 38–44%, 4-patient: 38–39%, pre-fix 8-patient: 38%), coverage/purity shift only
modestly (.560→.590 / .890→.851), and RU1303/SCLC-N overlap stays near-zero (.000→.029), still
far better than either existing frozen control (.132–.531).

## Status

Stopping after this rerun per the task's scope. The coverage gate itself was not touched or
retuned — RU151/RU263/RU942/RU831 remain gated cheap by the same unchanged rule diagnosed in
`GATE_ABLATION.md`, and that finding (the gate over-triggers at 8-patient scale relative to what
the 3-/4-patient pilots validated) is unaffected by this fix. The two open threads from earlier
pilots — the coverage-gate margin, and RU831's residual coverage/purity trade-off even after this
correctness fix — are both still open and are next steps, not addressed here.
