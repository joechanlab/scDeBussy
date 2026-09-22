# Label-free, data-driven coverage gate

Tests whether the manually-selected RU1303-only gate (validated post-matchability-fix in
[GATE_ABLATION_POSTFIX.md](GATE_ABLATION_POSTFIX.md)) can be recovered from pre-fit trajectory
information alone, without naming RU1303. Every other mechanic is frozen and reused unchanged:
trajectory-250 genes, unit precision, matchability handling, DTW-style warp/gap graph, γ=1,
terminal price values. Only the gate *decision rule* changes. Code:
[auto_gate.py](auto_gate.py).

## The rule

The endpoint/coverage score itself (`frozen_coverage`: each patient's last-10-landmark expression
centroid, matched against the median nearest position in every *other* patient's raw trajectory)
was already label-free and pre-fit — it uses no cell-type label, no held-out marker, and no fit
quality metric, and is unchanged from every earlier pilot. What changed before (and over-triggered
at 8-patient scale) was the fixed **margin** used to threshold it (`score ≤ max − 0.10`).

The simplest data-driven replacement: sort the 8 scores, find the single **largest gap** between
consecutive values, and gate on (cheap terminal price) everything below the midpoint of that gap
— an unsupervised 1D two-cluster split, not a fixed margin, and not a patient name.

**Candidate signal checked but not used:** fraction of matchable landmarks in the last 10 (or 20)
positions, as a direct test of "does this patient's *observed, well-supported* trajectory reach
the cohort's late state." Computed for all 8 patients, this signal is **uninformative here**:
every patient, including RU1303, has late-10 matchable fraction = 10/10 (RU831's 20 unmatchable
landmarks are concentrated mid-trajectory at indices 15 and 70–88, not late; RU942's 4 are at
indices 49–52). RU1303's truncation is not a landmark-support gap at all — its late landmarks are
fully supported, they just don't resemble any other patient's late state. This is reported as a
checked, ruled-out ingredient, not silently dropped.

## Gate decision

| Patient | Coverage score | Rank | Gap to next patient | Gate |
|---|---:|---|---:|---|
| RU1303 | .404 | 1 (lowest) | **.222** (largest gap in the cohort) | **on** |
| RU831 | .626 | 2 | .030 | off |
| RU263 | .657 | 3 | .010 | off |
| RU942 | .667 | 4 | .040 | off |
| RU151 | .707 | 5 | .172 (2nd-largest gap) | off |
| RU581 | .879 | 6 | .061 | off |
| RU1444 | .939 | 7 | .010 | off |
| RU1083 | .949 | 8 (highest) | — | off |

Threshold: .515 (midpoint of the .404–.626 gap). The largest gap (.222) is not a close call —
it's 1.3x the second-largest gap (.172, between RU151 and RU581, the boundary the old fixed-.10
margin implicitly fell inside of, which is exactly why it over-triggered). Full diagnostics:
[auto_gate/frozen_gate.json](auto_gate/frozen_gate.json).

**Result: `gate_on = {RU1303: True, all seven others: False}` — an exact match to the manually
selected RU1303-only gate, with no patient identity used anywhere in the rule.**

## Fit results: bit-identical to the RU1303-only gate, as expected

Since the automatic rule produces the identical gate assignment, the resulting fit is
numerically identical (confirmed, not just similar) to `GATE_ABLATION_POSTFIX.md`'s RU1303-only
result:

| Method | Coverage | Purity | Held-out marker concordance | RU1303/SCLC-N overlap | Mean disp. | p95 disp. |
|---|---:|---:|---:|---:|---:|---:|
| raw `s_local` | .865 | .772 | .421 | .531 | 0 | 0 |
| trajectory-250 restricted | .850 | .785 | .419 | .132 | .009 | .111 |
| RU1303-only gate (manual) | .855 | .786 | .521 | .000 | .0596 | .1659 |
| **automatic data-driven gate** | **.855** | **.786** | **.521** | **.000** | **.0596** | **.1659** |

Full tables: [auto_gate_comparison/metrics.csv](auto_gate_comparison/metrics.csv),
[auto_gate_heldout/heldout_summary.json](auto_gate_heldout/heldout_summary.json). Figure:
[auto_gate_comparison/htan_comparison.png](auto_gate_comparison/htan_comparison.png) (Raw |
trajectory-250 restricted | RU1303-only gate | automatic data-driven gate) — the bottom two
panels are pixel-identical, as expected from an identical gate assignment.

## Per-patient (final phase, γ=1, automatic gate)

| Patient | Coverage score | Gate | Coverage | Purity | Terminal skip | Internal gaps | MAP–post Δ |
|---|---:|---|---:|---:|---:|---:|---:|
| RU1083 | .949 | off | .940 | .777 | 0% | 0 | .0132 |
| RU1444 | .939 | off | .980 | .795 | 0% | 0 | .0082 |
| RU581 | .879 | off | .990 | .866 | 0% | 0 | .0211 |
| RU1303 | .404 | **on** | .620 | .868 | 30% | 0 | .0111 |
| RU151 | .707 | off | .910 | .566 | 0% | 0 | .0395 |
| RU263 | .657 | off | .970 | .679 | 0% | 0 | .0228 |
| RU942 | .667 | off | .840 | .743 | 0% | 7 | .0192 |
| RU831 | .626 | off | .590 | .857 | 0% | 26 | .0182 |

RU831 and RU942 keep their internal gaps (26, 7) even though neither receives cheap terminal
pricing under this rule — exactly the same pattern established in
`GATE_ABLATION_POSTFIX.md`: those gaps come from the matchability fix correctly reflecting each
patient's own landmark-support hole, not from gate configuration.

## Main question: can a simple pre-fit, label-free rule recover the RU1303-only behavior without manually naming RU1303?

**Yes, exactly.** The largest-gap split of the existing, already-label-free endpoint/coverage
score reproduces the manually-selected RU1303-only gate assignment precisely, using only pre-fit
trajectory information (no cell-type label, no held-out marker, no final alignment performance).
The rule is simple (a single unsupervised 1D two-cluster split, no combination of multiple
signals was needed once the late-matchable-fraction signal was checked and found uninformative
for this cohort) and the gap it selects on is decisive, not marginal (.222 vs the next-largest
.172).

## Status

Stopping after this test, per the task's instruction not to tune the rule repeatedly on the same
cohort. The rule was applied exactly once, with no adjustment. It has not been tested on any
other cohort or patient count, so its generalization beyond this specific 8-patient dataset is
unverified — a caveat for any future use, not a finding from this pilot.
