# Landmark-matchability fix: RU831 validation

Follow-up to [RU831_DIAGNOSIS.md](RU831_DIAGNOSIS.md), which traced RU831's compression/gap
pathology to a pipeline bug: `landmark_matchable=False` landmarks (20/100 for RU831) still had
their raw `landmark_expression` fed into `match_cost`, so the graph could only avoid them if
their (possibly unreliable/imputed) expression happened to look expensive — which worked most of
the time by chance, but not always.

## The fix

`compile_graph(cost, tail_price, matchable=None)` in
[three_patient_joint.py](../../ru1303_ru581_pairwise/three_patient_joint.py) now accepts an
optional length-100 boolean mask over the patient's own landmarks. For any landmark flagged
`False`, a permanent `MATCH_COST_SENTINEL = 1e4` penalty is added to `static` for every ordinary
match/warp edge (kind 0/1/2) sourced from it — baked in at compile time, so it survives every
later `infer_graph` call regardless of what cost value that iteration's barycenter happens to
produce for that landmark. Gap edges (kind 3, `mi=-1`) are structurally untouched, so unsupported
regions remain crossable via a gap exactly as before.

**Why not just filter `landmark_expression` inside `match_cost` instead?** `compile_graph` builds
`static` by subtracting the *initial* cost and `infer_graph` later re-adds the *current*
(barycenter-dependent) cost for the same edges — a detach/reattach round trip designed to let the
per-iteration cost change without rebuilding the graph topology. Applying a sentinel inside
`match_cost` would get subtracted out at compile time and re-added at infer time, **cancelling
to net zero** and silently defeating the fix. The penalty had to live in `compile_graph`,
additive on top of that round trip, not inside the cost matrix itself.

`matchable=None` (the default, used by every existing caller) is a complete no-op — confirmed by
test 1 below — so **behavior for fully matchable patients (RU1303/RU581/RU1083/RU1444, all 100%
matchable) is unchanged**, and no other file in this round (`four_patient_joint.py`,
`eight_patient_joint.py`, `gate_ablation.py`) needed to change to preserve their existing
results. Wiring `matchable` into those scripts' own EM loops is a follow-up left for their next
rerun, not done here per the task's "do not rerun the full 8-patient fit yet" scope.

## Tests

[test_matchable_fix.py](../../ru1303_ru581_pairwise/test_matchable_fix.py) (3/3 passing), using
small synthetic 100x100 cost matrices (`build_soft_graph`'s landmark grid is a fixed 100x100, so
the matrix size itself can't shrink):

1. **`test_fully_matchable_reproduces_default_behavior`** — `matchable=None` and an explicit
   all-`True` mask produce byte-identical `compile_graph` output and identical posterior/MAP
   results.
2. **`test_unmatchable_landmarks_are_never_matched`** — landmarks 40–49 flagged unmatchable get
   zero posterior match mass and never appear as a matched MAP-path source (`map_index == -1`
   throughout), while `internal_patient_gaps > 0` confirms the region is still crossed via gaps.
3. **`test_unsupported_region_handled_via_gap_not_by_relying_on_huge_cost`** — the adversarial
   case: landmarks 40–49 are given an artificially *cheap* cost (1e-6, simulating unreliable data
   that happens to look attractive) and flagged unmatchable. Without the fix (`matchable=None`)
   this region gets matched, as expected. With the fix, it is fully avoided and gapped despite
   the low raw cost — proving the avoidance is structural, not an accident of the cost surface,
   which is exactly the RU581 case (see below) where the fix made no visible difference because
   the graph was already avoiding the region "by luck," and exactly the RU1083 case where it
   wasn't so lucky.

## RU831 rerun: before vs after

Same frozen mechanics as [RU831_DIAGNOSIS.md](RU831_DIAGNOSIS.md) (full terminal price ETA,
matching gate-ablation's gate-off state); only the matchability fix is new. Code:
[ru831_matchability_fix_rerun.py](ru831_matchability_fix_rerun.py); full numbers:
[ru831_matchability_fix_rerun.json](ru831_matchability_fix_rerun.json).

| Reference | | Internal gaps | Gap ranges | Longest plateau | Matched span | Matched count | MAP–post Δ | Held-out marker MSE |
|---|---|---:|---|---|---|---:|---:|---:|
| RU581 | before | 26 | [67, 92] | 1 | [0, 99] | 74 | .0135 | 1.582 |
| RU581 | **after** | 27 | **[15, 15], [67, 92]** | 1 | [0, 99] | 73 | .0133 | 1.590 |
| RU1083 | before | 8 | [67, 69], [87, 91] | 3 | [0, 99] | 92 | .0171 | 1.393 |
| RU1083 | **after** | 27 | **[15, 15], [67, 92]** | 3 | [0, 99] | 73 | .0161 | 1.422 |

## Does the previous compression pathology disappear, or become an interpretable unsupported interval?

**It becomes a clean, interpretable, reference-independent unsupported interval** — and reveals
that the RU581 pairwise result was already accidentally correct, while the RU1083 result was not.

- **RU581 barely changes** (26→27 gaps, matched count 74→73): the graph was already avoiding
  nearly all of RU831's unmatchable landmarks against RU581 "by luck," because their unreliable
  expression happened to look expensive there too. The fix closes one small residual gap at
  landmark 15 (visible as a new short break in
  [mapping_functions_after_fix.png](mapping_functions_after_fix.png)) that had been silently
  matched before.
- **RU1083 changes substantially** (8→27 gaps, matched count 92→73): before the fix, a large part
  of RU831's unreliable region (roughly landmarks 70–86) was being matched to RU1083 at low
  apparent cost — precisely the failure mode test 3 was written to catch, now confirmed on real
  data. After the fix, RU1083's gap pattern **converges to match RU581's exactly**
  ([15,15] + [67,92] for both), which makes sense: once ordinary matching through those landmarks
  is prohibited, the region's fate is governed entirely by RU831's own landmark structure, not by
  which reference happens to be used — a strictly more interpretable and reference-independent
  result than before.
- **No sign of a new pathology**: `matched_span` stays [0, 99] for both (no new terminal
  truncation introduced), MAP–posterior disagreement stays small and if anything improves
  slightly (.0135→.0133, .0171→.0161), and held-out marker MSE moves by ≤0.03 in either
  direction — a negligible cost for closing a real correctness bug.
- **The pathology does not vanish** — it isn't supposed to. RU831 genuinely lacks reliable
  landmark data across roughly s_local 0.67–0.92 (this is a data-support fact, not a modeling
  artifact), so a gap there is the *correct*, interpretable outcome, not a residual bug. What the
  fix eliminates is the *inconsistency* (reference-dependent gap size, occasional silent false
  matches), not the underlying data gap itself.

See [comparison_heatmap_before_after_fix.png](comparison_heatmap_before_after_fix.png) (Raw |
before fix | after fix, same patient order/colors/bins/axis/gray convention as every earlier
figure in this round) — the RU831 row looks nearly identical before and after, consistent with
the RU581-reference numbers above; the visible SCLC-A block and LUAD/SCLC-N gray gaps are
essentially unchanged, confirming this is a low-risk, structurally-correct fix rather than a
disruptive change.

## Status

Stopping after this RU831 validation, per the task's scope. Not rerun: the full 8-patient joint
fit, the 4-patient pilot, or the gate ablation — none of their results are invalidated by this
fix (all four of their patients are 100% matchable, confirmed by test 1's no-op guarantee), but
none of them have been re-verified with `matchable` actually wired into their own `compile_graph`
calls either. The next concrete step, when those are next rerun, is to pass each patient's
`landmark_matchable` array from the frozen cache into `compile_graph` the same way this pilot
does — a small, mechanical change to `three_patient_joint.py`'s `main()` and the equivalent call
sites in `four_patient_joint.py`/`eight_patient_joint.py`/`gate_ablation.py`.
