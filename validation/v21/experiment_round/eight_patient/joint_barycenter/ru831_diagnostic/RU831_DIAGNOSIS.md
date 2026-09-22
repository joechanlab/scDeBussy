# RU831 pairwise diagnostic: why does it plateau/gap even with terminal gating off?

Follow-up to [GATE_ABLATION.md](../GATE_ABLATION.md), which found RU831's internal-gap
pathology is not caused by the coverage-gate over-triggering (it persists — mildly worsens —
when RU831 is given full terminal price). This pilot isolates RU831 in a focused pairwise
diagnostic against RU581 (primary) and RU1083 (second reference), using the exact frozen
mechanics of the current joint model: `match_cost` (unit-precision MSE over the frozen
trajectory-250 genes), `compile_graph`/`infer_graph` (the DTW-style warp/gap graph with its
built-in run-length regularization, BETA=5, RHO=0.02), imported directly from
`three_patient_joint.py` — not the older `scdebussy.align._dp`/`_model` API used in the original
RU1303–RU581 diagnostic. RU831 is given the **full** terminal price (ETA), matching the
gate-ablation state where its pathology still appears. No parameter is retuned. Code:
[ru831_pairwise.py](ru831_pairwise.py); full numbers: [ru831_diagnostic.json](ru831_diagnostic.json).

## Headline finding: RU831 has a genuine landmark-support hole, and the graph is reacting to it correctly

`RU831 matchable landmarks: 80/100` — the frozen trajectory-250 cache flags **20 of RU831's 100
landmarks as unmatchable** (indices 15, and a contiguous block 70–88), presumably from
insufficient real cells at that local-pseudotime range during landmark construction. Critically:
**none of the joint-model scripts in this round check or respect this flag.** `match_cost` is
computed directly from `landmark_expression` for every landmark regardless of `matchable`, and
the earlier pairwise diagnostic (`run_pairwise.py:108`) explicitly asserted both its patients
were 100%-matchable before proceeding — RU1303/RU581/RU1083/RU1444, the four patients validated
in every earlier pilot, all happen to be fully matchable, so this gap in the pipeline never
surfaced until RU831.

## Cost-surface check: is the problem concentrated in the unmatchable landmarks specifically?

Yes, sharply so. Splitting the best-achievable match cost (min over all reference positions) by
whether the landmark is flagged matchable:

| Reference | Window | Matchable landmarks: best-cost mean | Unmatchable landmarks: best-cost mean | Overall RU831 average |
|---|---|---:|---:|---:|
| RU581 | [65, 94] (n=30, 19 unmatchable) | **2.80** (n=11) | **47.90** (n=19) | 9.80 |
| RU1083 | [83, 93] (n=11, 6 unmatchable) | **1.83** (n=5) | **148.82** (n=6) | 9.62 |

Where RU831 has *real* landmark data, its best achievable match cost is actually **below** its
own overall average (2.80–1.83 vs ~9.6–9.8) — there is no sign of a genuine biological mismatch
or cost-surface problem among reliably-supported landmarks. Where the landmark is flagged
unmatchable, the best achievable cost explodes to **5–15x the overall average** (47.9 and 148.8),
which is exactly the enormous near-black cost spike visible in
[cost_matrix_paths.png](cost_matrix_paths.png) around RU831 pseudotime ≈0.87. The frozen
DP graph pays the (comparatively tiny, ≈0.517-per-landmark) terminal/gap price instead of these
wildly expensive matches — this is the graph behaving exactly as designed given its cost input;
the warp/gap mechanics are not malfunctioning.

## Internal gap / plateau locations

| Reference | Internal patient gaps | Gap landmark range(s) | Longest exact MAP plateau |
|---|---:|---|---|
| RU581 | 26 | **[67, 92]** (one contiguous 26-landmark block) | none (curve is broken/NaN there, not a repeated-value plateau) |
| RU1083 | 8 | [67, 69], [87, 91] | 3 landmarks at [36, 38] → value 0.424 (a real, small compression, not in the unmatchable block) |

The RU581 pairwise gap (26 landmarks) is larger than what appeared in the full 8-patient joint
fit (6 internal gaps) or the gate ablation (7) — expected, since the 8-/multi-patient barycenter
gives RU831 more (and better-supported) target positions to match against than a single
reference does; the pairwise case isolates the worst-case version of the same underlying problem.
The RU1083 case additionally shows a small 3-landmark compression at [36,38] (value 0.424) that
is *not* inside the unmatchable block — a minor, separate, much smaller effect not further
investigated here.

## Is a smooth alternative reachable, and is it actually better?

No — forcing a smooth path is reachable but makes the result *worse*, not better. An
unconstrained open-begin/open-end DTW path (same BAND=60 reachability check used in the original
RU1303–RU581 diagnostic) is technically reachable through the whole window, but it must match
*every* landmark including the unmatchable ones, at a mean cost of 31.5 (RU581) / 101.0 (RU1083)
per landmark in the window — statistically indistinguishable from `window_best_achievable_cost`
(31.4 / 100.0), confirming this is close to the *best possible* smooth path, and it is still far
worse than what gapping achieves. The
[comparison heatmap](comparison_heatmap.png) (Raw | joint-model pairwise mapping | open DTW band
60 "best smooth alternative") shows this concretely: the DTW-forced smooth alternative compresses
RU831 into a much narrower, sparser occupied range (a thin sliver of SCLC-N rather than the
joint model's wider, better-supported SCLC-A block) — smoothness here is achieved only by
accepting bad matches, not by finding a hidden good path the graph missed.

## MAP–posterior disagreement and held-out markers

Both stay unremarkable and are not evidence of a separate problem: MAP–posterior mean absolute
difference is 0.0135 (RU581) / 0.0171 (RU1083) — smaller than the 8-patient joint fit's RU263
(0.033), the largest in that round. Held-out 7-gene marker MSE (evaluated only at landmarks the
graph actually mapped) is 1.58 (RU581) / 1.39 (RU1083) — squarely in the same range as the other
gated/moderate-coverage patients in the full 8-patient fit (0.65–1.29), not an outlier. Where
RU831 *is* matched, both diagnostics say the alignment is trustworthy; the pathology is confined
to the region with no real data to align.

## Conclusion: which of the four categories is RU831's problem?

**Primarily (3) landmark support/preprocessing**, with the transition/gap mechanics (2) reacting
*correctly* to that input rather than malfunctioning, and no meaningful evidence for (1) a bad
cost surface or (4) genuinely absent biology *among the landmarks that have real data*:

1. Expression/cost surface — not the primary issue: matchable landmarks cost *less* than RU831's
   own average, both with RU581 and RU1083 as reference.
2. Transition/gap mechanics — working as intended: gapping is the objectively cheaper and
   biologically better choice given the cost input (confirmed by the DTW reachability check
   producing a worse, not better, result at comparable-to-optimal cost).
3. **Landmark support/preprocessing — the primary driver.** 19–20 of RU831's 100 trajectory-250
   landmarks are flagged `matchable=False` and were never excluded from `match_cost` anywhere in
   this round's joint-model scripts; their imputed/unreliable expression produces match costs
   5–15x the patient's own average, which the graph correctly avoids by gapping.
4. Genuinely missing/different biology — not distinguishable from (3) with current data (RU831
   may simply have too few cells in that pseudotime range to ever produce a reliable landmark,
   which is itself a data-support fact, not a signal that the underlying biology diverges), but
   there is no positive evidence for a biological mismatch specifically, since the matchable
   landmarks nearby align cheaply and consistently with both references.

## Recommendation

**No parameter (γ, gate threshold, graph costs, terminal prices) should be tuned in response to
this diagnostic** — doing so would not fix a support gap. The concrete, low-risk fix this
diagnosis points to is a **code-level one, not a parameter tune**: propagate the frozen
`landmark_matchable` flag into `match_cost`/`compile_graph` (e.g. force-infinite cost or a
forced-gap edge at unmatchable landmarks) across all of `three_patient_joint.py`/
`four_patient_joint.py`/`eight_patient_joint.py`/`gate_ablation.py`, rather than relying on the
DP graph to "discover" this indirectly through inflated imputed-expression cost. This is a
structural/plumbing fix, not a tuning change, and is out of scope to implement in this
diagnostic-only pilot per the task's instructions — flagging it as the next concrete step before
any further cohort-level tuning.

Stopping after this diagnosis; no cohort model changes made.
