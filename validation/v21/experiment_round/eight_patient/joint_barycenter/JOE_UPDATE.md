# HTAN alignment update

- **What was actually wrong in v2:** the original v2 fit barely moved cells from their raw,
  unaligned pseudotime — its purity (.728) was actually *below* just using the raw values
  (.772), because its warp/gap mechanics were too restrictive to represent real speed
  differences between patients' trajectories.

- **A real bug, now fixed:** patient RU831 has ~20% of its trajectory landmarks flagged as
  "not enough cells to trust" by the existing preprocessing — but the alignment code was
  quietly using that unreliable data anyway. For one of the two reference patients we checked,
  it got lucky and avoided it by accident; for the other, it silently produced bad matches. This
  is now fixed at the source (the model correctly skips those landmarks instead of guessing),
  and it's a correctness fix, not a tuning choice — it makes the code do what it already claimed
  to do.

- **What improved quantitatively:** the current best 8-patient fit (250-gene panel, tuned
  warp/gap mechanics, smoothed consensus averaging, and only patient RU1303 given a cheap
  "this trajectory ends early" price) beats the best previous candidate on coverage (.855 vs
  .850) *and* purity (.786 vs .785) at the same time — the first configuration this round to do
  both. It also drops RU1303's spurious overlap with other patients' end-stage cells from .132
  down to essentially zero, and roughly doubles agreement on held-out marker genes we never let
  the model see (.521 vs .419).

- **Why the automatic gate decision is still provisional:** we built a simple rule that picks
  out RU1303 as the "early-terminating" patient automatically, using only pre-fit data (no cell
  labels, no marker genes). On the numbers we have, it correctly identifies RU1303 alone. But
  when we stress-tested it — dropping one patient at a time, and resampling the underlying score
  with realistic noise — the *exact* RU1303-only decision only held up about half the time.
  Dropping just one specific other patient from the cohort flips the automatic decision entirely.
  So the mechanism is right in spirit, but the current threshold is not something we'd trust to
  generalize to a new cohort yet.

- **Tomorrow's next step:** replace the current hard yes/no gate with a model that expresses
  *how confident* we are that each patient's trajectory is a partial/early-terminating one,
  rather than a single cutoff. That confidence would then set the terminal pricing
  continuously instead of by a threshold. We'll validate this on the same small pairwise checks
  we've been using throughout before touching the full 8-patient cohort again.

**Bottom line:** the current best fit is a real, measured improvement over the previous best
candidate on coverage, purity, and — most clearly — on separating RU1303's biology from the
other patients' late-stage cells, and one genuine pipeline bug got fixed along the way. But the
automatic rule for deciding *which* patients get special handling is not yet robust enough to
call final, so we're treating today's result as the best validated fit so far, not the finished
model.
