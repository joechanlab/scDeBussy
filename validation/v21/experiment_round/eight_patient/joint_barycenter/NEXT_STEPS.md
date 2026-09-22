# Next steps

Continuing from `b28e09d`. Primary direction only — no implementation tonight, no repeated
threshold tuning on HTAN. See `EXPERIMENT_SUMMARY.md` for how we got here and
`stability/GATE_STABILITY.md` for the evidence behind this recommendation.

## Primary next direction: replace the hard coverage gate with a probabilistic/latent coverage model

The largest-gap threshold rule (`AUTO_GATE.md`) exactly reproduces the validated RU1303-only gate
on this cohort's point estimate, but `GATE_STABILITY.md` showed the *exact* split recovers only
~47% of the time under resampling of the score's own peer-position data, and flips outright if
RU581 is dropped from the cohort. A hard 0/1 threshold on a noisy 1D score is the wrong shape of
object for what we're actually modeling. Record the concept only:

### Desired latent variable / soft membership concept

- Each patient gets a continuous, per-patient **coverage/support probability** (or an uncertainty
  interval around its coverage score), not a binary gate flag.
- Frame it as: "P(this patient's observed trajectory is a partial/truncated view of the cohort's
  full state)" — a latent variable per patient, not a derived threshold decision.
- The model should report its own uncertainty (e.g., a posterior variance or credible interval on
  each patient's coverage), so a narrow-margin case like RU581's removal shows up as *low
  confidence* rather than a silent flip.

### Pre-fit evidence it should use

Same ingredients already available and already label-free/pre-fit — no new data collection:
- the existing endpoint/coverage score (terminal-centroid nearest-position, already computed by
  `frozen_coverage`),
- the full peer-position distribution underlying that score (not just its median — the bootstrap
  in `GATE_STABILITY.md` already resamples from this; a real model should use it directly as
  the likelihood, not resample after the fact),
- `landmark_matchable` fraction, checked in `AUTO_GATE.md` and found uninformative for RU1303
  specifically in this cohort, but potentially informative in a cohort where truncation and
  landmark-support loss coincide — keep as a candidate feature, not a proven one.
- Do **not** add cell-type labels, held-out markers, or any alignment-outcome metric — the
  label-free constraint stays.

### Terminal pricing should depend probabilistically on coverage uncertainty

- Replace `price = CHEAP_PRICE if gate_on else ETA` with something like
  `price(patient) = f(P(low coverage))` — a continuous interpolation between the cheap and full
  terminal price, weighted by the model's own confidence, rather than a step function.
- A patient with a confidently-low coverage estimate (like RU1303) should land near the cheap
  price; a patient with a genuinely ambiguous estimate (like the mid-coverage cluster under
  resampling) should land somewhere in between, not be forced to a coin-flip binary decision.

### What must be preserved unchanged

- The validated warp/gap DTW-style mechanics (unchanged since step 6/8 of the journey).
- The `landmark_matchable` propagation fix (`95567df`) — this is a correctness fix, not part of
  the gate-modeling question, and must stay wired into `compile_graph` exactly as-is.
- γ=1 barycenter curvature regularization — validated independently of the gate question at 3-,
  4-, and 8-patient scale; do not re-derive or re-sweep it as part of this work.

### Validation order

1. First validate the new latent coverage model on the **same pairwise controls** already used
   throughout this debugging arc (RU1303→RU581, RU831→RU581, RU831→RU1083) — same frozen genes,
   precision, mechanics, γ=1 — checking that it reproduces the already-established pairwise
   findings (RU1303's useful truncation, RU831's interpretable unsupported interval) with
   calibrated uncertainty, before touching the full cohort.
2. Only after the pairwise checks pass, rerun the same leave-one-out / bootstrap stability
   protocol from `GATE_STABILITY.md` on the new model's outputs, to confirm it's actually more
   stable, not just differently parameterized.
3. Only then run another full 8-patient fit. Do not skip straight to an 8-patient rerun.

## Explicitly out of scope for the next session unless the above validates cleanly

- Any further tuning of the largest-gap threshold itself.
- Any change to γ, warp/gap mechanics, gene panel, or precision mode.
- Any use of cell-type labels or held-out markers in gate construction (evaluation-only, as now).
