# Joint posterior warp: design contract (Phase 1)

Grounded in the current implementation as of commit `2571455` (`src/scdebussy/align/_model.py`,
`_dp.py`, `_barycenter.py`, `_api.py`, `_project.py`, `_types.py`, `tests/test_v2_lattice.py`)
and `NEXT_JOINT_WARP_PLAN.md`. Answers are ordered as specified. Every claim about "old mode" is
about the existing, untouched code path (`LatticeModel(..., warp="off")`, the default); every
claim about "joint mode" is about the new opt-in path (`warp="on"`), implemented in this same
change as `JointGraph` / `infer_joint` / `JointPosterior`, kept in separate code paths from old
mode rather than branches inside the old functions.

## 1. What does `A[i,j]` mean now vs. in joint mode?

Old mode (unchanged, `_dp.py:infer`, `log_a`/`a`): `A[i,j]` is the posterior probability mass of
the single `(1,1)` match edge at `(i,j)` — `P(the path uses the M-edge consuming patient landmark
i and consensus state j)`. Because old mode has only `(1,1)` match edges, `A[i,j]` simultaneously
answers three questions that happen to coincide: "is `i` matched", "is `j` matched", and "what is
`i` matched to". `sum_j A[i,j] = P(i matched to something) = M_i` and `sum_i A[i,j] = P(j matched
to something) = O_j` in old mode.

Joint mode (`JointPosterior.log_a`/`a`, `_dp.py:infer_joint`): `A[i,j]` is redefined, exactly per
the plan, as the **correspondence weight**, `A_ij = sum_e q_e * w_e,ij`, i.e. the coefficient of
`delta[i,j]` in expected match energy, accumulated over every match-kind edge (`(1,1)`, `(1,2)`,
`(2,1)`) that references `(i,j)` at all, weighted by that edge's per-cell weight (`1` for `(1,1)`,
`1/2` for each of the two cells a `(1,2)` edge touches, `1` for each of the two cells a `(2,1)`
edge touches). `A[i,j]` no longer means "probability `i` is matched to `j`" — a `(2,1)` edge at
`(i,j)`/`(i+1,j)` contributes weight `1` to *both* `A[i,j]` and `A[i+1,j]`, so `sum_i A[i,:,j]` can
exceed 1 in probability terms even though it is only ever the correct energy coefficient. This is
why occupancy (`M_i`, `O_j`, below) must be a *separate* quantity from `sum_j A[i,j]` /
`sum_i A[i,j]` once warp edges exist — the plan's own resolution, confirmed here by construction:
see the `(2,1)`-edge column double-count and `(1,2)`-edge half-weight example the plan already
gives in `NEXT_JOINT_WARP_PLAN.md:22,25`.

## 2. What conservation identities hold in each mode?

Old mode (unchanged, verified by `tests/test_v2_lattice.py::test_generated_equivalence`):
`sum_j A[i,j] + insertion[i] = 1` for every `i`; `sum_i A[i,j] + deletion[j] = 1` for every `j`;
`expected_matches + expected_insertions = n`; `expected_matches + expected_deletions = t`. These
hold because in old mode "the edge consuming row `i`" and "the edge consuming column `j`" are each
exactly one edge per path, and that edge is either the `(1,1)` match (counted in `A`) or a
gap/terminal edge (counted in insertion/deletion).

Joint mode (new, verified in `tests/test_v2_joint_warp.py`):
- `sum_j A[i,j] + patient_unmatched[i] = 1` for every `i` — **survives**, because exactly one edge
  per path is still "responsible" for consuming patient index `i` (the DAG's `i`-coordinate still
  advances monotonically over `{0,...,n}` and every increment belongs to exactly one edge, whether
  that edge is `(1,1)`, half of a `(1,2)` (contributes weight `1`, not `1/2`, to the *occupancy*
  sum, since occupancy is an indicator not a correspondence weight), or one side of a `(2,1)`).
  Note the left-hand side does NOT read `sum_j A[i,j] + patient_unmatched[i]`... it does; the
  reason it survives is that `patient_unmatched[i]` is occupancy-based (`1 - M_i`), not derived
  from `A`, and `M_i` is *defined* to equal `sum_j A[i,j]` in joint mode too (see Q3) — this only
  holds because along the patient axis every match-kind edge gives each patient index it touches a
  set of `A`-weights that sum to exactly 1 (one `(1,1)` cell weight 1; one `(1,2)` edge gives its
  single patient index two half-weight cells, `0.5+0.5=1`; one `(2,1)` edge gives each of its two
  patient indices one weight-1 cell). So `sum_j A[i,j]` is well-defined as "total weight touching
  row `i`" and equals the occupancy indicator's expectation by direct algebra. This is a genuine,
  checked fact, not an assumption — see Q3 and the `test_row_identity_by_construction` test.
- `sum_i A[i,j] + deletion[j] = 1` — **does not survive** (plan's own point,
  `NEXT_JOINT_WARP_PLAN.md:25`), because along the consensus axis the algebra above breaks: a
  `(2,1)` edge gives its single consensus index `j` two full weight-1 cells (`A[i,j]` and
  `A[i+1,j]` each get `+1`), so `sum_i A[:,j]` can be `2` at a `j` with occupancy `1`; a `(1,2)`
  edge gives each of its two consensus indices weight `0.5` alone (each `<1`), so `sum_i A[:,j]`
  can undercount occupancy there too. Replaced by `O_j + consensus_unmatched[j] = 1`, where `O_j`
  is the occupancy indicator's expectation directly (edge-indicator sum, not derived from `A`).
- Count identities: `E[matched patient states] + E[unmatched patient states] = n`,
  `E[matched consensus states] + E[unmatched consensus states] = t`, where a `(2,1)` edge counts
  as consuming two patient states and one consensus state, a `(1,2)` edge the reverse. Verified as
  `sum(M_i) + sum(U_i) == n` and, independently, as
  `expected_match_11 + expected_match_12 + 2*expected_match_21 == sum(M_i)` (and the symmetric
  consensus-side identity `expected_match_11 + 2*expected_match_12 + expected_match_21 ==
  sum(O_j)`), both checked in the new test module.

## 3. Insertion/deletion vs. occupancy

Old mode: `insertion[i] = 1 - sum_j A[i,j]` is computed as the complement of the match-marginal
row sum (`_dp.py:119`), not by summing gap-edge probabilities directly. This is valid *only*
because in old mode there is no other way to consume row `i` than "matched" or "one gap edge", and
the row-sum-of-`A` already equals `P(i matched)` exactly.

Joint mode: `patient_unmatched[i]` is NOT derived from `A` at all. It is computed the same way `A`
is computed today — as a direct aggregation of edge-posterior mass `q_e` over edges annotated as
"this edge leaves patient index `i` unmatched", split into `patient_unmatched_internal[i]` (I-type
internal gap edges) and `patient_unmatched_terminal[i]` (open-mode leading-skip / trailing-suffix
/ all-gap edges), so `patient_unmatched[i] = patient_unmatched_internal[i] +
patient_unmatched_terminal[i]`. `M_i` (`patient_occupancy[i]`) is likewise a direct edge-indicator
aggregation (see Q1/Q2: it happens to equal `sum_j A[i,j]` by construction, but is not *defined*
via `A`). The consensus side is symmetric: `O_j` (`consensus_occupancy[j]`) and
`consensus_unmatched_internal/terminal[j]`.

## 4. How does the barycenter M-step consume these?

`_barycenter.py:sufficient_statistics` and `expected_match_energy` consume `posterior.log_a`/`.a`
only — i.e. the correspondence-weight matrix `A`, never `insertion`/`deletion`/occupancy. The
M-step's closed form (`B[j,g] = sum_i A[i,j]*precision[i,g]*x[i,g] / sum_i A[i,j]*precision[i,g]`,
`_barycenter.py:75-83`) is exactly the weighted mean the plan requires, and it is correct for a
`(1,2)`/`(2,1)`-bearing `A` with **no code change**, because `A[i,j]` is already defined so that
"the coefficient of `delta[i,j]` in expected match energy" is the right quantity to weight
`x[i,:]` by when re-estimating `B[j,:]` (that is exactly what `expected_match_energy`,
`_barycenter.py:87`, computes: `sum p.a * costs(lm, barycenter)`, and `costs` computes `delta`
itself). This is a claim I verified with a dedicated test
(`test_barycenter_consumes_joint_A_unchanged` in the new module) rather than assumed: I built a
tiny joint-mode fit-like scenario (fixed `barycenter`, one patient with a forced `(1,2)` MAP path),
called `sufficient_statistics`/`expected_match_energy` unmodified against the `JointPosterior`'s
`.a`, and confirmed the M-step reproduces the same weighted mean as an equivalent old-mode
posterior with the `(1,2)` edge manually split into two independent `(1,1)`-style half-weight
correspondences feeding the same rows — i.e. `sufficient_statistics` cannot tell the difference
between "one `(1,2)` edge with two half-weight cells" and "two independent half-weight `(1,1)`-
style contributions", which is exactly the property the plan relies on. No change to
`_barycenter.py` was needed or made.

## 5. What assumptions rely on 1:1 matching, and what breaks with `(1,2)`/`(2,1)`?

- **Old-mode `insertion`/`deletion` as `A`-complements** (Q3): breaks, because it assumes each row
  and column has at most one match edge contributing to it, all of weight 1. Resolved by NOT
  reusing these fields in joint mode; new named occupancy fields replace them.
- **Old-mode `count_probability`** (`_dp.py:_count_messages`, step size always `0` or `1`): breaks,
  because warp edges advance the patient-count by 2 (`(2,1)`) while leaving one side matched but
  not "one landmark, one increment". Resolved by NOT extending `_count_messages`/`count_probability`
  into joint mode at all; the task's actual requirement is *expected* transition counts
  (`expected_match_11/12/21`), which are simple `ep`-sums per edge kind and need no DP kernel
  change (see Q2, computed as a pure NumPy post-pass over `ep` and a `kind` edge array).
- **`low_match_probability`/`no_match_probability`** (`_dp.py:90-97`, defined off
  `count_probability`): not extended to joint mode; no joint-mode equivalent is added in this
  phase because it would require deciding whether "low support" means low patient occupancy or low
  consensus occupancy, which the plan explicitly leaves open ("explicitly define whether low
  support uses matched patient or consensus occupancy", `NEXT_JOINT_WARP_PLAN.md:27`) and defers
  past the wiring into `fit()`, which this phase does not do. Left as a documented gap, not guessed.
- **Projection's `position_given_landmark = A[i,:]/M_i`** (`_project.py:66-69`): the *formula* is
  unaffected (Q7) but its interpretation changes: in old mode `A[i,:]/M_i` is a genuine conditional
  distribution over "which single consensus state is `i` matched to". In joint mode it remains a
  valid probability distribution (each row of `A` still sums to `M_i`, per Q2's row identity) but
  a `(1,2)` edge's mass sits at two adjacent `j`'s that are *jointly* explained by one warp choice,
  not two independent posterior alternatives — the conditional mean can now land at the true
  midpoint of a stretch rather than snapping to the better of two candidates, which is the
  intended behavior change, not a bug.
- **`AlignmentResult.write`'s posterior export** (`_types.py:151-164`): assumes `Posterior` fields
  (`insertion`, `deletion`, `match_count_probability`, ...). Not touched, since joint mode is not
  wired into `fit()`/`AlignmentResult` in this phase (Q7).

## 6. Is a patient landmark associated with one consensus location, multiple, an interval, or transition-level mass?

Transition-level mass, exactly as the plan implies but stated precisely here: a patient landmark
`i` is associated with a *distribution over match-kind edges* that touch it, each edge contributing
a correspondence row of one or two weighted `(j, w)` cells (`(1,1)`: one cell, weight 1; `(1,2)`:
one cell, weight... no — `i` in a `(1,2)` edge gets exactly one row-cell contribution of weight 1
total, split 0.5/0.5 across `j` and `j+1`, since `i` itself is not split, only its two consensus
partners are; `(2,1)`: `i` gets one cell, weight 1, at the shared `j`, and its partner `i+1` gets
its own separate weight-1 cell at the same `j`). So "one consensus location" (old mode) generalizes
to "a weighted set of at most two adjacent consensus locations" per `i`, never an open interval and
never more than two cells — the lattice's monotone step structure caps it there by construction
(only `(1,2)` steps exist, not `(1,3)` etc., per the plan's minimal contract). This is exactly
`A[i,:]` normalized by `M_i` (Q5's conditional distribution).

## 7. What should cell projection use?

Per the plan (`NEXT_JOINT_WARP_PLAN.md:29`): conditional position should still use `A[i,:]/M_i`
(unchanged formula — Q5/Q6 show this remains a valid, if now occasionally two-cell, distribution);
unconditional projection support (`P(cell projected)`) and coverage should use `M_i` = patient
occupancy, not `sum_j A[i,j]`, even though in joint mode these two happen to be numerically
identical (Q2) — the plan wants the *semantic* source of truth to be occupancy so that a future
consensus-side coverage metric (which does need `O_j`, not `sum_i A[i,j]`, per Q2) reads
consistently. **Scope decision for this phase**: `_project.py`/`_types.py` are NOT rewired to
branch on `JointPosterior` in this commit, because `_api.py:fit()` is explicitly out of scope
(task instructions: "do NOT wire this into `fit()`'s real E/M loop yet") and no `AlignmentResult`
in this phase can ever contain a `JointPosterior` — `project()`/`summarize()` only ever see old
`Posterior` objects, so branching code there would be untestable dead code. Instead I added a
narrow, tested helper, `occupancy_support(posterior)` in `_project.py`, that accepts either
`Posterior` or `JointPosterior` and returns `(matched_occupancy, unmatched)` using `sum_j A[i,j]`
for the old type and `.patient_occupancy`/`.patient_unmatched` for the new type, so that Phase 3's
real wiring has a single call site to swap in rather than duplicating the branch at every use site
in `project()`. This is the one deliberate deferral in this phase; it is explicitly not a claim
that `_project.py` is "done" for joint mode.

## 8. What should coverage/support/rejection mean under the expanded model?

- **Support (per patient landmark)**: `M_i` (patient occupancy), not `sum_j A[i,j]` as a *textual*
  matter (they coincide numerically per Q2, but see Q7 for why the semantic source matters).
- **Coverage (per patient, aggregate)**: unchanged formula shape
  (`exp(logsumexp(log_retained))/n_cells`, `_project.py:155`) but should be read as "expected
  fraction of a patient's cells whose local pseudotime falls in a region with nonzero patient
  occupancy", inheriting `M_i` transitively through `position_given_landmark`'s row-normalization.
- **Rejection / "unsupported"**: should be based on `patient_unmatched[i]` at the interpolation
  points a cell falls between (mirrors current `unprojected = 1 - retained mass`,
  `_project.py:86`), not on any single edge-kind count. A cell between two landmarks that are both
  entirely explained by expensive terminal unmatched mass should be flagged via
  `patient_unmatched_terminal`, not silently absorbed the way MAP-only diagnostics would.
- **Consensus-side coverage** (new in joint mode, no old-mode equivalent): should use `O_j`
  (`consensus_occupancy[j]`), not `sum_i A[i,j]`, per Q2 — this is the one place old and joint mode
  genuinely diverge in what's even the right marginal to look at, since old mode's `deletion[j]`
  IS `1 - sum_i A[i,j]` exactly, while joint mode's `consensus_unmatched[j]` is NOT `1 - sum_i
  A[i,j]` (that quantity isn't even guaranteed to be a valid probability — it can be negative when
  a `(2,1)` edge pushes `sum_i A[:,j]` above 1).

## 9. Resolved ambiguities not fully specified by the plan (engineering choices, not guesses about the math)

1. **Terminal-cost application scope**: the plan's `lambda_terminal_gap` has no old-mode analogue —
   old "open" mode terminal edges pay pure linear `eta` cost with *no* gap-open charge at all (that
   is the definition of "open" vs "global" mode, `_model.py:95-125`), and old "global" mode has no
   separate terminal concept (boundary gaps are ordinary affine-gap runs). I restricted `warp="on"`
   to `mode="open"` only (raises `ValueError` otherwise) and defined: a terminal edge pays
   `lambda_terminal_gap` once for its patient-side tail if that tail is nonempty, and again for its
   consensus-side tail if that tail is nonempty (so the all-gap edge, which has both tails nonempty
   whenever `n,t>0`, pays `2*lambda_terminal_gap`). This is the natural generalization of the
   existing affine-gap "one open charge per contiguous run" rule applied to the (at most one
   leading + one trailing) terminal run per side. Deferred: `mode="global"` + `warp="on"` combination.
2. **Reject-by-construction vs. calibrated prior against all-terminal paths**: implemented as a
   hard validation, `lambda_terminal_gap > 0` required whenever `warp="on"` (`_model.py`
   `__post_init__`). This does not forbid the all-terminal path from existing in the lattice (it
   must remain admissible for genuinely unalignable patients), but guarantees it is never free,
   resolving the plan's "impose a calibrated prior against degenerate all-terminal paths" clause
   without structurally banning a path that must sometimes be the correct MAP choice.
3. **`(1,2)`/`(2,1)` as the very first or very last match**: old-mode's "open" terminal handling
   bundles the leading skip cost into the first match edge itself (`_model.py:100`) rather than
   using a generic terminal-run edge; I extended this same bundling pattern to warp edges (new
   edges directly from source consuming a skip-prefix plus a `(1,2)`/`(2,1)` first match), rather
   than only allowing warp edges in the graph's interior. This was necessary for Case B/D tests
   where the cheapest path legitimately starts or ends with a warp step.
