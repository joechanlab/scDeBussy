# Session handoff: RU831 diagnostics + Git/worktree preservation

Written at the end of an HPC session, under a hard time constraint. No new
experiments, fits, or architecture changes were started or should be
inferred from this document beyond what is recorded below.

## 1. RU831 diagnostic branch

- Branch: `exp/ru831-diagnostics`
- Final SHA: `b78905e476e4a854067edf011442381dac3c4385`
- Worktree: `/tmp/scdebussy-ru831-diagnostics`, clean at time of writing.
- Pushed to `origin` (`git@github.com:joechanlab/scDeBussy.git`) this session;
  `origin/exp/ru831-diagnostics` now exists and matches local HEAD.
- Main deliverable: `validation/v21/experiment_round/eight_patient/joint_warp/RU831_DIAGNOSTICS.md`.
- **Major conclusion**: RU831's apparent "bistable" landmark-level displacement
  (0.115 moderate vs 0.246/0.889 severe regime) is a numerical-underflow
  artifact of `_project.py:summarize()` averaging over ~19-21 landmarks with
  near-zero-but-nonzero `patient_occupancy` (~6.9e-15) and zero finite `log_a`
  entries -- not evidence of genuine multimodal correspondence. Cell-level
  displacement (`project()`-based) is stable across all regimes (~0.150-0.155,
  CV=0.0023). RU831's real, well-corroborated issue is an
  internal/intermittent support gap (patient_unmatched_internal_mass 37.3,
  roughly double the comparison patients), matching a genuine density gap in
  the raw cell-type population (SCLC-A ends ~s_local 0.61, small SCLC-N
  population pinned near 1.0). A modest (5.7-6.3%) barycenter-compromise
  effect on leave-one-out exclusion was found, disproportionately affecting
  RU1303 rather than RU831's own region. No architecture change (branching /
  mixture consensus) is currently justified by this evidence; Part 6 of the
  deliverable lists what additional evidence would be needed.
- Persistent cache/artifact path (outside `/tmp`, verified present with a
  current manifest):
  `/data1/chanj3/wangm10/scDeBussy-v2/.htan_experiment_cache/exp-htan-eight-patient/joint_warp/ru831_diagnostics/`
  -- contains `ru831_diag_high_lambda_terminal_gap.npz`
  (SHA256 `3a67f95c45e0c8693fb60e6f440548a851e9dd90c06a639469638fdea86dd314`)
  and `MANIFEST.tsv`. No `.npz` files were found only under `/tmp`; nothing
  additional needed copying.
- Remote push status: **pushed**, `origin/exp/ru831-diagnostics` == local HEAD.

## 2. Diagnostic-fix branch (separate Codex task)

- Branch: `fix/joint-warp-diagnostics`
- HEAD: `6e25c8ac4151f6b9773cc1e24544102929faced1` (verified present in the
  common repository and matches the worktree's actual HEAD).
- Base: `ecf3134c561c61e3de006665b56f0c4603c3576d` (tip of
  `exp/joint-posterior-warp` at the time this and the RU831 branch both
  forked from it).
- Worktree: `/tmp/scdebussy-joint-diagnostics-fix`, clean.
- Contents (per the task description that produced it, not independently
  re-verified line-by-line in this session): corrects joint residual
  normalization, separates warp-edge and patient-state fractions, renames
  `endpoint_pileup` to `endpoint_occupancy`, adds the paired restricted-unit
  baseline to regenerated reporting, passes 611 full-suite tests, does not
  change fitting outputs.
- Remote push status: **pushed this session**,
  `origin/fix/joint-warp-diagnostics` now exists and matches local HEAD.
- **NOT merged or cherry-picked into `exp/ru831-diagnostics` or
  `exp/joint-posterior-warp`.** That reconciliation is explicitly deferred to
  a future session (see below).

## 3. Worktree / branch topology and preservation

Canonical repository: `/data1/chanj3/wangm10/scDeBussy` (git common dir
`/data1/chanj3/wangm10/scDeBussy/.git`). All worktrees below share this
common `.git`, including the ones under `/tmp` -- **none of them are
independent clones**, so their commits are already durable in the shared
object store as long as `/data1/chanj3/wangm10/scDeBussy/.git` survives,
independent of whether the `/tmp` working directories themselves survive.
This was explicitly verified (`git rev-parse --git-common-dir` from inside
`/tmp/scdebussy-ru831-diagnostics` resolves to the canonical `.git`).

Full inventory at time of writing (`git worktree list --porcelain`), all
confirmed clean (`git status --short` empty) and all now pushed to `origin`
except where noted:

| path | branch | HEAD | clean | on origin | notes |
|---|---|---|---|---|---|
| `/data1/chanj3/wangm10/scDeBussy` | `strict-soft-em` | `a8de9c4` | clean | (pre-existing, not touched) | canonical worktree, unrelated older branch; no `validation/` tree here |
| `/data1/chanj3/wangm10/scDeBussy-v2` | `refactor/v2` | `a647d69` | dirty (per session start `git status`, not re-checked here) | (pre-existing) | separate, explicitly out-of-scope workspace all session; not touched |
| `/data1/chanj3/wangm10/scDeBussy/docs/scDeBussy-current` | `current-state` | `1059257` | clean | (pre-existing) | not touched |
| `/data1/chanj3/wangm10/scDeBussy/docs/scDeBussy-old` | `old-state` | `eb2ade7` | clean | (pre-existing) | not touched |
| `/tmp/scdebussy-htan-eight` | `exp/joint-posterior-warp` | `ecf3134` | clean | **pushed this session** | Phase 3/4 base branch |
| `/tmp/scdebussy-joint-audit` | `audit/joint-posterior-warp-e774a9d` | `c7a5b6a` | clean | **pushed this session** | independent audit of Phase 1/2 |
| `/tmp/scdebussy-joint-diagnostics-fix` | `fix/joint-warp-diagnostics` | `6e25c8a` | clean | **pushed this session** | see section 2 |
| `/tmp/scdebussy-joint-phase34-audit` | `audit/joint-phase34-ecf3134` | `a9d2493` | clean | **pushed this session** | independent audit of Phase 3/4 |
| `/tmp/scdebussy-ru831-diagnostics` | `exp/ru831-diagnostics` | `b78905e` | clean | **pushed this session** | this document's own branch |

**Nothing was found at risk**: every `/tmp` worktree was clean (no
uncommitted or untracked scientifically-important work) and every branch
listed above is now on `origin`. If `/tmp` disappeared immediately after
this session, no commits would be lost (shared `.git` in
`/data1/chanj3/wangm10/scDeBussy`) and no branch refs would be lost (all
pushed to `origin`). The only thing that would be lost is the convenience of
the existing working directories themselves (re-creatable via `git worktree
add` from the pushed branches) and any non-Git persistent cache files not
already copied to `/data1/chanj3/wangm10/scDeBussy-v2/.htan_experiment_cache/`
-- the one produced this diagnostic round was verified copied there (section 1).

No worktrees were removed, pruned, reset, cleaned, merged, or rebased this
session, per instruction.

## 4. Recommended next session (do not start now)

1. Audit and rationalize the Git/worktree topology above -- five `/tmp`
   experimental/audit worktrees now exist off overlapping bases; decide
   which are still needed as working copies versus which can be safely
   removed now that all their branches are on `origin`.
2. Merge or cherry-pick `fix/joint-warp-diagnostics` (`6e25c8a`) into the
   appropriate joint-warp integration branch (likely a new branch off
   `exp/joint-posterior-warp`'s `ecf3134`, or directly onto it if that
   branch is meant to keep evolving).
3. Regenerate corrected Phase 3/4 reporting (`JOINT_WARP_RESULTS.md` and its
   run tables/figures) from the merged, corrected state -- the current
   `JOINT_WARP_RESULTS.md` on `exp/joint-posterior-warp` predates the
   diagnostics-fix branch's corrections (residual normalization, warp/
   patient-state fraction separation, `endpoint_pileup` -> `endpoint_occupancy`
   rename, paired restricted-unit baseline).
4. Reconcile this session's RU831 diagnostic findings
   (`RU831_DIAGNOSTICS.md`) with the independent Phase 3/4 audit's findings
   -- both point the same direction (joint model does not outperform the
   restricted-250/unit-precision baseline; RU831's apparent pathology is
   explainable without invoking multimodal correspondence) but were produced
   somewhat independently and should be cross-checked against each other
   once both branches are merged.
5. Decide whether the remaining complete-path-family-energy and genuine
   multi-initialization diagnostic work (flagged as gaps in
   `RU831_DIAGNOSTICS.md` Part 6) has enough expected information value to
   run, given that current evidence already leans away from a multimodal-
   correspondence explanation.
6. Do **not** implement a branched/mixture consensus architecture based on
   current evidence -- nothing produced so far meets the bar stated in
   `RU831_DIAGNOSTICS.md` Part 6 (complete competing path-family energies at
   matched parameters, or a clear consensus-compromise effect not explainable
   by support gaps or ordinary local warp).

## Summary

- RU831 diagnostic branch final SHA: `b78905e476e4a854067edf011442381dac3c4385` (`exp/ru831-diagnostics`) -- pushed.
- Fix branch SHA: `6e25c8ac4151f6b9773cc1e24544102929faced1` (`fix/joint-warp-diagnostics`) -- pushed.
- Branches pushed this session: `exp/ru831-diagnostics`, `fix/joint-warp-diagnostics`, `exp/joint-posterior-warp`, `audit/joint-posterior-warp-e774a9d`, `audit/joint-phase34-ecf3134`.
- Remaining uncommitted work: none found in any `/tmp` worktree.
- Anything lost if `/tmp` disappeared immediately: nothing -- all commits live in the shared canonical `.git` at `/data1/chanj3/wangm10/scDeBussy/.git`, and all relevant branch refs are now also on `origin`.
