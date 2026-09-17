# v2 implementation validation

Baseline: `a8de9c4d86fb620e2c08e21ee1edc67f29568c42` on `strict-soft-em`.
Implementation worktree: `/tmp/scdebussy-v2`, branch `refactor/v2`.
The original working tree, staged benchmark deletions, and untracked documents
were not copied into or altered by this worktree. No manuscript tag was changed.

The baseline algorithm and all pre-existing tests are unchanged. The package
initializer now uses lazy exports; importing the experimental API does not load
Torch. The combined regression suite passed **518 tests** (256 legacy,
262 v2), with one legacy warning. A subsequent 24-test API run passed three
additional edge cases, bringing the verified total to **521 distinct tests**.
The initial 289-test run already included the
first 33 v2 fixtures collected during startup; it was not a legacy-only count.

Validation uses `/data1/chanj3/wangm10/conda_envs/scdebussy-py311/bin/python`.
The system Python could not import AnnData because its NumPy/h5py binaries are
incompatible; it was not used to claim test success. Exact package versions and
source checksums are recorded in `environment.json` after final validation.

The benchmark contract is `scripts/v2/contract.json`; it was written before the
held-out suite ran. `smoke/` and `full/` contain copied contracts, per-case
records, and gate summaries. Raw case records are generated artifacts; summaries
and aggregate metrics are retained for review. A failed scientific gate is a
result, not permission to relax mathematical invariants or tune held-out seeds.

The historical documentation cites coverage 0.89 / purity 0.630 on nine patients.
The discovered saved `aligned_scdebussy_preprocessed_v2.h5ad` contains **eight**
patients; `aligned_scdebussy_preprocessed_v3.h5ad` contains nine. The real-data
runner therefore recomputes the baseline on the nine-patient input and checks
whether those historical metrics are reproduced before evaluating replacement.
It reports the exact input, configurations, cell/gene counts, metrics, and
warnings. It does not treat a filename as proof of matching preprocessing.

Current release posture: experimental; v1 remains recommended. Consult the
machine-readable summaries for completed gates and failures.

## Available nine-patient comparison

The same-input rerun on 34,553 cells and 2,370 genes produced:

| Metric | Frozen v1 soft-EM | v2 |
|---|---:|---:|
| Mean occupied-bin coverage | 0.71778 | 0.74333 |
| Cross-patient kNN purity (k=50) | 0.55256 | 0.70208 |
| Fraction with finite position | 1.00000 | 0.99991 |

V2 converged in 7 iterations; mean retained cell mass was 0.98333. Excluding
unsupported cells yields purity 0.70515 and coverage 0.73333. This is an
encouraging comparison on the available data, **not** reproduction of the
historical baseline: the v1 metrics differ from the cited 0.89/0.63, and the
unfiltered coverage difference also exceeds 0.02. Gate D remains failed.

## Synthetic validation and limits

All 755 cases completed and passed their frozen numeric suite checks:
10 calibration, 280 recovery, 300 null (100 each for exchangeable, blocked,
and paired designs), 120 robustness, and 45 sensitivity cases. The separate
19-case smoke suite also passed. The full metrics are in `full/metrics.csv`;
the post-run descriptive analysis is in `full/decomposition.json`.

These preset checks cover alignment error, finite projection, null rejection
rates, a confidently-incorrect projection criterion, convergence, and resolution
RMSE. They **do not establish a clean biological decomposition**:

- Pure delay at noise 0.04 triggered occupancy tests in all 10 held-out runs;
  its expression family had discoveries in 3 of 10 runs despite no planted
  expression change.
- Pure occupancy at noise 0.04 triggered kinetic tests in 6 of 10 runs.
- The pure-expression scenario at noise 0.04 recovered no planted genes after
  BH correction in this eight-patient preset. The preset's patient count and
  permutation resolution limit gene-level power; effect recovery is not established.
- Kinetic signs agreed across resolutions 50/100/200 and RMSE ranges were under
  0.008, but mean gap fractions changed by approximately 0.14–0.23. No significant
  genes were found in that resolution sweep, so gene-set stability is uninformative.

These are retained scientific findings, not hidden by the passing suite flags.
The API's tests establish condition association with fitted quantities, not
identification of independent biological causes. No package publication or
recommendation to replace v1 has been made.
