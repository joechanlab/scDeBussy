# Next experiment: joint posterior warp alignment

Design only. Do not implement this model in the current checkpoint. Freeze `eight_patient/feature_selection/selection_config.json` and `top_250.json`; do not search or rerank gene panels. The eight-patient experiment found a plausible RU1303 off-diagonal cost signal, but post-fit Viterbi movement reduced raw-input purity and had no calibrated posterior support.

## Smallest viable probabilistic contract

Use the existing source-to-sink DAG and forward/backward engine. Add an opt-in graph mode with match edges consuming `(1,1)`, `(1,2)`, or `(2,1)` landmarks. Each path remains monotone. Keep `(1,0)` and `(0,1)` edges for genuinely unmatched patient and consensus states. Annotate every edge with kind, consumed patient/consensus ranges, a *correspondence list* `(i,j,weight)`, warp count, and internal or terminal unmatched counts. The current affine-gap states may remain for internal gaps; use separate terminal edge costs in open mode. Terminal costs must apply to both leading and trailing suffix/prefix states, including the all-unmatched path. Require at least one match or impose a calibrated prior against degenerate all-terminal paths.

Use the prototype's per-patient-landmark emission rule so the EM objective is explicit:

- `(1,1)` at `(i,j)`: cost `delta[i,j]`, correspondence weight 1.
- `(1,2)` at `(i,j:j+2)`: cost `(delta[i,j]+delta[i,j+1])/2 + omega`; two correspondences of weight 1/2.
- `(2,1)` at `(i:i+2,j)`: cost `delta[i,j]+delta[i+1,j] + omega`; two correspondences of weight 1.

Thus each matched patient landmark contributes one unit of expression cost. Warp edges consume states without marking them unmatched. Keep `omega_warp`, `lambda_internal_gap` (plus existing per-state extension/open costs), and `lambda_terminal_gap` independent in config and output. A separate penalty for `(1,2)` versus `(2,1)` may be added only if synthetic tests show the shared penalty is insufficient.

## Marginals, conservation, and M-step

For edge posterior `q_e = exp(forward[source_e] - beta*energy_e + backward[target_e] - logZ)`, compute three distinct quantities:

1. **Correspondence weight** `A_ij = sum_e q_e * w_eij`. This is the coefficient of `delta_ij` in expected expression energy. `A` drives the barycenter and conditional landmark-to-consensus projection.
2. **State occupancy** `M_i = P(patient state i is consumed by any match edge)` and `O_j = P(consensus state j is consumed by any match edge)`. Each is in `[0,1]`. Patient unmatched probability `U_i=1-M_i`; consensus unmatched probability `D_j=1-O_j`. Accumulate internal and terminal parts separately from annotated gap edges.
3. **Transition diagnostics**: expected `(1,1)/(1,2)/(2,1)` counts, internal/terminal patient and consensus unmatched mass, gap opens, and a distribution of *matched patient states* (a `(2,1)` edge counts two, not one). Retain MAP edges only for visualization, not for inference.

The old identities in `tests/test_v2_lattice.py` are `sum_j A_ij + insertion_i=1`, `sum_i A_ij + deletion_j=1`, and expected matches plus insertions/deletions equals `n`/`t`. With the specified weights, the **patient-row identity remains** `sum_j A_ij + U_i=1`. The consensus-column identity **does not**: a `(2,1)` edge gives column correspondence weight 2 while occupancy is 1; a `(1,2)` edge gives weight 1/2 in each consumed column while occupancy is 1. Replace it with `O_j+D_j=1`. Count identities become `E[matched patient states]+E[unmatched patient states]=n` and `E[matched consensus states]+E[unmatched consensus states]=t`; warp-edge counts are separate. In old mode all quantities reduce to the existing identities.

For squared-error cost, the M-step still has the closed form `B[j,g] = sum_{p,i} A[p,i,j] * precision[p,i,g] * expression[p,i,g] / sum_{p,i} A[p,i,j] * precision[p,i,g]`. Keep prior `B[j,g]` only where the denominator has structural zero support. The current `sufficient_statistics` and `expected_match_energy` can use `A` unchanged **if** it is the weighted correspondence matrix above. The EM descent check must include the same edge-emission rule in each E-step; gap/warp costs are fixed during the M-step. The current `count_probability`, `expected_matches`, `insertion/deletion`, and `low_match_probability` names cannot silently retain edge-count meanings in joint mode. Add named fields or a joint-only posterior type and explicitly define whether low support uses matched patient or consensus occupancy.

Projection support must mean patient landmark matched occupancy `M_i=sum_j A_ij`, so `P(projected cell)+P(unprojected cell)=1` retains its current meaning. Conditional position uses `A_ij/M_i`; a `(1,2)` edge maps the patient landmark to the weighted midpoint. This does not make the posterior mean map necessarily monotone; report reversals and endpoint pile-ups rather than silently smoothing them. Consensus coverage and terminal missingness must use `O_j`, not `sum_i A_ij`.

## Files and backward compatibility

- `src/scdebussy/align/_model.py`: opt-in graph compiler, edge annotations, terminal costs, and small-lattice path oracle. Preserve current graph generation exactly for default v2.
- `src/scdebussy/align/_dp.py`: aggregate edge posteriors into `A`, occupancy, distinct gap classes, and matched-state count distribution. Forward/backward kernels can remain unchanged if edges stay topologically ordered.
- `src/scdebussy/align/_api.py`: new experimental mode and three independently recorded penalties; E/M uses joint graph at every iteration, with fixed-objective descent and final-posterior consistency checks.
- `src/scdebussy/align/_barycenter.py`: confirm weighted-correspondence support, M-step, and energy use the same emission coefficients; change only if needed for a distinct joint posterior type.
- `src/scdebussy/align/_project.py` and `_types.py`: consume named patient occupancy and consensus occupancy, preserve unconditional projection support and write provenance for the new contract.
- `tests/test_v2_lattice.py`, `tests/test_v2_api.py`, and a focused new joint-warp test module: preserve all old-mode assertions; add graph/oracle/EM/projection assertions for opt-in mode. Avoid broad refactoring.
- `validation/v21/experiment_round/`: add a new config and evaluator only after synthetic tests pass; reuse the frozen feature file and heatmap code.

Default `align.fit` must select the existing affine-gap graph and return its current fields/numerics. The new mode gets an explicit contract/version in config and provenance. If a `Posterior` field changes meaning, expose it under a new name or type rather than changing old-mode behavior.

## Synthetic gates before HTAN

For each case, compare exact tiny-lattice path enumeration with forward/backward `logZ`, edge marginals, `A`, state occupancy, and count distribution; test EM descent and final-posterior consistency. Run both NumPy and Numba inference where applicable.

1. **Pure stretch/compression, no missing state.** Construct shared expression landmarks sampled at different temporal rates, with all landmarks matchable and expensive internal/terminal gaps. Posterior warp probability should dominate gap probability, preserve near-unit patient and consensus occupancy, yield monotone MAP correspondence, and improve alignment versus no warp. Check both `(1,2)` and `(2,1)` directions.
2. **Missing internal segment.** Remove a contiguous interior expression state in one trajectory while retaining matching flanks. Internal unmatched posterior mass should concentrate in that segment; warp edges should not explain the missing state. Endpoints remain matched; terminal unmatched mass stays low.
3. **Missing terminal segment.** Remove a terminal consensus state while preserving the shared prefix. With terminal cost below internal cost, terminal unmatched posterior mass should dominate at the tail, while internal gap and forced warp mass remain low. Raising terminal cost should recover the expected tradeoff without making all paths unmatched.

Reject the design if all three cases are explained predominantly by the same edge class, if occupancy identities fail, if barycenter support becomes negative or exceeds its defined count interpretation, or if EM free energy increases beyond numerical slack.

## Minimal eight-patient evaluation after implementation

Use only the frozen trajectory-250 genes and **unit precision first**. Fix all other settings to `configs/trajectory_250_unit.json`. Calibrate one internal gap cost from that fit, then run four warp penalties chosen from the synthetic cost scale (for example `omega` 0.02, 0.05, 0.1, 0.2), each with (a) terminal cost equal to internal cost and (b) one prespecified lower terminal cost. This is eight joint fits, plus the existing restricted 250-unit fit and raw `s_local` control. Do not tune the gene panel or expand the grid based on HTAN labels.

For every run report convergence, iterations, coverage, cross-patient 50-neighbor cell-type purity, rejected and finite-unsupported fractions, mean/median/p95 displacement, RU1303 SCLC-A fraction inside other patients' SCLC-N q10–q90 interval, held-out marker concordance, expected internal versus terminal unmatched posterior mass, warp transition mass, and per-patient values. Plot each patient's `s_local -> aligned_position` and flag jumps, plateaus, reversals, compression, and endpoint pile-ups. Reuse `summarize_and_plot.py` conventions exactly: RU1083/RU1303/RU1444/RU151/RU263/RU581/RU831/RU942 order, 100 bins, yellow LUAD/red SCLC-A/cyan SCLC-N, gray <3 finite cells, separate rejected/unsupported strip, and `[0,1]` axes. Evaluate against raw input and the corresponding restricted-warp fit; do not choose by heatmap appearance alone.

Likely failure modes: extra paths alter entropy and cost calibration; cheap terminal edges absorb whole patients; `(2,1)` emission double-counts a consensus column if old occupancy code is reused; weak support makes an EM update unstable; a posterior-mean map can reverse or flatten despite monotone paths; fixed 250-gene selection remains cohort-specific; and an apparent purity gain may come from rejecting difficult cells or compressing RU831. Keep all of these visible in diagnostics.
