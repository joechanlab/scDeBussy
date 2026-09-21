"""RU831 diagnostics: check whether the reported bistable displacement is real.

Specifically: whether it is a floating-point-underflow artifact of the
kinetic-curve displacement diagnostic rather than a genuine shift of
posterior mass.

Motivation: posterior_diagnostics.py's row-entropy / multimodality / MAP-vs-
mean-path checks found RU831's per-row correspondence distributions to be
essentially IDENTICAL in shape and central tendency between the "moderate"
(o0.1_teq, mean/max displacement 0.115/0.213) and "severe" (o0.02_teq,
0.246/0.889) regimes for every landmark with a direct log_a entry. That is
inconsistent with a genuine bistable jump in posterior mass. This script
verifies a specific, more mundane explanation directly:

``_project.py:summarize()``'s kinetic-curve formula computes, per landmark
row i, ``exp(log_a[i,:] - log(patient_occupancy[i])) @ consensus -
local[i]``, restricted to rows where ``patient_occupancy[i] > 0`` (any
nonzero float, however small). For RU831's known unsupported tail block
(local pseudotime ~0.707-0.889, the same 19 landmarks flagged as
"unmatchable" by cost_geometry.py / the landmark discovery step), a
low-enough omega_warp can make some vanishingly improbable path through that
unsupported region numerically representable as a nonzero
``patient_occupancy`` (computed via a numerically-stable logaddexp
accumulation over many edges) while every INDIVIDUAL ``log_a[i, :]`` entry
for that same row underflows to exactly -inf (a coarser per-edge quantity).
When that happens, the row is treated as "occupied" (included in the
diplacement average) but its conditional-mean formula divides by
(numerically) zero support, degenerating to a "predicted position" of 0 and
therefore a displacement of magnitude equal to the landmark's own local
position (up to ~0.889) -- an artifact of the occupancy-vs-log_a numerical
threshold, not a real correspondence.

This script fits each candidate that matters for the bistability claim
(moderate baseline o0.1_teq, plus the three candidates independently flagged
as "severe" -- o0.02_teq, o0.02_tlow, o0.1_tlow -- all of which report
max_abs_displacement == 0.888888... for RU831) and reports, for each: the
count of "occupancy>0-but-no-direct-log_a" rows, their local positions and
occupancy magnitudes, and the resulting kinetic/displacement contribution,
compared against the cached report's mean/max displacement.
"""

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np

from scdebussy import align
from scdebussy.align._project import summarize


def transformed_data(path, genes_file):
    """Load the input h5ad and restrict to the frozen top-250 gene panel."""
    data = ad.read_h5ad(path)
    genes = json.loads(Path(genes_file).read_text())
    if len(genes) != len(set(genes)) or any(g not in data.var_names for g in genes):
        raise ValueError("gene panel contains duplicate or absent genes")
    return data[:, genes].copy()


def refit(adata, resolved_config):
    """Re-run fit() with a candidate's exact resolved config."""
    kwargs = dict(resolved_config)
    for key in ("lambda_i", "lambda_d", "calibration", "discovery_criterion"):
        kwargs.pop(key, None)
    return align.fit(adata, **kwargs)


def main():
    """CLI entry point: refit each candidate and report the occupancy/log_a underflow check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--genes-file", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--patient", default="RU831")
    parser.add_argument("--candidates", nargs="+", default=["o0.1_teq", "o0.02_teq", "o0.02_tlow", "o0.1_tlow"])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = transformed_data(args.adata, args.genes_file)
    report = {}
    for candidate in args.candidates:
        path = args.reports_dir / f"joint_warp_{candidate}.json"
        cached = json.loads(path.read_text())
        resolved = cached["resolved_fit_config"]
        print(f"[{candidate}] refitting", flush=True)
        result = refit(data, resolved)
        ix = list(result.patients).index(args.patient)
        lm = result.landmarks[ix]
        post = result.posteriors[ix]
        q = summarize(result)
        kinetic = q.kinetic_curve[ix]
        occ = post.patient_occupancy
        has_direct = np.isfinite(post.log_a).any(axis=1)
        degenerate = np.flatnonzero((occ > 0) & ~has_direct)
        official_mean = result.diagnostics[args.patient]["mean_displacement"]
        official_max = result.diagnostics[args.patient]["max_displacement"]
        cached_mean = cached["diagnostics"][args.patient]["mean_displacement"]
        cached_max = cached["diagnostics"][args.patient]["max_displacement"]
        report[candidate] = {
            "omega_warp": resolved["omega_warp"],
            "lambda_terminal_gap": resolved["lambda_terminal_gap"],
            "official_refit_mean_displacement": official_mean,
            "official_refit_max_displacement": official_max,
            "cached_mean_displacement": cached_mean,
            "cached_max_displacement": cached_max,
            "reproduces_cached_exactly": (official_mean == cached_mean and official_max == cached_max),
            "n_landmarks_matchable_by_data": int(lm.matchable.sum()),
            "n_landmarks_occupancy_gt_0": int((occ > 0).sum()),
            "n_landmarks_degenerate_occupancy_without_direct_log_a": int(len(degenerate)),
            "degenerate_landmark_local_positions": lm.coordinates[degenerate].tolist(),
            "degenerate_landmark_occupancy_values": occ[degenerate].tolist(),
            "degenerate_landmark_kinetic_values": kinetic[degenerate].tolist(),
            "mean_displacement_excluding_degenerate_rows": (
                float(np.mean(np.abs(np.delete(kinetic, degenerate)[np.isfinite(np.delete(kinetic, degenerate))])))
            ),
            "max_displacement_excluding_degenerate_rows": (
                float(np.max(np.abs(np.delete(kinetic, degenerate)[np.isfinite(np.delete(kinetic, degenerate))])))
            ),
        }
        print(json.dumps(report[candidate], indent=2), flush=True)

    (args.output_dir / "displacement_underflow_check.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
