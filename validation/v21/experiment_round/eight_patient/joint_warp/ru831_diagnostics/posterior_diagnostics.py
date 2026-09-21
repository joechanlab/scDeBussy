"""RU831 diagnostics, part 2: competing-path / posterior diagnostics.

Re-fits two of the six already-run joint-warp candidates (a "moderate"
displacement regime and a "severe" displacement regime, exact resolved
configs read from their cached .json reports) in-process, so the full
per-landmark JointPosterior (log_a) is available -- the cached .npz files
only carry cost matrices and scalar summary diagnostics, not the full
posterior matrix.

For RU831, RU1303, and a control patient, in both regimes: computes per-row
(per-landmark) entropy of the normalized correspondence distribution,
detects locally multimodal rows (two or more well-separated near-maximal
correspondence regions), and compares the MAP path (JointPosterior.map_edges)
to the posterior-mean path (conditional-mean consensus position per row,
matching _project.py:summarize's kinetic_curve construction) to check for
divergence -- itself evidence of multimodality if MAP and mean disagree.

Also verifies the refit reproduces the cached candidate .json diagnostics
(path_entropy, expected_match_11/12/21, mean/max displacement) within a
stated numerical tolerance, since this is the load-bearing assumption behind
reusing a re-derived posterior as a stand-in for the original fit.
"""

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scdebussy import align


def load_resolved_config(report_path):
    """Read a cached candidate .json report's resolved_fit_config block."""
    report = json.loads(Path(report_path).read_text())
    return report


def transformed_data(path, genes_file):
    """Load the input h5ad and restrict to the frozen top-250 gene panel."""
    data = ad.read_h5ad(path)
    genes = json.loads(Path(genes_file).read_text())
    if len(genes) != len(set(genes)) or any(g not in data.var_names for g in genes):
        raise ValueError("gene panel contains duplicate or absent genes")
    return data[:, genes].copy()


def refit(adata, resolved_config):
    """Re-run fit() with the exact resolved config of an existing candidate."""
    kwargs = dict(resolved_config)
    # drop derived/reporting-only fields fit() does not accept as kwargs
    for key in ("lambda_i", "lambda_d", "calibration", "discovery_criterion"):
        kwargs.pop(key, None)
    return align.fit(adata, **kwargs)


def verify_reproduction(result, cached_report, patient, tol=1e-6):
    """Compare refit diagnostics against the cached .json report for one patient."""
    cached = cached_report["diagnostics"][patient]
    refit_diag = result.diagnostics[patient]
    checks = {}
    for key in ("path_entropy", "expected_match_11", "expected_match_12", "expected_match_21"):
        a, b = cached[key], refit_diag[key]
        checks[key] = {
            "cached": a,
            "refit": b,
            "abs_diff": abs(a - b),
            "within_tol": abs(a - b) <= tol * max(1.0, abs(a)),
        }
    return checks


def row_entropy(log_a_row):
    """Shannon entropy (nats) of exp(log_a_row) normalized to a distribution.

    Rows with no finite support (fully unmatched landmark) return NaN.
    """
    finite = np.isfinite(log_a_row)
    if not finite.any():
        return float("nan"), 0
    m = log_a_row[finite].max()
    w = np.exp(log_a_row[finite] - m)
    z = w.sum()
    if z <= 0:
        return float("nan"), 0
    p = w / z
    ent = float(-(p * np.log(np.clip(p, 1e-300, None))).sum())
    return ent, int(finite.sum())


def multimodal_rows(log_a, consensus, gap_nats=1.0, min_component_mass=0.05):
    """Detect rows with >=2 well-separated near-maximal correspondence regions.

    For each row, cluster consensus indices with finite log_a into contiguous
    runs, keep runs whose max log-weight is within `gap_nats` of the row's
    overall max log-weight (i.e. log-probability gap <= gap_nats), and whose
    total probability mass (after normalizing the row) is >= min_component_mass.
    A row is "multimodal" if >=2 such runs remain and they are not adjacent.
    """
    results = []
    for i in range(log_a.shape[0]):
        row = log_a[i]
        finite = np.isfinite(row)
        if finite.sum() < 2:
            continue
        idx = np.flatnonzero(finite)
        splits = np.flatnonzero(np.diff(idx) > 1)
        runs = np.split(idx, splits + 1)
        row_max = row[finite].max()
        m = row_max
        w_full = np.where(finite, np.exp(row - m), 0.0)
        z = w_full.sum()
        if z <= 0:
            continue
        p_full = w_full / z
        kept = []
        for run in runs:
            run_max_log = row[run].max()
            run_mass = float(p_full[run].sum())
            if (row_max - run_max_log) <= gap_nats and run_mass >= min_component_mass:
                kept.append((run, run_max_log, run_mass))
        if len(kept) >= 2:
            results.append(
                {
                    "landmark": i,
                    "n_components": len(kept),
                    "component_positions": [float(consensus[run[np.argmax(row[run])]]) for run, _, _ in kept],
                    "component_masses": [m for _, _, m in kept],
                    "component_log_gap": [float(row_max - lm) for _, lm, _ in kept],
                }
            )
    return results


def map_vs_mean_path(landmarks, posterior, patient_ix, consensus):
    """Per-row MAP (mode of the row's marginal correspondence) vs posterior-mean path.

    ``JointPosterior.map_edges`` decodes the single best *global* path through
    the warp graph (a joint MAP over the whole sequence, via edge indices into
    ``JointGraph``'s internal correspondence tables); reconstructing per-row
    (i, j) pairs from it requires additional graph internals not otherwise
    needed here. For the local-multimodality question this script asks --
    does an individual landmark's own correspondence distribution have one
    clear mode or two competing ones -- the more direct and equally valid
    quantity is the row-wise argmax of ``log_a`` (each landmark's own MAP
    correspondence, i.e. the mode of its marginal distribution), which is what
    is computed and compared against the posterior-mean path below. This is
    stated explicitly rather than presented as the graph's joint MAP path.

    IMPORTANT: under warp="on", ``log_a`` is a correspondence *weight*, not a
    match probability (JOINT_WARP_CONTRACT.md; see also JOINT_WARP_RESULTS.md's
    "Mathematical contract" section) -- ``sum_j A[i,j]`` is NOT in general equal
    to the row's true occupancy probability, because a (1,2)/(2,1) warp edge
    contributes fractional weight to more than one row/column and occupancy is
    computed separately via directly-aggregated interval accumulation
    (``JointPosterior.patient_occupancy``). ``_project.py:summarize()`` (which
    is what ``fit()``'s own cached ``diagnostics[patient]['mean_displacement']``
    is computed from, see ``_api.py``'s call to ``summarize(result)``) divides
    by ``patient_occupancy``, not by ``logsumexp(log_a, axis=1)``. This
    function reproduces that normalization exactly for ``mean_pos`` (a first
    version of this script normalized by the row's own weight-sum instead,
    which is a different, systematically smaller-magnitude quantity -- see
    RU831_DIAGNOSTICS.md for why that first version's regime-to-regime
    comparison was misleading). ``map_pos`` (this row's own argmax of
    ``log_a``) is unaffected by this distinction, since argmax does not depend
    on the normalizing constant.
    """
    lm = landmarks[patient_ix]
    p = posterior[patient_ix]
    # NOTE: `valid` here requires a DIRECT log_a entry (`finite`), which is a
    # deliberately stricter condition than official summarize()'s `rows =
    # np.isfinite(log_m)` (occupancy > 0 alone). That stricter condition is
    # correct for this function's purpose (rows with no direct log_a entry
    # have no real correspondence shape to compare a MAP-vs-mean path on).
    # See displacement_underflow_check.py: rows with occupancy > 0 but NO
    # direct log_a entry are exactly the mechanism behind RU831's reported
    # aggregate "bistable displacement" -- a numerical-underflow artifact of
    # summarize()'s own broader `rows` condition, not a genuine correspondence
    # shift, and out of scope for this function's local-multimodality check.
    log_a = p.log_a
    occupancy = p.patient_occupancy
    finite = np.isfinite(log_a).any(axis=1)
    mean_pos = np.full(len(lm.coordinates), np.nan)
    map_pos = np.full(len(lm.coordinates), np.nan)
    if finite.any():
        m = np.where(finite[:, None], log_a, -np.inf)
        log_m = np.where(occupancy > 0, np.log(np.maximum(occupancy, 1e-300)), -np.inf)
        valid = finite & np.isfinite(log_m)
        weights = np.zeros_like(m)
        weights[valid] = np.exp(m[valid] - log_m[valid, None])
        mean_pos[valid] = weights[valid] @ consensus
        argmax_j = np.argmax(m, axis=1)
        map_pos[valid] = consensus[argmax_j[valid]]
    return map_pos, mean_pos


def main():
    """CLI entry point: refit both regimes and write entropy/multimodality/MAP-vs-mean outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--genes-file", type=Path, required=True)
    parser.add_argument("--moderate-report", type=Path, required=True, help="cached joint_warp_o0.1_teq.json")
    parser.add_argument("--severe-report", type=Path, required=True, help="cached joint_warp_o0.02_teq.json")
    parser.add_argument("--patients", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = transformed_data(args.adata, args.genes_file)

    all_checks = {}
    entropy_rows = []
    multimodal_summary = []
    path_rows = []

    for label, report_path in (("moderate_o0.1_teq", args.moderate_report), ("severe_o0.02_teq", args.severe_report)):
        cached_report = json.loads(report_path.read_text())
        resolved = cached_report["resolved_fit_config"]
        print(f"[{label}] refitting with resolved config from {report_path.name}", flush=True)
        result = refit(data, resolved)
        print(f"[{label}] converged={result.converged} iterations={len(result.history)}", flush=True)

        checks = {}
        for patient in args.patients:
            checks[patient] = verify_reproduction(result, cached_report, patient)
        all_checks[label] = checks

        consensus = result.consensus_coordinates
        for patient in args.patients:
            ix = list(result.patients).index(patient)
            lm = result.landmarks[ix]
            post = result.posteriors[ix]
            for i in range(post.log_a.shape[0]):
                ent, support = row_entropy(post.log_a[i])
                entropy_rows.append(
                    {
                        "regime": label,
                        "patient": patient,
                        "landmark": i,
                        "local_position": float(lm.coordinates[i]),
                        "matchable": bool(lm.matchable[i]),
                        "row_entropy_nats": ent,
                        "n_finite_support": support,
                        "global_path_entropy": post.entropy,
                    }
                )
            mm = multimodal_rows(post.log_a, consensus)
            for row in mm:
                multimodal_summary.append({"regime": label, "patient": patient, **row})

            map_pos, mean_pos = map_vs_mean_path(result.landmarks, result.posteriors, ix, consensus)
            diff = np.abs(map_pos - mean_pos)
            for i in range(len(map_pos)):
                path_rows.append(
                    {
                        "regime": label,
                        "patient": patient,
                        "landmark": i,
                        "local_position": float(lm.coordinates[i]),
                        "map_position": float(map_pos[i]) if np.isfinite(map_pos[i]) else None,
                        "mean_position": float(mean_pos[i]) if np.isfinite(mean_pos[i]) else None,
                        "abs_diff": float(diff[i]) if np.isfinite(diff[i]) else None,
                    }
                )

            # figure: log_a heatmap with MAP and mean overlays
            fig, ax = plt.subplots(figsize=(6.5, 6), dpi=160)
            im = ax.imshow(
                np.where(np.isfinite(post.log_a), post.log_a, np.nan),
                origin="lower",
                extent=[0, 1, 0, 1],
                aspect="auto",
                cmap="viridis",
            )
            ax.plot(mean_pos, lm.coordinates, color="#ff6b6b", lw=1.3, label="posterior mean path")
            ax.plot(map_pos, lm.coordinates, color="#00e5ff", lw=1.0, ls="--", label="row-mode (per-row MAP) path")
            for row in mm:
                ax.scatter(
                    row["component_positions"],
                    [lm.coordinates[row["landmark"]]] * len(row["component_positions"]),
                    color="white",
                    s=14,
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=5,
                )
            ax.set(
                xlabel="consensus position",
                ylabel=f"{patient} local landmark",
                title=f"{patient} log_a: {label}\n(white dots = multimodal rows)",
            )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log_a")
            ax.legend(fontsize=7, loc="lower right")
            fig.tight_layout()
            fig.savefig(args.output_dir / f"{patient}_{label}_posterior.png", bbox_inches="tight")
            plt.close(fig)

    # side-by-side entropy comparison figure across patients/regimes
    fig, axes = plt.subplots(1, len(args.patients), figsize=(5.5 * len(args.patients), 4.5), dpi=150, sharey=True)
    if len(args.patients) == 1:
        axes = [axes]
    for ax, patient in zip(axes, args.patients, strict=True):
        for label, color in (("moderate_o0.1_teq", "#1f77b4"), ("severe_o0.02_teq", "#d62728")):
            rows = [r for r in entropy_rows if r["patient"] == patient and r["regime"] == label and r["matchable"]]
            rows.sort(key=lambda r: r["local_position"])
            xs = [r["local_position"] for r in rows]
            ys = [r["row_entropy_nats"] for r in rows]
            ax.plot(xs, ys, label=label, color=color, lw=1.3)
        ax.set(xlabel="local pseudotime", title=patient)
    axes[0].set_ylabel("per-landmark correspondence entropy (nats)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "row_entropy_comparison.png", bbox_inches="tight")
    fig.savefig(args.output_dir / "row_entropy_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    (args.output_dir / "reproduction_check.json").write_text(json.dumps(all_checks, indent=2) + "\n")

    import csv

    with (args.output_dir / "row_entropy.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(entropy_rows[0]))
        writer.writeheader()
        writer.writerows(entropy_rows)

    if multimodal_summary:
        with (args.output_dir / "multimodal_rows.csv").open("w", newline="") as handle:
            fieldnames = sorted({k for r in multimodal_summary for k in r})
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for r in multimodal_summary:
                r = dict(r)
                for k in ("component_positions", "component_masses", "component_log_gap"):
                    r[k] = ";".join(f"{v:.4f}" for v in r[k])
                writer.writerow(r)
    else:
        (args.output_dir / "multimodal_rows.csv").write_text("no multimodal rows detected under the stated threshold\n")

    with (args.output_dir / "map_vs_mean_path.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(path_rows[0]))
        writer.writeheader()
        writer.writerows(path_rows)

    # aggregate summary
    summary = {}
    for label in ("moderate_o0.1_teq", "severe_o0.02_teq"):
        summary[label] = {}
        for patient in args.patients:
            rows = [r for r in entropy_rows if r["patient"] == patient and r["regime"] == label and r["matchable"]]
            ent = [r["row_entropy_nats"] for r in rows if np.isfinite(r["row_entropy_nats"])]
            pv = [
                r for r in path_rows if r["patient"] == patient and r["regime"] == label and r["abs_diff"] is not None
            ]
            diffs = [r["abs_diff"] for r in pv]
            n_mm = len([r for r in multimodal_summary if r["patient"] == patient and r["regime"] == label])
            summary[label][patient] = {
                "mean_row_entropy": float(np.mean(ent)) if ent else None,
                "median_row_entropy": float(np.median(ent)) if ent else None,
                "max_row_entropy": float(np.max(ent)) if ent else None,
                "n_multimodal_rows": n_mm,
                "n_matchable_rows": len(rows),
                "frac_multimodal_rows": n_mm / len(rows) if rows else None,
                "map_vs_mean_mean_abs_diff": float(np.mean(diffs)) if diffs else None,
                "map_vs_mean_max_abs_diff": float(np.max(diffs)) if diffs else None,
                "map_vs_mean_frac_large_disagreement_gt_0.05": (
                    float(np.mean([d > 0.05 for d in diffs])) if diffs else None
                ),
            }
    (args.output_dir / "posterior_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
