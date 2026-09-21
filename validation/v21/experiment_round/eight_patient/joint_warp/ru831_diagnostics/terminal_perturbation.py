"""RU831 diagnostics, part 3: terminal-support diagnostic and one targeted perturbation.

Characterizes, for RU831/RU1303/control, where each patient's cost-surface
valley terminates relative to its actual data support, using already-cached
diagnostics (no re-fit needed for that half). Then runs exactly ONE targeted
perturbation: refit all 8 patients with the moderate omega_warp=0.1 graph but
a substantially higher lambda_terminal_gap than any of the 6 original
candidates (which used 1.5508 "teq" or 0.5 "tlow"), to test whether strongly
discouraging terminal-gap solutions only changes RU831's unsupported tail, or
also changes its internal SCLC-A-adjacent warp/match pattern -- distinguishing
"RU831 just lacks late support" from "RU831's internal correspondence is
ambiguous even where data exists".

This is the one explicitly-permitted perturbation in the task; no grid sweep.
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


def transformed_data(path, genes_file):
    """Load the input h5ad and restrict to the frozen top-250 gene panel."""
    data = ad.read_h5ad(path)
    genes = json.loads(Path(genes_file).read_text())
    if len(genes) != len(set(genes)) or any(g not in data.var_names for g in genes):
        raise ValueError("gene panel contains duplicate or absent genes")
    return data[:, genes].copy()


def refit(adata, resolved_config, overrides):
    """Re-run fit() with a resolved config, applying explicit overrides."""
    kwargs = dict(resolved_config)
    for key in ("lambda_i", "lambda_d", "calibration", "discovery_criterion"):
        kwargs.pop(key, None)
    kwargs.update(overrides)
    return align.fit(adata, **kwargs)


def main():
    """Run the single permitted terminal-gap perturbation and write its comparison outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--genes-file", type=Path, required=True)
    parser.add_argument("--base-report", type=Path, required=True, help="cached joint_warp_o0.1_teq.json (moderate)")
    parser.add_argument(
        "--high-lambda", type=float, default=15.5, help="lambda_terminal_gap for the perturbation, ~10x teq"
    )
    parser.add_argument("--patients", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-copy-dir", type=Path, required=True, help="off-repo dir to copy any .npz cache to")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_copy_dir.mkdir(parents=True, exist_ok=True)

    data = transformed_data(args.adata, args.genes_file)
    base_report = json.loads(args.base_report.read_text())
    resolved = base_report["resolved_fit_config"]

    print(
        f"[perturbation] refitting all patients with lambda_terminal_gap={args.high_lambda} "
        f"(omega_warp unchanged at {resolved['omega_warp']})",
        flush=True,
    )
    result = refit(data, resolved, {"lambda_terminal_gap": args.high_lambda})
    print(f"[perturbation] converged={result.converged} iterations={len(result.history)}", flush=True)

    from scdebussy.align._barycenter import costs

    cache_path = args.cache_copy_dir / "ru831_diag_high_lambda_terminal_gap.npz"
    np.savez_compressed(
        cache_path,
        patient_order=np.asarray(result.patients, str),
        barycenter=result.barycenter,
        consensus_coordinates=result.consensus_coordinates,
        landmark_offsets=np.r_[0, np.cumsum([len(lm.coordinates) for lm in result.landmarks])],
        landmark_coordinates=np.concatenate([lm.coordinates for lm in result.landmarks]),
        landmark_matchable=np.concatenate([lm.matchable for lm in result.landmarks]),
        cost_matrices=np.stack([costs(lm, result.barycenter) for lm in result.landmarks]),
        resolved_fit_config=json.dumps(result.config),
        diagnostics=json.dumps(result.diagnostics),
    )
    import hashlib

    sha = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    (args.cache_copy_dir / "MANIFEST.tsv").write_text(f"{cache_path.name}\t{sha}\n")
    print(f"[perturbation] cache copied to {cache_path} sha256={sha}", flush=True)

    comparison = {}
    for patient in args.patients:
        new_diag = result.diagnostics[patient]
        rows = {}
        for label, path in (
            (
                "teq (1.5508, baseline)",
                "validation/v21/experiment_round/eight_patient/joint_warp/joint_warp_o0.1_teq.json",
            ),
            (
                "tlow (0.5, baseline)",
                "validation/v21/experiment_round/eight_patient/joint_warp/joint_warp_o0.1_tlow.json",
            ),
        ):
            cached = json.loads(Path(path).read_text())["diagnostics"][patient]
            rows[label] = cached
        rows[f"high_lambda ({args.high_lambda})"] = new_diag
        comparison[patient] = rows

    (args.output_dir / "terminal_perturbation_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")

    # figure: key diagnostic fields across the three lambda settings, per patient
    fields = [
        "mean_displacement",
        "max_displacement",
        "patient_unmatched_terminal_mass",
        "consensus_unmatched_terminal_mass",
        "endpoint_pileup",
        "expected_match_12",
        "expected_match_21",
        "path_entropy",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), dpi=140)
    axes = axes.ravel()
    labels = ["teq (1.5508)", "tlow (0.5)", f"high ({args.high_lambda})"]
    colors = {"RU831": "#d62728", "RU1303": "#1f77b4", "RU1444": "#2ca02c", "RU581": "#9467bd"}
    for ax, field in zip(axes, fields, strict=True):
        for patient in args.patients:
            rows = comparison[patient]
            values = [
                rows["teq (1.5508, baseline)"][field],
                rows["tlow (0.5, baseline)"][field],
                rows[f"high_lambda ({args.high_lambda})"][field],
            ]
            ax.plot(labels, values, marker="o", label=patient, color=colors.get(patient))
        ax.set_title(field, fontsize=9)
        ax.tick_params(axis="x", labelrotation=25, labelsize=7)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "terminal_perturbation_fields.png", bbox_inches="tight")
    fig.savefig(args.output_dir / "terminal_perturbation_fields.pdf", bbox_inches="tight")
    plt.close(fig)

    # internal (non-terminal) region check: compare match/warp composition
    # restricted to the interior of each patient's matchable range (drop the
    # extreme decile at each end) between teq/tlow baselines and the high-
    # lambda perturbation, using per-landmark posterior mass from the refit
    # (baseline internal composition is approximated from cached scalar
    # internal_unmatched_mass / e11 / warp mass fields, since that's what's
    # cached; the refit gives us the full per-landmark picture for the
    # perturbation itself, reported alongside).
    internal_summary = {}
    for patient in args.patients:
        ix = list(result.patients).index(patient)
        lm = result.landmarks[ix]
        post = result.posteriors[ix]
        matchable_ix = np.flatnonzero(lm.matchable)
        if len(matchable_ix) < 10:
            continue
        lo, hi = np.percentile(matchable_ix, [10, 90])
        interior = matchable_ix[(matchable_ix >= lo) & (matchable_ix <= hi)]
        log_a = post.log_a[interior]
        finite = np.isfinite(log_a)
        e11_like = float(np.exp(np.where(finite, log_a, -np.inf)).sum())
        internal_summary[patient] = {
            "n_interior_landmarks": int(len(interior)),
            "interior_local_range": [float(lm.coordinates[interior[0]]), float(lm.coordinates[interior[-1]])],
            "interior_correspondence_mass_sum": e11_like,
            "interior_mean_row_max_prob": float(
                np.mean(
                    [
                        np.exp(row[np.isfinite(row)] - row[np.isfinite(row)].max()).max()
                        / np.exp(row[np.isfinite(row)] - row[np.isfinite(row)].max()).sum()
                        for row in log_a
                        if np.isfinite(row).any()
                    ]
                )
            ),
        }
    (args.output_dir / "terminal_perturbation_interior_summary.json").write_text(
        json.dumps(internal_summary, indent=2) + "\n"
    )
    print(json.dumps(internal_summary, indent=2))


if __name__ == "__main__":
    main()
