"""Plot cost surfaces and tabulate RU1303 alternatives and gene contributions."""

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    return np.load(path, allow_pickle=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original, candidate = load(args.original), load(args.candidate)
    fig, axes = plt.subplots(8, 2, figsize=(10, 24), sharex=True, sharey=True, dpi=160)
    overview = []
    for col, (label, data) in enumerate((("Original all-gene", original), (args.candidate.stem, candidate))):
        names = data["patient_order"].tolist()
        for row, patient in enumerate(sorted(names)):
            p = names.index(patient)
            cost = data["cost_matrices"][p]
            offsets = data["landmark_offsets"]
            local = data["landmark_coordinates"][offsets[p]:offsets[p+1]]
            valid = data["landmark_matchable"][offsets[p]:offsets[p+1]]
            consensus = data["consensus_coordinates"]
            best = np.argmin(cost, axis=1)
            distance = np.abs(consensus[best[valid]] - local[valid])
            overview.append({"objective": label, "patient": patient,
                             "mean_row_min_distance": float(distance.mean()),
                             "median_row_min_distance": float(np.median(distance)),
                             "late_20_row_min_median": float(np.median(consensus[best[-20:]]))})
            ax = axes[row, col]
            image = np.log1p(cost / max(np.median(cost), 1e-12))
            ax.imshow(image, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="viridis")
            ax.plot(local, local, color="white", lw=1, ls="--", label="identity")
            ax.plot(consensus[best], local, color="#ff6b6b", lw=1, label="row minimum")
            ax.set_title(f"{patient}: {label}", fontsize=10)
            ax.set_ylabel("patient landmark")
            if row == 7: ax.set_xlabel("consensus position")
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.output_dir / "cost_matrices.png", bbox_inches="tight")
    with (args.output_dir / "cost_surface_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(overview[0])); writer.writeheader(); writer.writerows(overview)

    data = candidate
    p = data["patient_order"].tolist().index("RU1303")
    cost = data["cost_matrices"][p]
    offsets = data["landmark_offsets"]
    start, stop = offsets[p:p+2]
    local = data["landmark_coordinates"][start:stop]
    expression = data["landmark_expression"][start:stop]
    precision = data["landmark_precision"][start:stop]
    consensus = data["consensus_coordinates"]
    barycenter = data["barycenter"]
    late_rows = []
    contributions = []
    genes = data["genes"]
    for i in range(max(0, len(local)-20), len(local)):
        options = np.argsort(cost[i])[:5]
        identity = np.argmin(abs(consensus - local[i]))
        late_rows.append({"landmark": i, "local_position": float(local[i]),
                          "identity_position": float(consensus[identity]),
                          "identity_cost": float(cost[i, identity]),
                          "top_5_positions": ";".join(f"{consensus[j]:.4f}" for j in options),
                          "top_5_costs": ";".join(f"{cost[i, j]:.6g}" for j in options)})
        best = options[0]
        contrast = precision[i] * ((expression[i] - barycenter[identity])**2 - (expression[i] - barycenter[best])**2)
        for g in np.argsort(-abs(contrast))[:10]:
            contributions.append({"landmark": i, "gene": genes[g], "identity_minus_best_contribution": float(contrast[g]),
                                  "identity_position": float(consensus[identity]), "best_position": float(consensus[best])})
    for filename, rows in (("ru1303_late_candidates.csv", late_rows), ("ru1303_gene_contributions.csv", contributions)):
        with (args.output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=180)
    ax.imshow(np.log1p(cost / max(np.median(cost), 1e-12)), origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="viridis")
    ax.plot(local, local, "w--", lw=1.2, label="identity")
    ax.plot(consensus[np.argmin(cost, axis=1)], local, color="#ff6b6b", lw=1.3, label="lowest cost")
    ax.axhspan(.8, 1, color="white", alpha=.12)
    ax.set(xlabel="consensus position", ylabel="RU1303 local landmark", title=f"RU1303 cost surface: {args.candidate.stem}")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(args.output_dir / "ru1303_cost_matrix.png")


if __name__ == "__main__":
    main()
