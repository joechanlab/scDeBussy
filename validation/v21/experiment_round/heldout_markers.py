"""Evaluate predeclared lineage genes without using them in candidate fitting."""

import argparse
import csv
import json
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def smooth(time, expression, grid, radius=.06):
    distances = time[:, None] - grid[None, :]
    weight = np.exp(-.5 * (distances / radius) ** 2) * (abs(distances) <= 2 * radius)
    total = weight.sum(axis=0)
    estimate = np.divide(weight.T @ expression, total[:, None], out=np.full((len(grid), expression.shape[1]), np.nan), where=total[:, None] >= 10)
    return estimate


def evaluate(data, genes, positions, output_dir):
    selected = data[:, genes]
    x = selected.X.toarray() if hasattr(selected.X, "toarray") else np.asarray(selected.X)
    x = np.asarray(x, float)
    patient = data.obs.patient.astype(str).to_numpy()
    patient_names = sorted(np.unique(patient))
    grid = np.linspace(0, 1, 60)
    records = []
    fig, axes = plt.subplots(len(genes), len(positions), figsize=(3 * len(positions), 1.7 * len(genes)), sharex=True, dpi=150)
    for col, (name, position) in enumerate(positions.items()):
        trajectories = {}
        for label in patient_names:
            mask = (patient == label) & np.isfinite(position)
            trajectories[label] = smooth(position[mask], x[mask], grid)
        for g, gene in enumerate(genes):
            pairs = []
            peaks = []
            for a, left in enumerate(patient_names):
                trajectory = trajectories[left][:, g]
                if np.isfinite(trajectory).sum() >= 10 and np.nanmax(trajectory) > .1:
                    peaks.append(grid[np.nanargmax(trajectory)])
                axes[g, col].plot(grid, trajectory, lw=.8, alpha=.6)
                for right in patient_names[a+1:]:
                    other = trajectories[right][:, g]
                    valid = np.isfinite(trajectory) & np.isfinite(other)
                    if valid.sum() >= 10 and np.std(trajectory[valid]) > 1e-8 and np.std(other[valid]) > 1e-8:
                        pairs.append(np.corrcoef(trajectory[valid], other[valid])[0, 1])
            records.append({"method": name, "gene": gene, "mean_pairwise_trajectory_correlation": float(np.mean(pairs)) if pairs else float("nan"),
                            "n_pairs": len(pairs), "peak_position_spread": float(max(peaks)-min(peaks)) if peaks else float("nan"),
                            "n_patients_with_peak": len(peaks)})
            if col == 0: axes[g, col].set_ylabel(gene, fontsize=9)
            if g == 0: axes[g, col].set_title(name, fontsize=9)
    for ax in axes[-1]: ax.set_xlabel("pseudotime")
    fig.tight_layout(); fig.savefig(output_dir / "heldout_trajectories.png", bbox_inches="tight")
    with (output_dir / "heldout_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    aggregate = {}
    for name in positions:
        subset = [row for row in records if row["method"] == name]
        aggregate[name] = {"mean_gene_concordance": float(np.nanmean([row["mean_pairwise_trajectory_correlation"] for row in subset])),
                           "mean_peak_spread": float(np.nanmean([row["peak_position_spread"] for row in subset]))}
    (output_dir / "heldout_summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps(aggregate, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--markers", type=Path, required=True)
    parser.add_argument("--candidate", action="append", nargs=2, metavar=("NAME", "CACHE"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = ad.read_h5ad(args.adata)
    genes = json.loads(args.markers.read_text())["genes"]
    positions = {"s_local": data.obs.s_local.to_numpy(float)}
    for name, path in args.candidate:
        cache = np.load(path, allow_pickle=False)
        if not np.array_equal(cache["cell_ids"], np.asarray(data.obs_names, str)):
            raise ValueError(f"cache {path} does not match input")
        if "genes" in cache.files and set(genes) & set(cache["genes"]):
            raise ValueError(f"held-out genes leaked into {name} fitting")
        positions[name] = cache["position"]
    evaluate(data, genes, positions, args.output_dir)


if __name__ == "__main__":
    main()
