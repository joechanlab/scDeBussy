"""Per-patient s_local -> aligned_position warp-function plots for joint-warp candidates.

Complements summarize_and_plot.py (heatmap/score conventions) with the
finer-grained per-cell mapping view NEXT_JOINT_WARP_PLAN.md asks for: plateaus,
abrupt jumps, endpoint accumulation, and compression are all visible directly
as departures from the y=x raw-input line.
"""

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PATIENTS = ["RU1083", "RU1303", "RU1444", "RU151", "RU263", "RU581", "RU831", "RU942"]


def load(path, data):
    """Load one candidate's projected position, checking cell identity/order."""
    d = np.load(path, allow_pickle=False)
    if not np.array_equal(d["cell_ids"], np.asarray(data.obs_names, str)):
        raise ValueError(f"cell identity/order mismatch in {path}")
    return d["position"], json.loads(str(d["config"]))["name"]


def main():
    """Render per-patient s_local -> aligned_position warp-function plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = ad.read_h5ad(args.adata)
    s_local = data.obs["s_local"].to_numpy(float)
    patients = data.obs["patient"].astype(str).to_numpy()

    candidates = [("raw $s_{local}$", s_local)]
    for path in args.candidate:
        position, name = load(path, data)
        candidates.append((name, position))

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), dpi=170, sharex=True, sharey=True)
    palette = plt.cm.tab10(np.linspace(0, 1, len(candidates)))
    for i, patient in enumerate(PATIENTS):
        axis = axes.flat[i]
        mask = patients == patient
        for (name, position), color in zip(candidates, palette, strict=True):
            x, y = s_local[mask], np.asarray(position, float)[mask]
            finite = np.isfinite(x) & np.isfinite(y)
            order = np.argsort(x[finite])
            axis.plot(
                x[finite][order], y[finite][order], ".", ms=1.5, alpha=0.35, color=color, label=name if i == 0 else None
            )
        axis.plot([0, 1], [0, 1], color="black", lw=0.8, ls="--", alpha=0.6)
        axis.set_title(patient, fontsize=10)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
    for ax in axes[-1]:
        ax.set_xlabel("$s_{local}$")
    for ax in axes[:, 0]:
        ax.set_ylabel("aligned position")
    fig.legend(loc="upper left", bbox_to_anchor=(1.0, 0.95), fontsize=8, frameon=False)
    fig.suptitle("Per-patient warp function: $s_{local} \\to$ aligned position", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.output_dir / "warp_functions.png", bbox_inches="tight")
    fig.savefig(args.output_dir / "warp_functions.pdf", bbox_inches="tight")
    print(args.output_dir / "warp_functions.png")


if __name__ == "__main__":
    main()
