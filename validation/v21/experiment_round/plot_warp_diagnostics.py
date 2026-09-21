"""Plot the fixed-consensus sweep and patient warp functions."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--moderate", type=Path, required=True)
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(args.sweep.open()))
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), dpi=170)
    palette = {"1.0": "#3366aa", "0.2": "#dc8433", "0.1": "#ad3a3a"}
    for terminal in sorted({r["terminal_cost"] for r in rows}, reverse=True):
        part = [r for r in rows if r["terminal_cost"] == terminal]
        color = palette.get(terminal, "gray")
        displacement = np.array([float(r["mean_abs_displacement"]) for r in part])
        purity = np.array([float(r["purity"]) for r in part])
        coverage = np.array([float(r["coverage"]) for r in part])
        axes[0, 0].plot(displacement, purity, "o-", color=color, label=f"terminal {terminal}")
        axes[0, 1].plot(displacement, coverage, "o-", color=color)
        axes[1, 0].plot(coverage, purity, "o-", color=color)
        terminal_frac = np.array([float(r["terminal_consensus_gap"]) + float(r["terminal_patient_gap"]) for r in part]) / 800
        internal_frac = np.array([float(r["internal_consensus_gap"]) + float(r["internal_patient_gap"]) for r in part]) / 800
        warp_frac = np.array([float(r["warp_12"]) + float(r["warp_21"]) for r in part]) / 800
        axes[1, 1].plot(displacement, terminal_frac, "o-", color=color, ls="-", label=f"terminal {terminal}")
        axes[1, 1].plot(displacement, internal_frac, "s--", color=color, label=f"internal {terminal}")
        axes[1, 1].plot(displacement, warp_frac, "^:", color=color, label=f"warp {terminal}")
    axes[0, 0].axhline(.7715030274767686, color="black", lw=1, ls="--", label="raw purity")
    axes[0, 1].axhline(.865, color="black", lw=1, ls="--")
    axes[1, 0].axhline(.7715030274767686, color="black", lw=1, ls="--")
    axes[1, 0].axvline(.865, color="black", lw=1, ls="--")
    axes[0, 0].set(xlabel="mean absolute displacement", ylabel="purity")
    axes[0, 1].set(xlabel="mean absolute displacement", ylabel="coverage")
    axes[1, 0].set(xlabel="coverage", ylabel="purity")
    axes[1, 1].set(xlabel="mean absolute displacement", ylabel="fraction of 800 landmarks/rows")
    axes[0, 0].legend(fontsize=7)
    axes[1, 1].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(args.output_dir / "sweep_tradeoffs.png", bbox_inches="tight")

    fig, axes = plt.subplots(2, 4, figsize=(12, 5), dpi=170, sharex=True, sharey=True)
    flags = []
    for path, color, title in ((args.moderate, "#3366aa", "moderate warp"), (args.terminal, "#dc8433", "terminal setting")):
        cache = np.load(path, allow_pickle=False)
        names = cache["patient_order"].tolist()
        details = json.loads(str(cache["details"]))
        for i, patient in enumerate(sorted(names)):
            axis = axes.flat[i]
            mapped = cache["maps"][names.index(patient)]
            local = np.linspace(0, 1, len(mapped))
            axis.plot(local, local, color="#aaa", lw=1)
            axis.plot(local, mapped, color=color, lw=1.2, label=title)
            differences = np.diff(mapped)
            flags.append({"method": title, "patient": patient,
                          "maximum_adjacent_jump": float(differences.max()),
                          "longest_flat_run": int(max(np.diff(np.r_[-1, np.flatnonzero(differences > 1e-4), len(differences)])) - 1),
                          "terminal_consensus_gap": details[patient]["terminal_consensus_gap"],
                          "internal_patient_gap": details[patient]["internal_patient_gap"],
                          "internal_consensus_gap": details[patient]["internal_consensus_gap"]})
            axis.set_title(patient, fontsize=9)
    for ax in axes.flat:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0, 0].legend(fontsize=7)
    fig.supxlabel("local pseudotime"); fig.supylabel("aligned position")
    fig.tight_layout(); fig.savefig(args.output_dir / "patient_warp_functions.png", bbox_inches="tight")
    with (args.output_dir / "warp_flags.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flags[0])); writer.writeheader(); writer.writerows(flags)


if __name__ == "__main__":
    main()
