"""Purity/coverage/displacement tradeoff plots for the joint-warp run table."""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    """Render the purity/coverage/displacement tradeoff scatter plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.table)
    joint = df[df["run_type"] == "joint_posterior_warp"]
    others = df[df["run_type"] != "joint_posterior_warp"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=170)
    for ax, (x, y) in zip(
        axes,
        [("mean_abs_displacement", "purity"), ("mean_abs_displacement", "coverage"), ("coverage", "purity")],
        strict=True,
    ):
        ax.scatter(others[x], others[y], color="gray", marker="s", s=60, label="baseline")
        for _, row in others.iterrows():
            ax.annotate(
                row["method"], (row[x], row[y]), fontsize=7, color="gray", xytext=(3, 3), textcoords="offset points"
            )
        ax.scatter(joint[x], joint[y], color="tab:red", marker="o", s=60, label="joint warp")
        for _, row in joint.iterrows():
            tag = row["method"].replace("joint_warp_", "")
            ax.annotate(tag, (row[x], row[y]), fontsize=7, color="tab:red", xytext=(3, -8), textcoords="offset points")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
    axes[0].axhline(0.7715030274767686, color="black", lw=0.8, ls="--", label="raw purity")
    axes[1].axhline(0.865, color="black", lw=0.8, ls="--", label="raw coverage")
    axes[2].axhline(0.7715030274767686, color="black", lw=0.8, ls="--")
    axes[2].axvline(0.865, color="black", lw=0.8, ls="--")
    axes[0].legend(fontsize=7)
    fig.suptitle("Joint posterior-warp candidates vs baselines")
    fig.tight_layout()
    fig.savefig(args.output_dir / "tradeoffs.png", bbox_inches="tight")
    fig.savefig(args.output_dir / "tradeoffs.pdf", bbox_inches="tight")
    print(args.output_dir / "tradeoffs.png")


if __name__ == "__main__":
    main()
