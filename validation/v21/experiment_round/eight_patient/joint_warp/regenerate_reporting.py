"""Regenerate corrected Phase 4 tables and comparison heatmap from frozen caches.

This reporting-only command does not fit models or change the six-candidate grid.
"""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True,
                        help="persistent exp-htan-eight-patient cache directory")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]  # experiment_round
    output = args.output_dir
    baseline = args.cache_root / "eight_patient/results"
    joint = args.cache_root / "joint_warp"
    dtw = args.cache_root / "eight_patient/warp_sweep_unit/dtw_omega_0.06_terminal_1.npz"
    selected = joint / "joint_warp_o0.1_teq.npz"
    table = [
        sys.executable, str(output / "aggregate_joint_warp.py"),
        "--adata", str(args.adata), "--joint-dir", str(joint),
        "--report-dir", str(output),
        "--baseline", "original_v2", str(baseline / "original_v2.npz"),
        "--baseline", "trajectory_250", str(baseline / "trajectory_250.npz"),
        "--paired-unit-cache", str(baseline / "trajectory_250_unit.npz"),
        "--baseline", "moderate_dtw", str(dtw),
        "--heldout-summary", str(output / "heldout/heldout_summary.json"),
        "--output", str(output / "run_table.csv"),
    ]
    figure = [
        sys.executable, str(root / "summarize_and_plot.py"),
        "--adata", str(args.adata), "--candidate-dir", str(baseline),
        "--candidate", str(baseline / "original_v2.npz"),
        "--candidate", str(baseline / "trajectory_250.npz"),
        "--candidate", str(baseline / "trajectory_250_unit.npz"),
        "--candidate", str(selected), "--candidate", str(dtw),
        "--output-dir", str(output / "final"),
    ]
    tradeoff = [sys.executable, str(output / "plot_tradeoffs.py"),
                "--table", str(output / "run_table.csv"),
                "--output-dir", str(output / "figures")]
    subprocess.run(table, check=True)
    subprocess.run(figure, check=True)
    subprocess.run(tradeoff, check=True)


if __name__ == "__main__":
    main()
