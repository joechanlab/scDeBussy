"""Aggregate per-job JSON results into a single CSV (and optionally parquet).

Usage
-----
    # After SLURM array completes:
    python scripts/benchmark/aggregate_results.py \
        --results_dir results/ \
        --output summary.csv \
        [--parquet]

    # Optionally restrict to a subset of scenarios:
    python scripts/benchmark/aggregate_results.py \
        --results_dir results/ \
        --output summary.csv \
        --scenarios warp_medium noise_heavy combined_challenge
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Columns that are straightforward scalars from the top-level JSON
# ---------------------------------------------------------------------------
_TOP_LEVEL_SCALARS = ["method", "evaluation_mode", "scenario", "seed", "runtime_s"]


def _flatten_result(rec: dict) -> dict[str, Any]:
    """Flatten a single JSON result dict into a flat dict of scalar values."""
    out: dict[str, Any] = {}

    for key in _TOP_LEVEL_SCALARS:
        out[key] = rec.get(key)

    # Baseline section
    for section in ("baseline", "aligned"):
        d = rec.get(section, {})
        out[f"{section}_global_pearson_r"] = d.get("global_pearson_r")
        out[f"{section}_global_rmse"] = d.get("global_rmse")
        # Per-patient → flatten as patient_0_pearson_r, patient_0_rmse, ...
        for pp in d.get("per_patient", []):
            pid = pp.get("patient", "?").replace(" ", "_")
            out[f"{section}_{pid}_pearson_r"] = pp.get("pearson_r")
            out[f"{section}_{pid}_rmse"] = pp.get("rmse")
            out[f"{section}_{pid}_n_cells"] = pp.get("n_cells")

    # Aligned-specific fairness stats
    aligned = rec.get("aligned", {})
    out["aligned_n_patients_improved"] = aligned.get("n_patients_improved")
    out["aligned_worst_patient_rmse"] = aligned.get("worst_patient_rmse")

    # Unsupervised section (backward-compatible key)
    for k, v in rec.get("unsupervised", {}).items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[f"unsupervised_{k}"] = v

    # New structured unsupervised sections
    for k, v in rec.get("unsupervised_generic", {}).items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[f"unsupervised_generic_{k}"] = v
    for k, v in rec.get("unsupervised_method", {}).items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[f"unsupervised_method_{k}"] = v

    tuning = rec.get("tuning", {})
    if isinstance(tuning, dict):
        out["tuning_enabled"] = bool(tuning.get("enabled", False))
        out["tuning_mode"] = tuning.get("mode")
        out["tuning_objective"] = tuning.get("objective")
        out["tuning_n_trials"] = tuning.get("n_trials")
        out["tuning_n_successful_trials"] = tuning.get("n_successful_trials")
        out["tuning_best_score"] = tuning.get("best_score")

    # Convenience: delta pearson / delta rmse
    b_r = out.get("baseline_global_pearson_r")
    a_r = out.get("aligned_global_pearson_r")
    if b_r is not None and a_r is not None:
        out["delta_pearson_r"] = float(a_r) - float(b_r)

    b_rmse = out.get("baseline_global_rmse")
    a_rmse = out.get("aligned_global_rmse")
    if b_rmse is not None and a_rmse is not None:
        out["delta_rmse"] = float(a_rmse) - float(b_rmse)  # negative = improved

    return out


def load_results(
    results_dir: str,
    scenarios: list[str] | None = None,
    methods: list[str] | None = None,
) -> pd.DataFrame:
    """Load and flatten all JSON results under *results_dir*.

    Parameters
    ----------
    results_dir : str
        Root directory that contains per-scenario subdirectories with
        ``seed_<N>.json`` files.
    scenarios : list of str or None
        Optional allow-list of scenario names.  When ``None`` all scenarios
        found under *results_dir* are loaded.

    Returns
    -------
    df : pd.DataFrame
        One row per (scenario, seed) pair.
    """
    pattern = os.path.join(results_dir, "**", "seed_*.json")
    paths = sorted(glob.glob(pattern, recursive=True))

    if not paths:
        sys.exit(f"[ERROR] No result files found under: {results_dir}")

    rows = []
    errors = []
    for path in paths:
        try:
            with open(path) as fh:
                rec = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            errors.append((path, str(exc)))
            continue

        if scenarios and rec.get("scenario") not in scenarios:
            continue
        if methods and rec.get("method", "scdebussy") not in methods:
            continue

        rows.append(_flatten_result(rec))

    if errors:
        print(f"[WARN] Skipped {len(errors)} unreadable files:")
        for p, e in errors[:10]:
            print(f"  {p}: {e}")

    if not rows:
        sys.exit("[ERROR] No valid result records found after filtering.")

    df = pd.DataFrame(rows)
    # Sort for reproducibility
    sort_cols = [c for c in ["method", "scenario", "seed"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def print_summary_table(df: pd.DataFrame) -> None:
    """Print a per-method/per-scenario mean ± std summary to stdout."""
    key_metrics = [
        "aligned_global_pearson_r",
        "aligned_global_rmse",
        "aligned_n_patients_improved",
        "aligned_worst_patient_rmse",
        "delta_pearson_r",
        "delta_rmse",
        "runtime_s",
    ]
    available = [c for c in key_metrics if c in df.columns]

    if "scenario" not in df.columns:
        return

    if "method" not in df.columns:
        df = df.assign(method="unknown")

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY  (mean ± std across seeds)")
    print("=" * 80)

    grouped = df.groupby(["method", "scenario"], sort=True)
    for (method, scenario), grp in grouped:
        print(f"\n  Method   : {method}")
        print(f"  Scenario : {scenario}  (n={len(grp)} seeds)")
        for col in available:
            vals = grp[col].dropna()
            if vals.empty:
                continue
            m = vals.mean()
            s = vals.std()
            unit = "s" if col == "runtime_s" else ""
            print(f"    {col:<40s}  {m:+.4f} ± {s:.4f}{unit}")

    print("=" * 80 + "\n")


def main():
    """
    Aggregate per-job JSON files from scenario subdirectories into a single CSV summary.

    Parses command-line arguments to determine the input results directory, output file path,
    and optional filters for scenarios and methods. Loads results using the specified filters,
    writes the aggregated data to a CSV file (and optionally to a Parquet file), and prints
    a summary table of the results.

    Command-line Arguments:
        --results_dir (str): Root directory containing per-scenario JSON subdirectories. Required.
        --output (str): Path for the output CSV file. Required.
        --parquet (bool): If set, also write a parquet file alongside the CSV. Default: False.
        --scenarios (list[str]): Optional list of scenario names to include. Default: None (all scenarios).
        --methods (list[str]): Optional list of methods to include. Default: None (all methods).

    Returns
    -------
        None

    Side Effects:
        - Creates output directory if it does not exist.
        - Writes aggregated results to a CSV file.
        - Optionally writes aggregated results to a Parquet file.
        - Prints status messages and a summary table to stdout.
    """
    parser = argparse.ArgumentParser(
        description="Aggregate per-job JSON files into a single CSV summary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_dir", required=True, help="Root directory containing per-scenario JSON subdirectories."
    )
    parser.add_argument("--output", required=True, help="Path for the output CSV file.")
    parser.add_argument("--parquet", action="store_true", help="Also write a parquet file alongside the CSV.")
    parser.add_argument(
        "--scenarios", nargs="*", default=None, help="Optional list of scenario names to include (default: all)."
    )
    parser.add_argument(
        "--methods", nargs="*", default=None, help="Optional list of methods to include (default: all)."
    )
    args = parser.parse_args()

    df = load_results(args.results_dir, scenarios=args.scenarios, methods=args.methods)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[OK] CSV written → {args.output}  ({len(df)} rows, {len(df.columns)} columns)")

    if args.parquet:
        parquet_path = os.path.splitext(args.output)[0] + ".parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"[OK] Parquet written → {parquet_path}")

    print_summary_table(df)


if __name__ == "__main__":
    main()
