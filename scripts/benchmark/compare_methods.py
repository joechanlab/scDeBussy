"""Summarize and rank methods from a compare-run results directory.

Usage
-----
python scripts/benchmark/compare_methods.py \
    --results_dir /scratch/chanj3/wangm10/compare_run_tuning \
    --tuning true \
    --output /scratch/chanj3/wangm10/compare_run_tuning/comparison_summary.csv

Optional flags:
    --methods   identity scdebussy cellalign_fixed_reference
    --scenarios warp_medium noise_heavy combined_challenge
    --tuning    all|true|false filter rows by tuning_enabled
    --parquet   also write a parquet file alongside the CSV

Outputs
-------
* Console: formatted per-scenario comparison table + global ranking table
* <output>.csv  : long-form per-(method, scenario) summary
* <output>_ranked.csv : global ranking with win-rates across all scenarios
"""

from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.benchmark.aggregate_results import load_results  # noqa: E402

# ---------------------------------------------------------------------------
# Metric configuration
# ---------------------------------------------------------------------------

# (column, display_name, higher_is_better)
_METRICS: list[tuple[str, str, bool | None]] = [
    ("aligned_global_pearson_r", "Pearson r ↑", True),
    ("aligned_global_rmse", "RMSE ↓", False),
    ("delta_pearson_r", "ΔPearson r ↑", True),
    ("delta_rmse", "ΔRMSE ↓", False),
    ("aligned_n_patients_improved", "N patts improved ↑", True),
    ("aligned_worst_patient_rmse", "Worst-patt RMSE ↓", False),
    ("runtime_s", "Runtime (s)", None),  # None = info only
]

_PRIMARY_METRIC = "aligned_global_pearson_r"
_PRIMARY_HIGHER_IS_BETTER = True
_TUNING_CHOICES = ("all", "true", "false")


def _single_axis_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Return only methods that emit a single aligned axis for direct ranking."""
    if "evaluation_mode" not in df.columns:
        return df.copy()
    mask = df["evaluation_mode"] == "single_axis"
    return df.loc[mask].copy()


def _pairwise_only_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Return methods that only emit pairwise diagnostics, not a global axis."""
    if "evaluation_mode" not in df.columns:
        return df.iloc[0:0].copy()
    mask = df["evaluation_mode"] == "pairwise_only"
    return df.loc[mask].copy()


def _filter_by_tuning(df: pd.DataFrame, tuning: str) -> pd.DataFrame:
    """Restrict rows by tuning_enabled when requested."""
    if tuning == "all":
        return df.copy()

    if "tuning_enabled" not in df.columns:
        raise SystemExit(
            "[compare] Requested --tuning filter, but results do not contain tuning_enabled. "
            "Re-run aggregation on benchmark outputs written by the updated run_scenario.py."
        )

    want_enabled = tuning == "true"
    mask = df["tuning_enabled"].fillna(False).astype(bool) == want_enabled
    filtered = df.loc[mask].copy()
    if filtered.empty:
        raise SystemExit(f"[compare] No rows remain after applying --tuning {tuning}.")
    return filtered


def _report_tuning_filter_effects(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    *,
    tuning: str,
    requested_methods: list[str] | None,
) -> None:
    """Print per-method diagnostics when --tuning drops requested methods."""
    if tuning == "all" or df_before.empty:
        return

    methods_before = set(df_before["method"].unique())
    methods_after = set(df_after["method"].unique())

    if requested_methods:
        target_methods = [m for m in requested_methods if m in methods_before]
    else:
        target_methods = sorted(methods_before)

    dropped = [m for m in target_methods if m not in methods_after]
    if not dropped:
        return

    print(f"[compare] Methods removed by --tuning {tuning}: {sorted(dropped)}")
    if "tuning_enabled" not in df_before.columns:
        return

    for method in sorted(dropped):
        method_rows = df_before[df_before["method"] == method]
        n_rows = int(len(method_rows))
        n_tuned = int(method_rows["tuning_enabled"].fillna(False).astype(bool).sum())
        n_untuned = n_rows - n_tuned
        print(f"[compare]   - {method}: total_rows={n_rows}, tuned_rows={n_tuned}, untuned_rows={n_untuned}")

    if tuning == "true":
        print("[compare] Hint: use --tuning all to include both tuned and untuned rows.")
    else:
        print("[compare] Hint: use --tuning all to include both tuned and untuned rows.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt(val, decimals: int = 4) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "  —  "
    if isinstance(val, float):
        return f"{val:+.{decimals}f}" if decimals <= 4 else f"{val:.{decimals}f}"
    return str(val)


def _rank_within_group(grp: pd.DataFrame, col: str, higher_is_better: bool) -> pd.Series:
    """Return per-row ranks (1 = best) within a (scenario, seed) group."""
    ascending = not higher_is_better
    return grp[col].rank(ascending=ascending, method="min")


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------


def build_scenario_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean ± std across seeds for each (method, scenario, metric)."""
    metric_cols = [m[0] for m in _METRICS if m[0] in df.columns]
    grouped = df.groupby(["method", "scenario"])
    agg = grouped[metric_cols].agg(["mean", "std"])
    # Flatten multiindex columns → "pearson_r_mean", etc.
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg["n_seeds"] = grouped.size()

    if "tuning_enabled" in df.columns:
        agg["tuning_enabled_all"] = grouped["tuning_enabled"].all()
    if "tuning_n_trials" in df.columns:
        agg["tuning_n_trials_mean"] = grouped["tuning_n_trials"].mean()
    if "tuning_best_score" in df.columns:
        agg["tuning_best_score_mean"] = grouped["tuning_best_score"].mean()

    return agg.reset_index()


def build_global_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Rank methods using three complementary views, then combine them.

    Views
    -----
    1. Mean rank across all seeds and scenarios on the primary metric.
    2. Fraction of (scenario, seed, opponent) instances where this method wins.
    3. Mean Z-score of the primary metric across all seeds×scenarios.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "method",
                "overall_rank",
                "mean_rank",
                "win_rate",
                "mean_z_score",
            ]
        )

    primary = _PRIMARY_METRIC
    higher = _PRIMARY_HIGHER_IS_BETTER

    methods = sorted(df["method"].unique())

    # ---- 1. Average rank per (scenario, seed) --------------------------------
    rank_col = "__rank__"
    df = df.copy()
    df[rank_col] = df.groupby(["scenario", "seed"], group_keys=False)[primary].transform(
        lambda s: s.rank(ascending=not higher, method="min")
    )
    mean_rank = df.groupby("method")[rank_col].mean().rename("mean_rank")

    # ---- 2. Head-to-head win rate -------------------------------------------
    # For each (scenario, seed), compare each pair of methods.
    wins: dict[str, float] = dict.fromkeys(methods, 0.0)
    matches: dict[str, int] = dict.fromkeys(methods, 0)
    head_to_head: dict[tuple[str, str], dict] = {}

    for group_key, grp in df.groupby(["scenario", "seed"]):
        _scenario, _seed = group_key
        vals = grp.set_index("method")[primary].to_dict()
        for m_a, m_b in combinations(methods, 2):
            if m_a not in vals or m_b not in vals:
                continue
            v_a, v_b = vals[m_a], vals[m_b]
            if np.isnan(v_a) or np.isnan(v_b):
                continue
            pair = tuple(sorted([m_a, m_b]))
            head_to_head.setdefault(pair, dict.fromkeys(pair, 0))
            matches[m_a] += 1
            matches[m_b] += 1
            if (higher and v_a > v_b) or (not higher and v_a < v_b):
                wins[m_a] += 1
                head_to_head[pair][m_a] += 1
            elif (higher and v_b > v_a) or (not higher and v_b < v_a):
                wins[m_b] += 1
                head_to_head[pair][m_b] += 1
            else:
                wins[m_a] += 0.5
                wins[m_b] += 0.5
                head_to_head[pair][m_a] += 0.5
                head_to_head[pair][m_b] += 0.5

    win_rate = pd.Series(
        {m: wins[m] / matches[m] if matches[m] > 0 else np.nan for m in methods},
        name="win_rate",
    )

    # ---- 3. Z-score on primary metric ---------------------------------------
    grand_mean = df[primary].mean()
    grand_std = df[primary].std()
    mean_z = (
        df.groupby("method")[primary]
        .mean()
        .sub(grand_mean)
        .div(grand_std if grand_std > 1e-12 else 1.0)
        .rename("mean_z_score")
    )

    # ---- 4. Mean values for informational columns ---------------------------
    extra_cols = [m[0] for m in _METRICS if m[0] in df.columns and m[0] != primary]
    agg_extra = df.groupby("method")[extra_cols].mean()

    tuning_frames = []
    if "tuning_enabled" in df.columns:
        tuning_frames.append(df.groupby("method")["tuning_enabled"].all().rename("tuning_enabled"))
    if "tuning_n_trials" in df.columns:
        tuning_frames.append(df.groupby("method")["tuning_n_trials"].mean().rename("tuning_n_trials"))
    if "tuning_best_score" in df.columns:
        tuning_frames.append(df.groupby("method")["tuning_best_score"].mean().rename("tuning_best_score"))

    extra_parts = [mean_rank, win_rate, mean_z, agg_extra, *tuning_frames]

    ranking = pd.concat(extra_parts, axis=1).reindex(methods)
    ranking.index.name = "method"

    # Overall rank: sort by win_rate desc, break ties by mean_rank asc
    ranking = ranking.sort_values(
        ["win_rate", "mean_rank"],
        ascending=[False, True],
    )
    ranking.insert(0, "overall_rank", range(1, len(ranking) + 1))
    return ranking.reset_index()  # "method" index → column


def build_head_to_head_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a method × method win-count matrix on the primary metric."""
    if df.empty:
        return pd.DataFrame()

    primary = _PRIMARY_METRIC
    higher = _PRIMARY_HIGHER_IS_BETTER
    methods = sorted(df["method"].unique())
    matrix = pd.DataFrame(np.nan, index=methods, columns=methods)

    for group_key, grp in df.groupby(["scenario", "seed"]):
        _scenario, _seed = group_key
        vals = grp.set_index("method")[primary].to_dict()
        for m_a in methods:
            for m_b in methods:
                if m_a == m_b or m_a not in vals or m_b not in vals:
                    continue
                v_a, v_b = vals[m_a], vals[m_b]
                if np.isnan(v_a) or np.isnan(v_b):
                    continue
                current = matrix.at[m_a, m_b]
                current = 0.0 if np.isnan(current) else float(current)
                if (higher and v_a > v_b) or (not higher and v_a < v_b):
                    matrix.at[m_a, m_b] = current + 1
                else:
                    matrix.at[m_a, m_b] = current

    # Normalize each row by total non-NaN comparisons for that pair
    for m_a in methods:
        for m_b in methods:
            if m_a == m_b:
                matrix.at[m_a, m_b] = np.nan
                continue
            total = df.groupby(["scenario", "seed"]).size().shape[0]
            if total > 0 and not np.isnan(matrix.at[m_a, m_b]):
                matrix.at[m_a, m_b] = float(matrix.at[m_a, m_b]) / total

    return matrix


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

_SEP = "─" * 100


def _try_tabulate(df: pd.DataFrame, **kwargs) -> str:
    try:
        from tabulate import tabulate

        return tabulate(df, headers="keys", showindex=False, **kwargs)
    except ImportError:
        return df.to_string(index=False)


def print_scenario_comparison(summary: pd.DataFrame, methods: list[str]) -> None:
    """Print a per-scenario table comparing methods side-by-side."""
    scenarios = sorted(summary["scenario"].unique())

    print()
    print("=" * 100)
    print(f"  PER-SCENARIO COMPARISON   (mean ± std across seeds)   [primary: {_PRIMARY_METRIC}]")
    print("=" * 100)

    for scenario in scenarios:
        sub = summary[summary["scenario"] == scenario].set_index("method")
        n = int(sub.get("n_seeds", pd.Series()).iloc[0]) if not sub.empty else "?"
        print(f"\n  SCENARIO: {scenario}  (n={n} seeds per method)")
        print(_SEP)

        rows = []
        for col, label, hib in _METRICS:
            if f"{col}_mean" not in summary.columns:
                continue
            row = {"Metric": label}
            best_val: float | None = None
            for m in methods:
                if m not in sub.index:
                    row[m] = "n/a"
                    continue
                mean = sub.at[m, f"{col}_mean"]
                std = sub.at[m, f"{col}_std"] if f"{col}_std" in sub.columns else np.nan
                if np.isnan(mean):
                    row[m] = "  —  "
                    continue
                cell = f"{mean:+.4f}"
                if not np.isnan(std):
                    cell += f" ±{std:.4f}"
                row[m] = cell
                if hib is not None:
                    if best_val is None:
                        best_val = mean
                    elif (hib and mean > best_val) or (not hib and mean < best_val):
                        best_val = mean

            # Mark best method with *
            if hib is not None and best_val is not None:
                for m in methods:
                    if m in row and isinstance(row[m], str) and row[m].strip() not in ("n/a", "—"):
                        try:
                            v = float(row[m].split()[0])
                            if abs(v - best_val) < 1e-9:
                                row[m] = "* " + row[m]
                        except ValueError:
                            pass
            rows.append(row)

        print(_try_tabulate(pd.DataFrame(rows)))
    print()


def print_global_ranking(ranking: pd.DataFrame) -> None:
    """
    Print a formatted global ranking table of methods based on performance metrics.

    This function displays a comprehensive ranking of methods sorted by win rate (descending)
    and mean rank (ascending). It formats numerical values to 4 decimal places with sign
    indicators and handles missing/NaN values by displaying them as dashes.

    Parameters
    ----------
    ranking : pd.DataFrame
        A DataFrame containing method rankings with columns including:
        - overall_rank: The overall rank of the method
        - method: The name of the method
        - win_rate: Percentage of cases where method won
        - mean_rank: Average rank across benchmarks
        - mean_z_score: Average z-score
        - aligned_global_pearson_r: Pearson correlation coefficient
        - aligned_global_rmse: Root mean squared error
        - delta_pearson_r: Change in Pearson correlation
        - delta_rmse: Change in RMSE
        - aligned_n_patients_improved: Number of patients improved
        - runtime_s: Runtime in seconds

    Returns
    -------
    None
        Prints the formatted ranking table to stdout.

    Notes
    -----
    - Only columns present in the input DataFrame are displayed
    - The primary metric used for ranking is determined by _PRIMARY_METRIC
    - Numerical values are formatted with sign indicators (+/-) to 4 decimal places
    - Missing or NaN values are displayed as "—"
    """
    print()
    print("=" * 100)
    print("  GLOBAL METHOD RANKING")
    print(f"  Primary metric: {_PRIMARY_METRIC}  ({'higher' if _PRIMARY_HIGHER_IS_BETTER else 'lower'} is better)")
    print("  Sorted by: win_rate ↓, then mean_rank ↑")
    print("=" * 100)

    display_cols = [
        "overall_rank",
        "method",
        "tuning_enabled",
        "tuning_n_trials",
        "tuning_best_score",
        "win_rate",
        "mean_rank",
        "mean_z_score",
        "aligned_global_pearson_r",
        "aligned_global_rmse",
        "delta_pearson_r",
        "delta_rmse",
        "aligned_n_patients_improved",
        "runtime_s",
    ]
    display_cols = [c for c in display_cols if c in ranking.columns]

    fmt_df = ranking[display_cols].copy()
    for col in fmt_df.columns:
        if col in ("overall_rank", "method"):
            continue
        if col == "tuning_enabled":
            fmt_df[col] = fmt_df[col].map(
                lambda x: (
                    "—" if (x is None or (isinstance(x, float) and np.isnan(x))) else ("true" if bool(x) else "false")
                )
            )
            continue
        fmt_df[col] = fmt_df[col].apply(
            lambda x: (
                "—"
                if (x is None or (isinstance(x, float) and np.isnan(x)))
                else f"{x:+.4f}"
                if isinstance(x, float)
                else str(x)
            )
        )

    print(_try_tabulate(fmt_df, tablefmt="simple"))
    print()


def print_pairwise_only_summary(df: pd.DataFrame) -> None:
    """Print a summary for pairwise-only methods without forcing axis-based ranking."""
    if df.empty:
        return

    print()
    print("=" * 100)
    print("  PAIRWISE-ONLY METHODS")
    print("  These methods are excluded from the global single-axis ranking because they do not emit")
    print("  aligned_global_* metrics by design. Summary below uses pairwise diagnostics only.")
    print("=" * 100)

    cols = [
        "method",
        "scenario",
        "tuning_enabled",
        "tuning_n_trials",
        "runtime_s",
        "unsupervised_method_fraction_failed_pairs",
    ]
    available = [col for col in cols if col in df.columns]
    if not available:
        print("No comparable pairwise-only summary columns available.")
        print()
        return

    grouped = df.groupby(["method", "scenario"], sort=True)[available[2:]].mean(numeric_only=True).reset_index()
    if "tuning_enabled" in df.columns:
        grouped["tuning_enabled"] = (
            df.groupby(["method", "scenario"], sort=True)["tuning_enabled"].all().reset_index(drop=True)
        )
    print(_try_tabulate(grouped, tablefmt="simple"))
    print()


def print_head_to_head(h2h: pd.DataFrame) -> None:
    """
    Print a formatted head-to-head win rate comparison table.

    Displays a table showing the fraction of (scenario×seed) games each method wins against each other method.

    Args:
        h2h (pd.DataFrame): A DataFrame containing head-to-head win rates between methods,
                           with methods as both index and columns.

    Returns
    -------
        None
    """
    print()
    print("=" * 100)
    print("  HEAD-TO-HEAD WIN RATE  (row = fraction of (scenario×seed) games row beats column)")
    print("=" * 100)
    fmt = h2h.applymap(lambda x: "—" if np.isnan(x) else f"{x:.2%}")
    print(_try_tabulate(fmt.reset_index().rename(columns={"index": "method \\ vs"})))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Summarize and rank methods from a compare benchmark run.

    This function loads benchmark results from a directory structure, analyzes them across
    multiple scenarios and methods, and generates comprehensive summary reports including
    per-scenario summaries, global rankings, and head-to-head comparisons.

    The results are saved as CSV files (and optionally as Parquet files) to the specified
    output path. Three output files are generated:
    - Scenario summary: Per-scenario performance metrics for each method
    - Global ranking: Overall method rankings across all scenarios
    - Head-to-head: Pairwise comparisons between methods

    Command-line arguments:
        --results_dir (str): Root directory containing per-method subdirectories with
            seed_*.json result files.
        --output (str): Path for the per-scenario summary CSV file.
        --methods (list, optional): Restrict analysis to specified methods. If not provided,
            all methods found in results_dir are included.
        --scenarios (list, optional): Restrict analysis to specified scenarios. If not provided,
            all scenarios found in the data are included.
        --parquet (bool): If set, also write results as Parquet files alongside CSV files.

    Returns
    -------
        None

    Raises
    ------
        SystemExit: If required arguments are not provided.
    """
    parser = argparse.ArgumentParser(
        description="Summarize and rank methods from a compare benchmark run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_dir", required=True, help="Root directory containing per-method subdirs with seed_*.json files."
    )
    parser.add_argument("--output", required=True, help="Path for the per-scenario summary CSV.")
    parser.add_argument("--methods", nargs="*", default=None, help="Restrict to these methods (default: all found).")
    parser.add_argument(
        "--scenarios", nargs="*", default=None, help="Restrict to these scenarios (default: all found)."
    )
    parser.add_argument(
        "--tuning",
        default="all",
        choices=list(_TUNING_CHOICES),
        help="Filter rows by tuning_enabled: all=keep all results, true=only tuned runs, false=only untuned runs.",
    )
    parser.add_argument("--parquet", action="store_true", help="Also write parquet alongside the CSV.")
    args = parser.parse_args()

    print(f"\n[compare] Loading results from: {args.results_dir}")
    df_all = load_results(args.results_dir, scenarios=args.scenarios, methods=args.methods)
    df = _filter_by_tuning(df_all, args.tuning)
    _report_tuning_filter_effects(
        df_all,
        df,
        tuning=args.tuning,
        requested_methods=args.methods,
    )
    print(
        f"[compare] Loaded {len(df)} rows  ·  "
        f"methods: {sorted(df['method'].unique())}  ·  "
        f"scenarios: {sorted(df['scenario'].unique())}"
    )
    if "tuning_enabled" in df.columns:
        tuned_rows = int(df["tuning_enabled"].fillna(False).astype(bool).sum())
        print(f"[compare] Tuning filter={args.tuning}  ·  tuned rows: {tuned_rows}/{len(df)}")

    axis_df = _single_axis_subset(df)
    pairwise_df = _pairwise_only_subset(df)

    if axis_df.empty:
        print("[compare] No single-axis methods found; skipping axis-based ranking.")
    elif len(axis_df) != len(df):
        excluded = sorted(pairwise_df["method"].unique()) if not pairwise_df.empty else []
        print(f"[compare] Excluding pairwise-only methods from global ranking: {excluded}")

    methods = sorted(axis_df["method"].unique()) if not axis_df.empty else []

    # ---- Build summaries ----------------------------------------------------
    scenario_summary = build_scenario_summary(axis_df) if not axis_df.empty else pd.DataFrame()
    global_ranking = build_global_ranking(axis_df)
    h2h = build_head_to_head_table(axis_df)

    # ---- Print ---------------------------------------------------------------
    if not scenario_summary.empty:
        print_scenario_comparison(scenario_summary, methods)
    if not global_ranking.empty:
        print_global_ranking(global_ranking)
    if not h2h.empty:
        print_head_to_head(h2h)
    print_pairwise_only_summary(pairwise_df)

    # ---- Save ----------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    scenario_summary.to_csv(args.output, index=False)
    print(f"[compare] Scenario summary CSV  → {args.output}")

    ranked_path = os.path.splitext(args.output)[0] + "_ranked.csv"
    global_ranking.to_csv(ranked_path, index=False)
    print(f"[compare] Global ranking CSV    → {ranked_path}")

    h2h_path = os.path.splitext(args.output)[0] + "_head2head.csv"
    h2h.to_csv(h2h_path)
    print(f"[compare] Head-to-head CSV      → {h2h_path}")

    if args.parquet:
        scenario_summary.to_parquet(args.output.replace(".csv", ".parquet"), index=False)
        global_ranking.to_parquet(ranked_path.replace(".csv", ".parquet"), index=False)
        print("[compare] Parquet files written.")

    print()


if __name__ == "__main__":
    main()
