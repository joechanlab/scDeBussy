"""Plot benchmark comparison across methods: Pearson r and RMSE.

Usage
-----
    python scripts/benchmark/plot_comparison.py \
        --results_dir /scratch/chanj3/wangm10/compare_run_tuning \
        --output_dir  /scratch/chanj3/wangm10/compare_run_tuning/figures

    To regenerate after new runs or with a subset:
    python scripts/benchmark/plot_comparison.py \
        --results_dir /scratch/chanj3/wangm10/compare_run_tuning \
        --output_dir  /scratch/chanj3/wangm10/compare_run_tuning/figures \
        --scenarios warp_medium warp_hard_monotone combined_challenge \
        --fmt png --dpi 200

Optional flags:
    --methods   identity scdebussy cellalign_fixed_reference
                cellalign_consensus genes2genes_fixed_reference genes2genes_consensus
    --scenarios warp_medium noise_heavy combined_challenge
    --fmt       pdf   (default: pdf; also accepts png, svg)
    --dpi       150   (rasterisation DPI for png output)

Figures produced
----------------
1.  boxplot_pearson_r.pdf   – aligned Pearson r per scenario, all methods
2.  boxplot_rmse.pdf        – aligned RMSE per scenario, all methods
3.  delta_pearson_r.pdf     – ΔPearson r vs baseline per scenario
4.  delta_rmse.pdf          – ΔRMSE vs baseline per scenario
5.  heatmap_pearson_r.pdf   – method × scenario mean Pearson r heatmap
6.  heatmap_rmse.pdf        – method × scenario mean RMSE heatmap
7.  win_rate_bar.pdf        – global head-to-head win-rate bar chart
8.  rank_jitter.pdf         – per-seed rank strip (Pearson r) across scenarios
9.  overview_grid.pdf       – 2×2 summary grid (Pearson r + RMSE abs + delta)
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_ROOT, _SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.benchmark.aggregate_results import load_results  # noqa: E402

# ---------------------------------------------------------------------------
# Colour / style config
# ---------------------------------------------------------------------------
_METHOD_COLORS = {
    "identity": "#6baed6",  # steel blue
    "scdebussy": "#2ca02c",  # green
    "cellalign_fixed_reference": "#d62728",  # red
    "cellalign_consensus": "#ff7f0e",  # orange
    "genes2genes_fixed_reference": "#9467bd",  # violet
    "genes2genes_consensus": "#8c564b",  # brown
}
_METHOD_LABELS = {
    "identity": "Baseline (identity)",
    "scdebussy": "scDeBussy",
    "cellalign_fixed_reference": "CellAlign (medoid ref)",
    "cellalign_consensus": "CellAlign (consensus)",
    "genes2genes_fixed_reference": "Genes2Genes (medoid ref)",
    "genes2genes_consensus": "Genes2Genes (consensus)",
}
_SCENARIO_ORDER = [
    "clean_baseline",
    "noise_heavy",
    "noise_student_t",
    "warp_easy",
    "warp_medium",
    "warp_hard_monotone",
    "warp_sigmoid",
    "warp_nonlinear",
    "partial_coverage",
    "cell_imbalance",
    "high_p",
    "combined_challenge",
]
_SCENARIO_LABELS = {
    "clean_baseline": "Clean\nbaseline",
    "noise_heavy": "Noise\nheavy",
    "noise_student_t": "Noise\nStudent-t",
    "warp_easy": "Warp\neasy",
    "warp_medium": "Warp\nmedium",
    "warp_hard_monotone": "Warp\nhard mono.",
    "warp_sigmoid": "Warp\nsigmoid",
    "warp_nonlinear": "Warp\nnonlinear",
    "partial_coverage": "Partial\ncoverage",
    "cell_imbalance": "Cell\nimbalance",
    "high_p": "High P\n(10 patients)",
    "combined_challenge": "Combined\nchallenge",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _method_order(df: pd.DataFrame, methods: list[str] | None) -> list[str]:
    all_methods = sorted(df["method"].unique())
    if methods:
        all_methods = [m for m in methods if m in all_methods]

    # Rank methods by overall performance: higher Pearson and lower RMSE.
    need_cols = {"aligned_global_pearson_r", "aligned_global_rmse"}
    if not need_cols.issubset(df.columns) or len(all_methods) <= 1:
        return all_methods

    perf = (
        df[df["method"].isin(all_methods)]
        .groupby("method")[["aligned_global_pearson_r", "aligned_global_rmse"]]
        .mean()
        .reindex(all_methods)
    )

    pearson_mean = perf["aligned_global_pearson_r"].mean()
    pearson_std = perf["aligned_global_pearson_r"].std()
    rmse_mean = perf["aligned_global_rmse"].mean()
    rmse_std = perf["aligned_global_rmse"].std()

    z_pearson = (perf["aligned_global_pearson_r"] - pearson_mean) / (pearson_std if pearson_std > 1e-12 else 1.0)
    z_rmse = (perf["aligned_global_rmse"] - rmse_mean) / (rmse_std if rmse_std > 1e-12 else 1.0)

    # Higher composite means better overall performance.
    perf["overall_score"] = z_pearson - z_rmse

    perf = perf.sort_values(
        ["overall_score", "aligned_global_pearson_r", "aligned_global_rmse"],
        ascending=[False, False, True],
    )
    return perf.index.tolist()


def _scenario_order(df: pd.DataFrame) -> list[str]:
    present = set(df["scenario"].unique())
    return [s for s in _SCENARIO_ORDER if s in present] + sorted(present - set(_SCENARIO_ORDER))


def _label(m: str) -> str:
    return _METHOD_LABELS.get(m, m)


def _color(m: str) -> str:
    return _METHOD_COLORS.get(m, "#888888")


def _slabel(s: str) -> str:
    return _SCENARIO_LABELS.get(s, s)


def _savefig(fig: plt.Figure, path: str, fmt: str, dpi: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    full = f"{path}.{fmt}"
    fig.savefig(full, format=fmt, dpi=dpi, bbox_inches="tight")
    print(f"  [saved] {full}")
    plt.close(fig)


def _add_significance_bracket(ax, x1, x2, y, h, pval):
    """Draw a bracket with significance stars between positions x1 and x2."""
    if pval < 0.001:
        stars = "***"
    elif pval < 0.01:
        stars = "**"
    elif pval < 0.05:
        stars = "*"
    else:
        return  # not significant — skip bracket
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, c="0.3")
    ax.text((x1 + x2) / 2, y + h, stars, ha="center", va="bottom", fontsize=8, color="0.2")


# ---------------------------------------------------------------------------
# Figure 1 & 2 – Box plots per scenario for abs metrics
# ---------------------------------------------------------------------------


def _boxplot_metric(df, methods, scenarios, col, ylabel, title, higher_is_better, fmt, dpi, out_prefix):
    n_sc = len(scenarios)
    n_m = len(methods)
    width = 0.7 / n_m
    offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * width

    fig, ax = plt.subplots(figsize=(max(10, n_sc * 1.1), 4.5))

    for i, sc in enumerate(scenarios):
        sub = df[df["scenario"] == sc]
        for j, m in enumerate(methods):
            vals = sub[sub["method"] == m][col].dropna().to_numpy()
            if vals.size == 0:
                continue
            bp = ax.boxplot(
                vals,
                positions=[i + offsets[j]],
                widths=width * 0.88,
                patch_artist=True,
                notch=False,
                showfliers=True,
                flierprops={"marker": ".", "markersize": 3, "alpha": 0.5, "color": _color(m)},
                medianprops={"color": "white", "linewidth": 0.3},
                whiskerprops={"linewidth": 0.9, "color": _color(m)},
                capprops={"linewidth": 0.9, "color": _color(m)},
                boxprops={"linewidth": 0.0},
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(_color(m))
                patch.set_alpha(0.82)

    # Reference line at 0 for delta metrics
    if col.startswith("delta_"):
        ax.axhline(0, color="0.5", linewidth=0.8, linestyle="--", zorder=0)

    ax.set_xticks(range(n_sc))
    ax.set_xticklabels([_slabel(s) for s in scenarios], rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", pad=8)

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=_color(m), alpha=0.82, label=_label(m)) for m in methods]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linewidth=0.4, color="0.85", zorder=0)

    _savefig(fig, out_prefix, fmt, dpi)


def _draw_grouped_metric_boxplots_on_axis(ax, df, methods, scenarios, col):
    """Draw method-colored grouped boxplots for one metric onto a provided axis."""
    n_m = len(methods)
    width = 0.7 / n_m
    offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * width

    for i, sc in enumerate(scenarios):
        sub = df[df["scenario"] == sc]
        for j, m in enumerate(methods):
            vals = sub[sub["method"] == m][col].dropna().to_numpy()
            if vals.size == 0:
                continue
            bp = ax.boxplot(
                vals,
                positions=[i + offsets[j]],
                widths=width * 0.88,
                patch_artist=True,
                notch=False,
                showfliers=True,
                flierprops={"marker": ".", "markersize": 3, "alpha": 0.5, "color": _color(m)},
                medianprops={"color": "white", "linewidth": 1},
                whiskerprops={"linewidth": 0.9, "color": _color(m)},
                capprops={"linewidth": 0.9, "color": _color(m)},
                boxprops={"linewidth": 0.0},
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(_color(m))
                patch.set_alpha(0.82)

    if col.startswith("delta_"):
        ax.axhline(0, color="0.5", linewidth=0.8, linestyle="--", zorder=0)

    # Publication panel style: no background grid guides.


def _combined_pearson_rmse_boxplot(df, methods, scenarios, fmt, dpi, out_prefix):
    """Create a stacked publication panel: Pearson (top) and RMSE (bottom)."""
    n_sc = len(scenarios)
    fig, axes = plt.subplots(2, 1, figsize=(max(10, n_sc * 1.1), 7.2), sharex=True)

    # Top panel: Pearson
    _draw_grouped_metric_boxplots_on_axis(axes[0], df, methods, scenarios, col="aligned_global_pearson_r")
    axes[0].set_ylabel("Pearson r", fontsize=20)
    axes[0].tick_params(axis="y", labelsize=12)
    axes[0].tick_params(axis="x", labelbottom=False)

    # Bottom panel: RMSE
    _draw_grouped_metric_boxplots_on_axis(axes[1], df, methods, scenarios, col="aligned_global_rmse")
    axes[1].set_ylabel("RMSE", fontsize=20)
    axes[1].tick_params(axis="y", labelsize=12)
    axes[1].set_xticks(range(n_sc))
    axes[1].set_xticklabels([_slabel(s) for s in scenarios], rotation=0, fontsize=12)

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=_color(m), alpha=0.82, label=_label(m)) for m in methods]
    axes[0].legend(
        handles=legend_handles,
        loc="lower left",
        ncol=1,
        bbox_to_anchor=(0.01, 0.02),
        framealpha=0.9,
        fontsize=13,
    )

    fig.tight_layout()
    _savefig(fig, out_prefix, fmt, dpi)


# ---------------------------------------------------------------------------
# Figure 5 & 6 – Heatmaps
# ---------------------------------------------------------------------------


def _heatmap(df, methods, scenarios, col, title, fmt_str, cmap, fmt, dpi, out_prefix):
    pivot = df.groupby(["method", "scenario"])[col].mean().unstack("scenario").reindex(index=methods, columns=scenarios)

    fig, ax = plt.subplots(figsize=(max(9, len(scenarios) * 0.85), max(3, len(methods) * 0.9)))
    vmin, vmax = pivot.stack().min(), pivot.stack().max()
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    for i in range(len(methods)):
        for j in range(len(scenarios)):
            v = pivot.iloc[i, j]
            if np.isnan(v):
                continue
            text_color = "white" if abs(v - (vmin + vmax) / 2) > (vmax - vmin) * 0.3 else "black"
            ax.text(j, i, fmt_str.format(v), ha="center", va="center", fontsize=8, color=text_color, fontweight="bold")

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([_slabel(s) for s in scenarios], rotation=30, ha="right")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([_label(m) for m in methods])
    ax.set_title(title, fontweight="bold", pad=8)
    plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    fig.tight_layout()

    _savefig(fig, out_prefix, fmt, dpi)


# ---------------------------------------------------------------------------
# Figure 7 – Win-rate bar chart
# ---------------------------------------------------------------------------


def _win_rate_bar(df, methods, col, higher_is_better, fmt, dpi, out_prefix):
    from itertools import combinations

    wins = dict.fromkeys(methods, 0)
    matches = dict.fromkeys(methods, 0)

    for (_sc, _seed), grp in df.groupby(["scenario", "seed"]):
        vals = grp.set_index("method")[col].to_dict()
        for m_a, m_b in combinations(methods, 2):
            if m_a not in vals or m_b not in vals:
                continue
            v_a, v_b = vals.get(m_a), vals.get(m_b)
            if v_a is None or v_b is None or np.isnan(v_a) or np.isnan(v_b):
                continue
            matches[m_a] += 1
            matches[m_b] += 1
            if (higher_is_better and v_a > v_b) or (not higher_is_better and v_a < v_b):
                wins[m_a] += 1
            elif (higher_is_better and v_b > v_a) or (not higher_is_better and v_b < v_a):
                wins[m_b] += 1
            else:
                wins[m_a] += 0.5
                wins[m_b] += 0.5

    rates = {m: wins[m] / matches[m] if matches[m] > 0 else 0.0 for m in methods}
    sorted_methods = sorted(methods, key=lambda m: -rates[m])

    fig, ax = plt.subplots(figsize=(5, 3.2))
    xs = range(len(sorted_methods))
    bars = ax.bar(
        xs,
        [rates[m] for m in sorted_methods],
        color=[_color(m) for m in sorted_methods],
        alpha=0.85,
        width=0.55,
        edgecolor="white",
        linewidth=0.5,
    )
    for bar, m in zip(bars, sorted_methods, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{rates[m]:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.axhline(0.5, color="0.4", linewidth=0.8, linestyle="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([_label(m) for m in sorted_methods], rotation=12, ha="right")
    ax.set_ylabel("Head-to-head win rate")
    ax.set_ylim(0, min(1.0, max(rates.values()) + 0.12))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title("Global head-to-head win rate (Pearson r)", fontweight="bold", pad=8)
    ax.grid(axis="y", linewidth=0.4, color="0.88", zorder=0)

    _savefig(fig, out_prefix, fmt, dpi)


# ---------------------------------------------------------------------------
# Figure 8 – Rank jitter strip per scenario
# ---------------------------------------------------------------------------


def _rank_jitter(df, methods, scenarios, col, higher_is_better, fmt, dpi, out_prefix):
    rng = np.random.default_rng(0)
    n_sc = len(scenarios)

    fig, ax = plt.subplots(figsize=(max(10, n_sc * 1.1), 4))

    for i, sc in enumerate(scenarios):
        sub = df[df["scenario"] == sc].copy()
        sub["__rank__"] = sub.groupby("seed")[col].transform(
            lambda s: s.rank(ascending=not higher_is_better, method="min")
        )
        for _j, m in enumerate(methods):
            vals = sub[sub["method"] == m]["__rank__"].dropna().to_numpy()
            if vals.size == 0:
                continue
            jitter = rng.uniform(-0.15, 0.15, size=vals.size)
            ax.scatter(
                np.full(vals.size, i) + jitter,
                vals,
                color=_color(m),
                alpha=0.55,
                s=18,
                zorder=3,
                label=_label(m) if i == 0 else None,
            )
            ax.plot(
                [i - 0.22, i + 0.22],
                [np.mean(vals), np.mean(vals)],
                color=_color(m),
                linewidth=2.2,
                zorder=4,
            )

    ax.set_xticks(range(n_sc))
    ax.set_xticklabels([_slabel(s) for s in scenarios], rotation=0)
    ax.set_ylabel("Rank (1 = best)")
    ax.invert_yaxis()
    ax.set_yticks(range(1, len(methods) + 1))
    ax.set_title("Per-seed rank on Pearson r (lower = better, bar = mean)", fontweight="bold", pad=8)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(axis="y", linewidth=0.4, color="0.88", zorder=0)

    _savefig(fig, out_prefix, fmt, dpi)


# ---------------------------------------------------------------------------
# Figure 9 – 2×2 overview grid
# ---------------------------------------------------------------------------


def _overview_grid(df, methods, scenarios, fmt, dpi, out_prefix):
    panels = [
        ("aligned_global_pearson_r", "Pearson r ↑", True, "#2166ac"),
        ("aligned_global_rmse", "RMSE ↓", False, "#d73027"),
        ("delta_pearson_r", "ΔPearson r ↑", True, "#4dac26"),
        ("delta_rmse", "ΔRMSE ↓", False, "#f1a340"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    axes = axes.ravel()

    for ax, (col, ylabel, _hib, _accent) in zip(axes, panels, strict=True):
        n_m = len(methods)
        width = 0.7 / n_m
        offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * width

        for i, sc in enumerate(scenarios):
            sub = df[df["scenario"] == sc]
            for j, m in enumerate(methods):
                vals = sub[sub["method"] == m][col].dropna().to_numpy()
                if vals.size == 0:
                    continue
                bp = ax.boxplot(
                    vals,
                    positions=[i + offsets[j]],
                    widths=width * 0.88,
                    patch_artist=True,
                    notch=False,
                    showfliers=False,
                    medianprops={"color": "white", "linewidth": 1.6},
                    whiskerprops={"linewidth": 0.8, "color": _color(m)},
                    capprops={"linewidth": 0.8, "color": _color(m)},
                    boxprops={"linewidth": 0.0},
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(_color(m))
                    patch.set_alpha(0.80)

        if col.startswith("delta_"):
            ax.axhline(0, color="0.4", linewidth=0.8, linestyle="--", zorder=0)

        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels([_slabel(s) for s in scenarios], rotation=30, ha="right", fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontweight="bold", fontsize=10, pad=5)
        ax.grid(axis="y", linewidth=0.4, color="0.88", zorder=0)

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=_color(m), alpha=0.82, label=_label(m)) for m in methods]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(methods),
        bbox_to_anchor=(0.5, 1.01),
        framealpha=0.95,
        fontsize=9,
    )
    fig.suptitle("Benchmark comparison: all scenarios", fontsize=12, fontweight="bold", y=1.04)

    _savefig(fig, out_prefix, fmt, dpi)


# ---------------------------------------------------------------------------
# Figure 10 – Paired scatter: each method vs. identity baseline
# ---------------------------------------------------------------------------


def _paired_scatter(df, methods, col, ylabel, higher_is_better, fmt, dpi, out_prefix):
    non_baseline = [m for m in methods if m != "identity"]
    if not non_baseline:
        return

    n = len(non_baseline)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2), squeeze=False)

    for ax, m in zip(axes[0], non_baseline, strict=True):
        base = df[df["method"] == "identity"].set_index(["scenario", "seed"])[col]
        other = df[df["method"] == m].set_index(["scenario", "seed"])[col]
        merged = pd.concat([base.rename("baseline"), other.rename("method")], axis=1).dropna()

        x = merged["baseline"].to_numpy()
        y = merged["method"].to_numpy()
        sc_labels = merged.index.get_level_values("scenario").to_numpy()
        unique_sc = _scenario_order(df)
        sc_present = [s for s in unique_sc if s in np.unique(sc_labels)]
        cmap = plt.get_cmap("tab20", len(sc_present))
        sc_color = {s: cmap(i) for i, s in enumerate(sc_present)}

        for sc in sc_present:
            mask = sc_labels == sc
            ax.scatter(
                x[mask], y[mask], color=sc_color[sc], alpha=0.55, s=22, label=_slabel(sc).replace("\n", " "), zorder=3
            )

        lo = min(x.min(), y.min()) - 0.02
        hi = max(x.max(), y.max()) + 0.02
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, zorder=2, label="y = x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"Baseline (identity)\n{ylabel}")
        ax.set_ylabel(f"{_label(m)}\n{ylabel}")
        ax.set_title(f"{_label(m)} vs Baseline", fontweight="bold", pad=6)
        ax.set_aspect("equal")
        ax.grid(linewidth=0.4, color="0.88", zorder=0)

        # Count wins/losses
        if higher_is_better:
            n_win = int((y > x).sum())
        else:
            n_win = int((y < x).sum())
        ax.text(
            0.03, 0.97, f"Wins {n_win}/{len(x)}", transform=ax.transAxes, va="top", ha="left", fontsize=8, color="0.3"
        )

    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="lower center",
        ncol=min(6, len(sc_present)),
        bbox_to_anchor=(0.5, -0.18),
        framealpha=0.9,
        fontsize=7.5,
    )
    fig.suptitle(f"Paired scatter: each method vs baseline ({ylabel})", fontweight="bold", fontsize=11, y=1.01)
    fig.tight_layout()

    _savefig(fig, out_prefix, fmt, dpi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Generate benchmark comparison visualizations for pseudotime alignment methods.

    Loads benchmark results from a specified directory and creates multiple types of
    comparison figures including boxplots, heatmaps, scatter plots, and ranking visualizations.

    Command-line arguments:
        --results_dir (str, required): Path to directory containing benchmark results.
        --output_dir (str, required): Path to directory where output figures will be saved.
        --methods (list, optional): Specific methods to include in plots. If None, includes all.
        --scenarios (list, optional): Specific scenarios to include in plots. If None, includes all.
        --fmt (str, default='pdf'): Output format for figures ('pdf', 'png', or 'svg').
        --dpi (int, default=150): Resolution in dots per inch for output figures.

    Outputs:
        Generates 11 different visualization files comparing method performance:
        1. Boxplot of absolute Pearson r (aligned vs. ground truth)
        2. Boxplot of absolute RMSE (aligned vs. ground truth)
        3. Combined publication panel (Pearson + RMSE)
        4. Boxplot of delta Pearson r (improvement over baseline)
        5. Boxplot of delta RMSE (improvement over baseline)
        6. Heatmap of mean Pearson r by method and scenario
        7. Heatmap of mean RMSE by method and scenario
        8. Win rate bar chart
        9. Rank jitter plot
        10. Overview 2×2 grid
        11. Paired scatter plot comparing Pearson r to baseline
        12. Paired scatter plot comparing RMSE to baseline

    Raises
    ------
        SystemExit: If required arguments are not provided or parsing fails.
    """
    parser = argparse.ArgumentParser(
        description="Plot benchmark comparison figures (Pearson r and RMSE).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--fmt", default="pdf", choices=["pdf", "png", "svg"])
    parser.add_argument("--dpi", default=150, type=int)
    args = parser.parse_args()

    print(f"\n[plot] Loading results from: {args.results_dir}")
    df = load_results(args.results_dir, scenarios=args.scenarios, methods=args.methods)
    print(
        f"[plot] {len(df)} rows  ·  "
        f"methods: {sorted(df['method'].unique())}  ·  "
        f"scenarios: {sorted(df['scenario'].unique())}"
    )

    methods = _method_order(df, args.methods)
    scenarios = _scenario_order(df)
    out = args.output_dir
    fmt = args.fmt
    dpi = args.dpi

    print(f"[plot] method order (best → worst): {methods}")

    print(f"\n[plot] Writing figures → {out}/")

    # 1. Absolute Pearson r
    _boxplot_metric(
        df,
        methods,
        scenarios,
        col="aligned_global_pearson_r",
        ylabel="Pearson r (aligned vs. ground truth)",
        title="Aligned pseudotime quality — Pearson r (higher is better)",
        higher_is_better=True,
        fmt=fmt,
        dpi=dpi,
        out_prefix=os.path.join(out, "boxplot_pearson_r"),
    )

    # 2. Absolute RMSE
    _boxplot_metric(
        df,
        methods,
        scenarios,
        col="aligned_global_rmse",
        ylabel="RMSE (aligned vs. ground truth)",
        title="Aligned pseudotime quality — RMSE (lower is better)",
        higher_is_better=False,
        fmt=fmt,
        dpi=dpi,
        out_prefix=os.path.join(out, "boxplot_rmse"),
    )

    # 2b. Combined publication panel: Pearson (top) + RMSE (bottom)
    _combined_pearson_rmse_boxplot(
        df,
        methods,
        scenarios,
        fmt=fmt,
        dpi=dpi,
        out_prefix=os.path.join(out, "figure2_benchmark_combo"),
    )

    # 3. ΔPearson r
    _boxplot_metric(
        df,
        methods,
        scenarios,
        col="delta_pearson_r",
        ylabel="ΔPearson r (aligned − baseline)",
        title="Improvement over baseline — ΔPearson r (higher is better)",
        higher_is_better=True,
        fmt=fmt,
        dpi=dpi,
        out_prefix=os.path.join(out, "delta_pearson_r"),
    )

    # 4. ΔRMSE
    _boxplot_metric(
        df,
        methods,
        scenarios,
        col="delta_rmse",
        ylabel="ΔRMSE (aligned − baseline)",
        title="Improvement over baseline — ΔRMSE (lower is better; negative = improved)",
        higher_is_better=False,
        fmt=fmt,
        dpi=dpi,
        out_prefix=os.path.join(out, "delta_rmse"),
    )

    # 5. Heatmap Pearson r
    _heatmap(
        df,
        methods,
        scenarios,
        col="aligned_global_pearson_r",
        title="Mean Pearson r — method × scenario",
        fmt_str="{:.3f}",
        cmap="YlGn",
        fmt=fmt,
        dpi=dpi,
        out_prefix=os.path.join(out, "heatmap_pearson_r"),
    )

    # 6. Heatmap RMSE
    _heatmap(
        df,
        methods,
        scenarios,
        col="aligned_global_rmse",
        title="Mean RMSE — method × scenario",
        fmt_str="{:.3f}",
        cmap="YlOrRd_r",
        fmt=fmt,
        dpi=dpi,
        out_prefix=os.path.join(out, "heatmap_rmse"),
    )

    # 7. Win rate bar
    _win_rate_bar(
        df,
        methods,
        col="aligned_global_pearson_r",
        higher_is_better=True,
        fmt=fmt,
        dpi=dpi,
        out_prefix=os.path.join(out, "win_rate_bar"),
    )

    # 8. Rank jitter
    _rank_jitter(
        df,
        methods,
        scenarios,
        col="aligned_global_pearson_r",
        higher_is_better=True,
        fmt=fmt,
        dpi=dpi,
        out_prefix=os.path.join(out, "rank_jitter"),
    )

    # 9. Overview 2×2 grid
    _overview_grid(
        df,
        methods,
        scenarios,
        fmt=fmt,
        dpi=dpi,
        out_prefix=os.path.join(out, "overview_grid"),
    )

    # 10. Paired scatter vs baseline (Pearson r)
    _paired_scatter(
        df,
        methods,
        col="aligned_global_pearson_r",
        ylabel="Pearson r",
        higher_is_better=True,
        fmt=fmt,
        dpi=dpi,
        out_prefix=os.path.join(out, "scatter_pearson_r"),
    )

    # 11. Paired scatter vs baseline (RMSE)
    _paired_scatter(
        df,
        methods,
        col="aligned_global_rmse",
        ylabel="RMSE",
        higher_is_better=False,
        fmt=fmt,
        dpi=dpi,
        out_prefix=os.path.join(out, "scatter_rmse"),
    )

    print(f"\n[plot] Done. {len(os.listdir(out))} files in {out}/\n")


if __name__ == "__main__":
    main()
