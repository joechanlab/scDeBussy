import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cross_batch_purity_sweep(adata, uns_key="purity_sweep", ax=None, figsize=(6, 4)):
    """Plot the stability of cross-batch purity across different neighborhood sizes."""
    if uns_key not in adata.uns:
        raise ValueError(f"'{uns_key}' not found in adata.uns. Run tl.cross_batch_purity_sweep first.")

    scores_dict = adata.uns[uns_key]
    k_values = list(scores_dict.keys())
    scores = list(scores_dict.values())

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(k_values, scores, marker="o", linestyle="-", color="#D95319", linewidth=2)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Number of Nearest Neighbors (k)", fontweight="bold")
    ax.set_ylabel("Cross-Batch Purity Score", fontweight="bold")
    ax.set_title("Purity Stability Across Neighborhood Sizes")
    ax.grid(True, alpha=0.3)

    # Despine for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if ax is None:
        plt.tight_layout()
        return fig
    return ax


def plot_sweep_history(sweep_df, objective_col="objective_score", ax=None, figsize=(8, 4)):
    """Plot tuning objective trajectory across trials."""
    if not isinstance(sweep_df, pd.DataFrame):
        raise TypeError("sweep_df must be a pandas DataFrame.")
    if "trial" not in sweep_df.columns:
        raise ValueError("sweep_df must contain a 'trial' column.")
    if objective_col not in sweep_df.columns:
        raise ValueError(f"sweep_df must contain an {objective_col!r} column.")

    ok = sweep_df[sweep_df.get("status", "ok") == "ok"].copy()
    ok = ok.dropna(subset=[objective_col])
    if ok.empty:
        raise ValueError("No successful trials with finite objective values were found.")

    ok = ok.sort_values("trial")
    created_ax = ax is None
    if created_ax:
        fig, ax = plt.subplots(figsize=figsize)

    x = ok["trial"].to_numpy(dtype=float)
    y = ok[objective_col].to_numpy(dtype=float)
    best_idx = int(np.argmin(y))

    ax.plot(x, y, marker="o", linewidth=1.5, markersize=4, color="#1f77b4", label="Trial objective")
    ax.scatter([x[best_idx]], [y[best_idx]], color="#2ca02c", zorder=3, label="Best trial")
    ax.set_xlabel("Trial")
    ax.set_ylabel(objective_col)
    ax.set_title("Parameter Search Trajectory")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    if created_ax:
        plt.tight_layout()
        return fig
    return ax


def plot_runtime_performance_tradeoff(
    ranking_df,
    *,
    runtime_col="runtime_s",
    score_col="aligned_global_pearson_r",
    method_col="method",
    ax=None,
    figsize=(7, 5),
):
    """Plot method runtime versus performance as a tradeoff scatter."""
    if not isinstance(ranking_df, pd.DataFrame):
        raise TypeError("ranking_df must be a pandas DataFrame.")
    for col in (runtime_col, score_col, method_col):
        if col not in ranking_df.columns:
            raise ValueError(f"ranking_df must contain {col!r}.")

    data = ranking_df[[method_col, runtime_col, score_col]].dropna().copy()
    if data.empty:
        raise ValueError("No rows with complete runtime/performance values were found.")

    created_ax = ax is None
    if created_ax:
        fig, ax = plt.subplots(figsize=figsize)

    x = data[runtime_col].to_numpy(dtype=float)
    y = data[score_col].to_numpy(dtype=float)
    methods = data[method_col].astype(str).tolist()

    ax.scatter(x, y, s=60, alpha=0.85, color="#1f77b4")
    for xi, yi, m in zip(x, y, methods, strict=False):
        ax.annotate(m, (xi, yi), textcoords="offset points", xytext=(5, 4), fontsize=8)

    ax.set_xscale("log")
    ax.set_xlabel("Runtime (s, log scale)")
    ax.set_ylabel(score_col)
    ax.set_title("Runtime vs Performance Tradeoff")
    ax.grid(True, alpha=0.3)

    if created_ax:
        plt.tight_layout()
        return fig
    return ax
