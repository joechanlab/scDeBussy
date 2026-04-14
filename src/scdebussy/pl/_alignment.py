import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


def plot_barycenter_boundaries(adata, barycenter_key="barycenter", ax=None, figsize=(10, 4)):
    """
    Visualize barycenter boundaries

    Visualize the transcriptomic velocity of the global barycenter and the
    automatically detected segment boundaries.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing the detected boundaries.
    barycenter_key : str, optional
        Key in `adata.uns` where the barycenter and boundaries are stored.
    ax : matplotlib.axes.Axes, optional
        Axes object to draw the plot onto, otherwise creates a new figure.
    figsize : tuple, optional
        Size of the figure.
    """
    if barycenter_key not in adata.uns or "auto_boundaries" not in adata.uns[barycenter_key]:
        raise ValueError("Boundary data not found. Run scdebussy.tl.detect_barycenter_boundaries first.")

    bound_data = adata.uns[barycenter_key]["auto_boundaries"]
    bary_time = adata.uns[barycenter_key]["aligned_pseudotime"]

    velocity = bound_data["velocity_curve"]
    peaks = bound_data["peak_indices"]
    segments = bound_data["segments"]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # 1. Plot the continuous velocity curve
    ax.plot(bary_time, velocity, color="black", linewidth=2, label="Transcriptomic Velocity")

    # 2. Draw the vertical boundary lines at the peaks
    for i, peak in enumerate(peaks):
        label = "Detected Boundary" if i == 0 else ""
        ax.axvline(x=bary_time[peak], color="#D95319", linestyle="--", linewidth=2, label=label)

    # 3. Shade the segments with alternating background colors
    for i, seg in enumerate(segments):
        start_t = seg["start_time"]
        end_t = seg["end_time"]
        bg_color = "#F0F0F0" if i % 2 == 0 else "#FFFFFF"
        ax.axvspan(start_t, end_t, facecolor=bg_color, alpha=0.5, zorder=0)
        mid_point = start_t + (end_t - start_t) / 2
        ax.text(
            mid_point,
            np.max(velocity) * 0.95,
            f"Seg {i + 1}",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color="#555555",
        )

    # 4. Aesthetics
    ax.set_xlabel("Global Barycenter Pseudotime", fontsize=12, fontweight="bold")
    ax.set_ylabel("Rate of Change (Velocity)", fontsize=12, fontweight="bold")
    ax.set_title("Automatic Boundary Detection via Transcriptomic Velocity", fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    if ax is None:
        plt.tight_layout()
        return fig
    return ax


def plot_em_convergence(adata, barycenter_key="barycenter", ax=None, figsize=(5, 4)):
    """Plot EM objective/cost across iterations for scDeBussy alignment."""
    if barycenter_key not in adata.uns:
        raise ValueError(f"{barycenter_key!r} not found in adata.uns.")

    bary_meta = adata.uns[barycenter_key]
    if not isinstance(bary_meta, dict) or "em_convergence" not in bary_meta:
        raise ValueError("EM convergence data not found. Run scDeBussy with updated convergence metadata enabled.")

    conv = bary_meta["em_convergence"]
    if not isinstance(conv, dict):
        raise ValueError("Invalid em_convergence payload in adata.uns.")

    costs = np.asarray(conv.get("iteration_costs", []), dtype=float)
    if costs.size == 0:
        raise ValueError("em_convergence.iteration_costs is empty.")

    created_ax = ax is None
    if created_ax:
        fig, ax = plt.subplots(figsize=figsize)

    iterations = np.arange(1, costs.size + 1)
    ax.plot(iterations, costs, marker="o", color="#1f77b4", linewidth=1.8, markersize=4)
    ax.set_xlabel("EM Iteration")
    ax.set_ylabel("Total Alignment Cost")
    ax.set_title("scDeBussy EM Convergence")
    ax.grid(True, alpha=0.3)

    final_iter = int(conv.get("final_iteration", costs.size))
    converged = bool(conv.get("converged", False))
    final_cost = float(costs[min(final_iter, costs.size) - 1])
    marker_color = "#2ca02c" if converged else "#d62728"
    ax.scatter([min(final_iter, costs.size)], [final_cost], color=marker_color, zorder=3)
    ax.text(
        min(final_iter, costs.size),
        final_cost,
        " converged" if converged else " max_iter",
        va="bottom",
        ha="left",
        fontsize=9,
        color=marker_color,
    )

    if created_ax:
        plt.tight_layout()
        return fig
    return ax


def plot_running_enrichment_ridge(
    enrichment_df: pd.DataFrame,
    gene_sets_to_plot: list,
    ordered_genes,
    *,
    gene_curve=None,
    cutoff_points=(1, 2),
    figsize=(8, 10),
    smooth_window: int = 10,
    overlap: float = 0.5,
    short_names=None,
    transition_window=None,
    transition_window_color: str = "#FDEBD0",
    transition_window_alpha: float = 0.5,
    save_path=None,
):
    """Plot stacked ridge plots of running gene-set enrichment along ordered genes.

    Each ridge corresponds to one gene set; the fill colour is drawn from the
    ``viridis`` palette.  An optional top annotation bar marks early / mid / late
    gene phases when ``gene_curve`` is supplied.

    Parameters
    ----------
    enrichment_df : pd.DataFrame
        Output of :func:`scdebussy.tl.compute_running_enrichment` with columns
        ``gene_set``, ``position``, ``score``.
    gene_sets_to_plot : list of str
        Ordered list of gene-set names to show (top to bottom).
    ordered_genes : list or array-like of str
        Gene list in temporal order used when computing ``enrichment_df``.
    gene_curve : pd.DataFrame, optional
        Barycenter expression DataFrame whose first column is pseudotime and
        remaining columns are genes.  When supplied a phase-annotation bar is
        drawn above the ridges; when ``None`` the bar is omitted.
    cutoff_points : tuple of two floats, optional
        Pseudotime cutoffs separating early / mid / late phases.
        Only used when ``gene_curve`` is provided.  Defaults to ``(1, 2)``.
    figsize : tuple of two floats, optional
        Figure size ``(width, height)``.  Defaults to ``(8, 10)``.
    smooth_window : int, optional
        Width (in genes) for uniform smoothing of each enrichment curve.
        Defaults to 10.
    overlap : float, optional
        Fractional overlap between adjacent ridge axes.  Defaults to ``0.5``.
    short_names : list of str, optional
        Short display names for each gene set (same order as
        ``gene_sets_to_plot``).  Defaults to cleaned versions of the set names.
    transition_window : tuple of two floats or None, optional
        Pseudotime interval ``(start, end)`` to mark with vertical boundary
        lines on each ridge. Requires ``gene_curve`` to map pseudotime to gene
        positions. Defaults to ``None`` (no transition markers).
    transition_window_color : str, optional
        Color used for the transition window boundary lines. Defaults to
        ``"#FDEBD0"``.
    transition_window_alpha : float, optional
        Alpha value for the transition window boundary lines. Defaults to ``0.5``.
    save_path : str or None, optional
        File path to save the figure.  Defaults to ``None`` (no save).

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure.
    """
    ordered_genes = list(ordered_genes)
    n_sets = len(gene_sets_to_plot)
    n_genes = len(ordered_genes)

    if transition_window is not None and gene_curve is None:
        raise ValueError("transition_window requires gene_curve to map pseudotime to gene positions.")

    if transition_window is not None:
        if len(transition_window) != 2:
            raise ValueError("transition_window must contain exactly two values: (start, end).")
        transition_start, transition_end = float(transition_window[0]), float(transition_window[1])
        if not (np.isfinite(transition_start) and np.isfinite(transition_end)):
            raise ValueError("transition_window must contain finite numeric values.")
        if transition_start > transition_end:
            raise ValueError("transition_window start must be <= end.")
    else:
        transition_start, transition_end = None, None

    gene_peak_pseudotime = None
    if gene_curve is not None:
        pseudotime = np.asarray(gene_curve.iloc[:, 0].values, dtype=float)
        gene_peak_pseudotime = np.full(n_genes, np.nan, dtype=float)
        for i, gene in enumerate(ordered_genes):
            if gene in gene_curve.columns:
                peak_idx = int(np.argmax(gene_curve[gene].values))
                gene_peak_pseudotime[i] = float(pseudotime[peak_idx])

    transition_gene_span = None
    if transition_start is not None and gene_peak_pseudotime is not None:
        in_window = np.where((gene_peak_pseudotime >= transition_start) & (gene_peak_pseudotime <= transition_end))[0]
        if in_window.size > 0:
            transition_gene_span = (float(in_window.min()), float(in_window.max() + 1))

    has_bar = gene_curve is not None
    n_rows = n_sets + (1 if has_bar else 0)
    height_ratios = ([0.4] if has_bar else []) + [1] * n_sets

    fig = plt.figure(figsize=figsize)
    gs_layout = gridspec.GridSpec(n_rows, 1, height_ratios=height_ratios, figure=fig)
    gs_layout.update(hspace=0)
    axes = [fig.add_subplot(gs_layout[i]) for i in range(n_rows)]

    bar_ax = axes[0] if has_bar else None
    ridge_axes = axes[1:] if has_bar else axes

    # ── phase annotation bar ─────────────────────────────────────────────────
    if has_bar:
        from matplotlib.colors import ListedColormap

        assert bar_ax is not None
        assert gene_peak_pseudotime is not None

        gene_phases = np.zeros(n_genes, dtype=int)
        for i, peak_pt in enumerate(gene_peak_pseudotime):
            if np.isfinite(peak_pt):
                if peak_pt < cutoff_points[0]:
                    gene_phases[i] = 0
                elif peak_pt < cutoff_points[1]:
                    gene_phases[i] = 1
                else:
                    gene_phases[i] = 2

        purples_cmap = plt.get_cmap("Purples")
        colors_list = [tuple(c) for c in purples_cmap(np.linspace(0.45, 0.9, 3))]
        phase_cmap = ListedColormap(colors_list)
        bar_ax.pcolormesh(
            np.arange(n_genes + 1),
            [0, 1],
            gene_phases.reshape(1, -1),
            cmap=phase_cmap,
            shading="auto",
        )
        bar_ax.set_yticks([])
        bar_ax.set_xlim(0, n_genes)
        bar_ax.set_xticks([])
        for s in bar_ax.spines.values():
            s.set_visible(False)
        for idx, phase_label in enumerate(["Early", "Mid", "Late"]):
            mask = np.where(gene_phases == idx)[0]
            if len(mask) > 0:
                bar_ax.text(
                    float(mask.mean()),
                    1.3,
                    phase_label,
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color=colors_list[idx],
                    clip_on=False,
                )

    # ── ridge plots ──────────────────────────────────────────────────────────
    fill_colors = plt.cm.viridis(np.linspace(0, 0.8, n_sets))

    for i, (ax_r, gs_name) in enumerate(zip(ridge_axes, gene_sets_to_plot, strict=False)):
        subset = enrichment_df[enrichment_df["gene_set"] == gs_name].sort_values("position")
        if subset.empty:
            ax_r.axis("off")
            continue

        if i > 0:
            pos = ax_r.get_position()
            ax_r.set_position([pos.x0, pos.y0 + overlap * i * 0.05, pos.width, pos.height])

        x = subset["position"].values
        y_raw = subset["score"].values
        y = uniform_filter1d(y_raw, size=smooth_window)
        ridge_fill_zorder = i * 3 + 1
        ridge_line_zorder = ridge_fill_zorder + 1
        transition_zorder = ridge_line_zorder + 1

        if transition_gene_span is not None:
            ax_r.axvline(
                transition_gene_span[0],
                color=transition_window_color,
                alpha=transition_window_alpha,
                linewidth=1.5,
                zorder=transition_zorder,
            )
            ax_r.axvline(
                transition_gene_span[1],
                color=transition_window_color,
                alpha=transition_window_alpha,
                zorder=transition_zorder,
                linewidth=1.5,
            )

        ax_r.plot(x, y, color="white", linewidth=1.5, alpha=0.9, zorder=ridge_line_zorder)
        ax_r.fill_between(x, 0, y, color=fill_colors[i], alpha=0.8, zorder=ridge_fill_zorder)
        ax_r.set_ylim(bottom=0)
        ax_r.set_xlim(0, n_genes)
        ax_r.patch.set_alpha(0)
        ax_r.set_yticks([])
        for side in ["top", "right", "left", "bottom"]:
            ax_r.spines[side].set_visible(False)
        ax_r.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        if short_names is not None:
            display_label = short_names[i]
        else:
            display_label = (
                gs_name.replace(" Human", "")
                .replace(" Mouse", "")
                .split(" (")[0]
                .replace("_", " ")
                .replace("all ", "")
                .replace("-", " ")
            )
        ax_r.text(
            -0.02,
            0.1,
            display_label,
            transform=ax_r.transAxes,
            fontsize=10,
            fontweight="bold",
            ha="right",
            va="bottom",
        )

    # ── bottom x-axis ────────────────────────────────────────────────────────
    bottom_ax = ridge_axes[-1]
    bottom_ax.spines["bottom"].set_visible(True)
    bottom_ax.set_xlabel("Gene Position", fontsize=12, fontweight="bold")
    bottom_ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
