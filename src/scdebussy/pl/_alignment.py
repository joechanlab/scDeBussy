import warnings

import matplotlib.colors as mcolors
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


def _compute_patient_coverage(adata, patient_key: str, pseudotime_key: str, n_bins: int) -> np.ndarray:
    """Return shape-(n_bins,) array of fraction of patients with ≥1 cell per pseudotime bin.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with aligned pseudotime and patient labels in ``.obs``.
    patient_key : str
        Column in ``adata.obs`` identifying patients.
    pseudotime_key : str
        Column in ``adata.obs`` containing aligned pseudotime values in ``[0, 1]``.
    n_bins : int
        Number of uniform bins covering ``[0, 1]``.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_bins,)`` with values in ``[0, 1]``.
    """
    pt = np.asarray(adata.obs[pseudotime_key].values, dtype=float)
    patients = adata.obs[patient_key].values
    n_patients = len(np.unique(patients))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    coverage = np.zeros(n_bins, dtype=float)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (pt >= lo) & (pt <= hi if b == n_bins - 1 else pt < hi)
        n_contributing = len(np.unique(patients[mask]))
        coverage[b] = n_contributing / n_patients if n_patients > 0 else 0.0
    return coverage


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
    adata=None,
    patient_key: str = "patient",
    aligned_pseudotime_key: str = "aligned_pseudotime",
    n_coverage_bins: int = 50,
    coverage_cmap: str = "YlGn",
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
    adata : AnnData or None, optional
        If provided, a patient-coverage bar is drawn below the ridges.  Requires
        ``gene_curve`` to map pseudotime to gene positions.  Defaults to ``None``.
    patient_key : str, optional
        Column in ``adata.obs`` identifying patients.  Defaults to ``"patient"``.
    aligned_pseudotime_key : str, optional
        Column in ``adata.obs`` containing aligned pseudotime in ``[0, 1]``.
        Defaults to ``"aligned_pseudotime"``.
    n_coverage_bins : int, optional
        Number of uniform pseudotime bins used when computing patient coverage.
        Defaults to ``50``.
    coverage_cmap : str, optional
        Matplotlib colormap name for the patient-coverage bar.
        Defaults to ``"YlGn"``.
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

    if adata is not None and gene_curve is None:
        raise ValueError("adata requires gene_curve to map patient coverage to gene positions.")

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
    has_coverage = adata is not None  # gene_curve already validated above when adata is not None
    n_rows = n_sets + (1 if has_bar else 0) + (1 if has_coverage else 0)
    height_ratios = ([0.4] if has_bar else []) + [1] * n_sets + ([0.3] if has_coverage else [])

    fig = plt.figure(figsize=figsize)
    gs_layout = gridspec.GridSpec(n_rows, 1, height_ratios=height_ratios, figure=fig)
    gs_layout.update(hspace=0)
    axes = [fig.add_subplot(gs_layout[i]) for i in range(n_rows)]

    bar_ax = axes[0] if has_bar else None
    ridge_start = 1 if has_bar else 0
    ridge_axes = axes[ridge_start : ridge_start + n_sets]
    coverage_ax = axes[-1] if has_coverage else None

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

    # ── bottom x-axis (on last ridge when no coverage bar, else on coverage bar) ───
    if not has_coverage:
        bottom_ax = ridge_axes[-1]
        bottom_ax.spines["bottom"].set_visible(True)
        bottom_ax.set_xlabel("Gene Position", fontsize=12, fontweight="bold")
        bottom_ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True)

    # ── patient coverage bar ─────────────────────────────────────────────────
    if has_coverage:
        assert coverage_ax is not None
        assert gene_peak_pseudotime is not None

        _cov = _compute_patient_coverage(adata, patient_key, aligned_pseudotime_key, n_coverage_bins)
        cov_edges = np.linspace(0.0, 1.0, n_coverage_bins + 1)
        gene_coverage = np.zeros(n_genes, dtype=float)
        for i in range(n_genes):
            tau = gene_peak_pseudotime[i]
            if np.isfinite(tau):
                b = min(int(np.searchsorted(cov_edges[1:], tau)), n_coverage_bins - 1)
                gene_coverage[i] = _cov[b]

        coverage_ax.pcolormesh(
            np.arange(n_genes + 1),
            [0, 1],
            gene_coverage.reshape(1, -1),
            cmap=plt.get_cmap(coverage_cmap),
            vmin=0.0,
            vmax=1.0,
            shading="auto",
        )
        coverage_ax.set_yticks([0.5])
        coverage_ax.set_yticklabels(["% Patients"], fontsize=8)
        coverage_ax.set_xlim(0, n_genes)
        coverage_ax.set_xlabel("Gene Position", fontsize=12, fontweight="bold")
        coverage_ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True)
        for s in coverage_ax.spines.values():
            s.set_visible(False)
        coverage_ax.spines["bottom"].set_visible(True)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_gene_barycenter(
    adata,
    genes: list,
    *,
    barycenter_key: str = "barycenter",
    aligned_pseudotime_key: str = "aligned_pseudotime",
    cell_type_key: str | None = None,
    patient_key: str | None = None,
    cell_type_colors: dict | None = None,
    phase_segments: list | None = None,
    cell_alpha: float = 0.15,
    cell_size: float = 2.0,
    patient_palette: str = "tab10",
    line_width: float = 2.0,
    ylim: "tuple | dict | None" = None,
    show_patient_legend: bool = True,
    smooth_window: int = 0,
    figsize: tuple | None = None,
    save_path: str | None = None,
    cell_type_min_patients: int = 1,
) -> "plt.Figure":
    """Plot barycenter gene expression curves against aligned pseudotime.

    For each requested gene a panel shows individual cells as background scatter
    points coloured by patient, with the barycenter consensus curve drawn on top.
    An optional shared cell-type annotation strip sits above all gene panels.

    Parameters
    ----------
    adata : AnnData
        Aligned AnnData with barycenter stored in ``adata.uns[barycenter_key]``.
    genes : list of str
        Gene names to plot (e.g. ``["ASCL1", "NEUROD1"]``).
    barycenter_key : str, optional
        Key in ``adata.uns`` for the barycenter. Default ``"barycenter"``.
    aligned_pseudotime_key : str, optional
        Column in ``adata.obs`` with aligned pseudotime in ``[0, 1]``.
        Default ``"aligned_pseudotime"``.
    cell_type_key : str or None, optional
        Column in ``adata.obs`` for cell-type labels drawn in the annotation bar.
        When ``None`` the bar is omitted.
    patient_key : str or None, optional
        Column in ``adata.obs`` identifying patients for scatter colouring.
        When ``None`` all cells are plotted in a single neutral colour.
    cell_type_colors : dict or None, optional
        Mapping ``{cell_type: color}`` for the annotation bar.
        Inferred from ``tab10`` when ``None``.
    phase_segments : list of (int, int, str) or None, optional
        Explicit ``(start_bin, end_bin, label)`` tuples — using the same bin
        indices as the barycenter — to colour and label the annotation bar.
        Each bin is assigned the colour of the segment it falls in.
        When ``None``, colours are derived from the per-bin dominant cell type
        and boundaries are inferred automatically from transitions.
    cell_alpha : float, optional
        Transparency for scatter points. Default ``0.15``.
    cell_size : float, optional
        Marker size for scatter points. Default ``2.0``.
    patient_palette : str, optional
        Matplotlib colormap name for patient colours. Default ``"tab10"``.
    line_width : float, optional
        Line width for the barycenter curve. Default ``2.0``.
    ylim : tuple, dict, or None, optional
        Y-axis limits for the gene panels.  Accepts:

        - ``None`` (default): automatic limits.
        - ``(ymin, ymax)``: the same limits applied to every gene panel.
        - ``{gene: (ymin, ymax)}``: per-gene limits; genes not in the dict
          use automatic limits.
    show_patient_legend : bool, optional
        Whether to draw the patient colour legend. Default ``True``.
    smooth_window : int, optional
        Width of the uniform smoothing kernel applied to the barycenter curve
        before plotting.  ``0`` or ``1`` means no smoothing.  Values around
        ``5``–``15`` remove bin-to-bin EM noise while preserving broad trends.
        Default ``0``.
    figsize : tuple or None, optional
        Figure size ``(width, height)``. Inferred from gene count when ``None``.
    save_path : str or None, optional
        File path to save the figure. Default ``None`` (no save).
    cell_type_min_patients : int, optional
        Minimum number of distinct patients that must have cells in a barycenter
        bin for that bin to receive a cell-type label in the annotation bar.
        Bins backed by fewer patients are shown as "Unassigned" (grey).  Requires
        ``patient_key`` to be set; ignored when ``patient_key`` is ``None`` or
        ``phase_segments`` is provided.  Default ``1`` (no filtering).

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure.
    """
    import scipy.sparse as sp

    if barycenter_key not in adata.uns:
        raise ValueError(f"{barycenter_key!r} not found in adata.uns.")

    bary = adata.uns[barycenter_key]
    bary_expr = np.asarray(bary["expression"], dtype=float)  # (n_bins, n_genes)
    n_bins = bary_expr.shape[0]

    gene_names = list(adata.var_names.astype(str))
    gene_index = {g: i for i, g in enumerate(gene_names)}

    valid_genes = [g for g in genes if g in gene_index]
    missing = [g for g in genes if g not in gene_index]
    if missing:
        warnings.warn(f"Genes not found in adata.var_names: {missing}", stacklevel=2)
    if not valid_genes:
        raise ValueError("None of the requested genes are present in adata.var_names.")

    n_genes = len(valid_genes)
    has_bar = cell_type_key is not None

    # ── figure layout ─────────────────────────────────────────────────────────
    if figsize is None:
        figsize = (5.0, 2.2 * n_genes + (0.5 if has_bar else 0.0))

    height_ratios = ([0.15] if has_bar else []) + [1.0] * n_genes
    n_rows = len(height_ratios)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_rows, 1, height_ratios=height_ratios, figure=fig, hspace=0.5)

    gene_offset = 1 if has_bar else 0
    bar_ax = fig.add_subplot(gs[0]) if has_bar else None
    gene_axes = []
    for i in range(n_genes):
        share = bar_ax if (has_bar and i == 0) else (gene_axes[0] if i > 0 else None)
        gene_axes.append(fig.add_subplot(gs[gene_offset + i], sharex=share))

    # ── cell-type annotation bar ──────────────────────────────────────────────
    if has_bar:
        if phase_segments is not None:
            # Assign each bin the label of the segment it falls in.
            label_by_bin = np.full(n_bins, "Unassigned", dtype=object)
            for left_b, right_b, seg_label in phase_segments:
                label_by_bin[left_b : min(right_b + 1, n_bins)] = seg_label
            dominant_by_bin = label_by_bin.tolist()
        else:
            ct_obs = adata.obs[cell_type_key].astype(str).to_numpy()
            aligned_pt_obs = adata.obs[aligned_pseudotime_key].to_numpy(dtype=float)
            valid_mask = np.isfinite(aligned_pt_obs)
            bin_idx_obs = np.clip((aligned_pt_obs[valid_mask] * (n_bins - 1)).astype(int), 0, n_bins - 1)
            ct_valid = ct_obs[valid_mask]
            # Patient labels per cell for the min-patients filter (None if not applicable).
            pat_valid = None
            if patient_key is not None and cell_type_min_patients > 1:
                pat_valid = adata.obs[patient_key].astype(str).to_numpy()[valid_mask]
            dominant_by_bin = []
            for b in range(n_bins):
                bin_mask = bin_idx_obs == b
                labels_b = ct_valid[bin_mask]
                if len(labels_b) == 0:
                    dominant_by_bin.append("Unassigned")
                elif pat_valid is not None and len(np.unique(pat_valid[bin_mask])) < cell_type_min_patients:
                    dominant_by_bin.append("Unassigned")
                else:
                    dominant_by_bin.append(str(pd.Series(labels_b).value_counts().index[0]))

        unique_types = sorted({d for d in dominant_by_bin if d != "Unassigned"})
        if cell_type_colors is None:
            cmap_tab = plt.cm.get_cmap("tab10")
            cell_type_colors = {ct: cmap_tab(i % 10) for i, ct in enumerate(unique_types)}

        bar_rgba = np.array(
            [[plt.matplotlib.colors.to_rgba(cell_type_colors.get(d, "#808080")) for d in dominant_by_bin]]
        )
        bar_ax.imshow(bar_rgba, aspect="auto", extent=[-0.5, n_bins - 0.5, 0.0, 1.0], interpolation="nearest")
        bar_ax.set_xlim(-0.5, n_bins - 0.5)
        bar_ax.set_yticks([])
        bar_ax.tick_params(axis="x", bottom=False, top=False, labelbottom=False, labeltop=False)
        for spine in bar_ax.spines.values():
            spine.set_visible(False)

        if phase_segments is not None:
            for left_b, right_b, seg_label in phase_segments:
                center_bin = (left_b + min(right_b, n_bins - 1)) / 2.0
                bar_ax.text(center_bin, -0.65, seg_label, ha="center", va="bottom", fontsize=9, clip_on=False)
            for _, right_b, _ in phase_segments[:-1]:
                bar_ax.axvline(right_b + 0.5, color="white", linewidth=1.2, alpha=0.95)
        else:
            for t in range(1, n_bins):
                if dominant_by_bin[t] != dominant_by_bin[t - 1]:
                    bar_ax.axvline(t - 0.5, color="white", linewidth=1.0, alpha=0.8)

    # ── patient colour map ────────────────────────────────────────────────────
    patient_labels_all = None
    unique_patients: list = []
    patient_color_map: dict = {}
    if patient_key is not None:
        patient_labels_all = adata.obs[patient_key].astype(str).to_numpy()
        unique_patients = sorted(np.unique(patient_labels_all).tolist())
        n_patients = len(unique_patients)
        pat_cmap = plt.cm.get_cmap(patient_palette)
        patient_color_map = {p: pat_cmap(i / max(n_patients - 1, 1)) for i, p in enumerate(unique_patients)}

    # ── per-gene panels ───────────────────────────────────────────────────────
    # Convert cell pseudotime [0, 1] → bin scale [0, n_bins-1] for a consistent x-axis.
    aligned_pt_all = adata.obs[aligned_pseudotime_key].to_numpy(dtype=float)
    valid_cell_mask = np.isfinite(aligned_pt_all)
    pt_valid_bins = aligned_pt_all[valid_cell_mask] * (n_bins - 1)

    bary_x = np.arange(n_bins, dtype=float)  # bin indices 0 … n_bins-1

    for i, gene in enumerate(valid_genes):
        ax = gene_axes[i]
        g_idx = gene_index[gene]

        col = adata.X[:, g_idx]
        expr_all = np.asarray(col.todense()).ravel() if sp.issparse(col) else np.asarray(col, dtype=float).ravel()
        expr_valid = expr_all[valid_cell_mask]

        # Z-score using barycenter mean/std across bins (same reference as the heatmap).
        bary_y_raw = bary_expr[:, g_idx]
        mu = bary_y_raw.mean()
        sigma = bary_y_raw.std() + 1e-12
        expr_valid = (expr_valid - mu) / sigma
        bary_y = (bary_y_raw - mu) / sigma

        if smooth_window > 1:
            bary_y = uniform_filter1d(bary_y, size=smooth_window)

        if patient_key is not None:
            patients_valid = patient_labels_all[valid_cell_mask]
            for patient in unique_patients:
                pmask = patients_valid == patient
                ax.scatter(
                    pt_valid_bins[pmask],
                    expr_valid[pmask],
                    c=[patient_color_map[patient]],
                    s=cell_size,
                    alpha=cell_alpha,
                    linewidths=0,
                    rasterized=True,
                )
        else:
            ax.scatter(
                pt_valid_bins,
                expr_valid,
                c="#888888",
                s=cell_size,
                alpha=cell_alpha,
                linewidths=0,
                rasterized=True,
            )

        ax.plot(bary_x, bary_y, color="black", linewidth=line_width, zorder=5)

        # y-axis limits
        if isinstance(ylim, dict):
            gene_ylim = ylim.get(gene)
        else:
            gene_ylim = ylim
        if gene_ylim is not None:
            ax.set_ylim(gene_ylim)

        ax.set_title(gene, fontsize=11, fontweight="bold")
        ax.set_ylabel("Z-scored expression", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i < n_genes - 1:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("Aligned pseudotime bin", fontsize=11)

    # ── patient legend ────────────────────────────────────────────────────────
    if show_patient_legend and patient_key is not None and patient_color_map:
        from matplotlib.lines import Line2D

        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=patient_color_map[p], markersize=5, label=p)
            for p in unique_patients
        ]
        gene_axes[-1].legend(
            handles=handles,
            title="Patient",
            bbox_to_anchor=(1.01, 1.0),
            loc="upper left",
            frameon=False,
            fontsize=8,
            title_fontsize=8,
        )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_warp_celltype_map(
    adata,
    cell_type_key: str,
    barycenter_key: str = "barycenter",
    patient_key: str | None = None,
    pseudotime_key: str | None = None,
    cell_type_colors: dict | None = None,
    highlight_gene: str | None = None,
    figsize: tuple | None = None,
    title: str | None = None,
    save_path: str | None = None,
) -> "plt.Figure":
    """Diagnostic heatmap of patient pseudotime bin → barycenter mapping, coloured by majority cell type.

    Each row represents one patient.  Each column is a barycenter bin.  The colour
    at ``(patient, barycenter_bin)`` reflects the majority cell type of the patient
    cells whose local pseudotime mapped to that barycenter position via the DTW warp
    path.  Grey cells indicate barycenter bins not covered by that patient.

    Useful for debugging unexpected gene expression patterns — e.g., a gene that
    should peak in one cell type but appears dominated by another in the barycenter.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix after running ``scDeBussy.fit()``.
    cell_type_key : str
        Key in ``adata.obs`` containing cell type labels.
    barycenter_key : str, optional
        Key in ``adata.uns`` where barycenter results are stored.  Default ``"barycenter"``.
    patient_key : str, optional
        Key in ``adata.obs`` for patient/batch IDs.  Inferred from stored params if ``None``.
    pseudotime_key : str, optional
        Key in ``adata.obs`` for local pseudotime.  Inferred from stored params if ``None``.
    cell_type_colors : dict, optional
        ``{cell_type_label: color}`` mapping.  Falls back to ``adata.uns[f"{cell_type_key}_colors"]``
        when available, otherwise uses ``tab20``.
    highlight_gene : str, optional
        If given, draw the barycenter expression curve for this gene in a panel above the
        heatmap so you can correlate the expression peak with cell-type contributions.
    figsize : tuple, optional
        Figure size ``(width, height)``.  Auto-scaled from patient count when ``None``.
    title : str, optional
        Heatmap title.  A default descriptive title is used when ``None``.
    save_path : str, optional
        If given, save the figure to this path at 150 dpi.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> scdebussy.pl.plot_warp_celltype_map(adata, cell_type_key="cell_type", highlight_gene="CHGA")
    """
    if barycenter_key not in adata.uns:
        raise ValueError(f"{barycenter_key!r} not found in adata.uns. Run scDeBussy first.")

    bary = adata.uns[barycenter_key]
    params = bary.get("params", {})

    if patient_key is None:
        patient_key = params.get("patient_key", "patient")
    if pseudotime_key is None:
        pseudotime_key = params.get("pseudotime_key", "pseudotime")

    n_bins = int(params.get("n_bins", np.asarray(bary["expression"]).shape[0]))
    patient_ids = list(bary["patient_ids"])
    warp_paths = bary["warp_paths"]  # list[list[tuple[int, int]]]

    # ── cell-type colour map ──────────────────────────────────────────────────
    all_celltypes = sorted(adata.obs[cell_type_key].astype(str).unique().tolist())
    if cell_type_colors is None:
        uns_key = f"{cell_type_key}_colors"
        if uns_key in adata.uns:
            cats = list(adata.obs[cell_type_key].astype("category").cat.categories.astype(str))
            cell_type_colors = dict(zip(cats, adata.uns[uns_key], strict=False))
        else:
            cmap_auto = plt.cm.get_cmap("tab20", len(all_celltypes))
            cell_type_colors = {ct: cmap_auto(i) for i, ct in enumerate(all_celltypes)}

    # ── build majority-cell-type grid: (n_patients × n_bins) ─────────────────
    # None means uncovered (no cells map to that barycenter bin for this patient).
    grid: list[list[str | None]] = [[None] * n_bins for _ in range(len(patient_ids))]

    for p_idx, pid in enumerate(patient_ids):
        pat_mask = adata.obs[patient_key].astype(str) == str(pid)
        pat_obs = adata.obs[pat_mask]
        if pat_obs.empty:
            continue

        pt_vals = pat_obs[pseudotime_key].values.astype(float)
        bin_assignments = np.floor(pt_vals * n_bins).astype(int).clip(0, n_bins - 1)
        pat_celltypes = pat_obs[cell_type_key].astype(str).values

        warp = warp_paths[p_idx]  # list of (pat_bin, bary_bin)

        # Group warp path by barycenter bin to collect contributing patient bins.
        bary_to_pat_bins: dict[int, list[int]] = {}
        for pat_bin, bary_bin in warp:
            bary_to_pat_bins.setdefault(bary_bin, []).append(pat_bin)

        for bary_bin, pat_bins in bary_to_pat_bins.items():
            cell_mask = np.isin(bin_assignments, pat_bins)
            if cell_mask.sum() > 0:
                counts = pd.Series(pat_celltypes[cell_mask]).value_counts()
                grid[p_idx][bary_bin] = counts.index[0]

    # ── resolve highlight gene if requested ──────────────────────────────────
    gene_expr: np.ndarray | None = None
    if highlight_gene is not None:
        gene_names = list(adata.var_names.astype(str))
        if highlight_gene not in gene_names:
            warnings.warn(f"Gene {highlight_gene!r} not found in adata.var_names; skipping overlay.", stacklevel=2)
        else:
            bary_expr = np.asarray(bary["expression"], dtype=float)
            g_idx = gene_names.index(highlight_gene)
            gene_expr = bary_expr[:, g_idx]

    # ── figure layout ─────────────────────────────────────────────────────────
    n_patients = len(patient_ids)
    if figsize is None:
        heatmap_h = max(3.0, 0.32 * n_patients)
        gene_h = 1.8 if gene_expr is not None else 0.0
        figsize = (14, heatmap_h + gene_h + 1.5)

    has_gene_panel = gene_expr is not None
    if has_gene_panel:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(
            3,
            1,
            height_ratios=[1.8, max(3.0, 0.32 * n_patients), 1.0],
            hspace=0.12,
            figure=fig,
        )
        ax_gene = fig.add_subplot(gs[0])
        ax_main = fig.add_subplot(gs[1], sharex=ax_gene)
        ax_legend = fig.add_subplot(gs[2])
    else:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(
            2,
            1,
            height_ratios=[max(3.0, 0.32 * n_patients), 1.0],
            hspace=0.12,
            figure=fig,
        )
        ax_main = fig.add_subplot(gs[0])
        ax_legend = fig.add_subplot(gs[1])
        ax_gene = None
    ax_legend.axis("off")

    # ── gene expression panel ─────────────────────────────────────────────────
    if has_gene_panel and ax_gene is not None:
        ax_gene.plot(np.arange(n_bins), gene_expr, color="#333333", linewidth=1.8)
        ax_gene.fill_between(np.arange(n_bins), gene_expr, alpha=0.15, color="#333333")
        ax_gene.set_ylabel(f"{highlight_gene}\n(barycenter)", fontsize=9)
        ax_gene.set_xlim(-0.5, n_bins - 0.5)
        ax_gene.tick_params(axis="x", labelbottom=False)
        ax_gene.spines["top"].set_visible(False)
        ax_gene.spines["right"].set_visible(False)
        ax_gene.set_title(
            title or "Patient pseudotime bin → barycenter mapping (majority cell type)",
            fontsize=11,
            pad=6,
        )

    # ── build RGBA image for the heatmap ─────────────────────────────────────
    uncovered_rgba = np.array([0.92, 0.92, 0.92, 1.0])
    rgba = np.tile(uncovered_rgba, (n_patients, n_bins, 1))

    used_celltypes: set[str] = set()
    for p_idx in range(n_patients):
        for b_idx in range(n_bins):
            ct = grid[p_idx][b_idx]
            if ct is not None:
                used_celltypes.add(ct)
                color = cell_type_colors.get(ct, "#808080")
                rgba[p_idx, b_idx] = mcolors.to_rgba(color)

    ax_main.imshow(rgba, aspect="auto", interpolation="nearest", origin="upper")

    # ── heatmap aesthetics ────────────────────────────────────────────────────
    ax_main.set_yticks(range(n_patients))
    label_size = max(5, min(9, 180 // max(n_patients, 1)))
    ax_main.set_yticklabels(patient_ids, fontsize=label_size)
    ax_main.set_xlim(-0.5, n_bins - 0.5)
    ax_main.set_ylim(n_patients - 0.5, -0.5)

    tick_fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
    tick_positions = [f * (n_bins - 1) for f in tick_fracs]
    ax_main.set_xticks(tick_positions)
    ax_main.set_xticklabels(["0", "0.25", "0.5", "0.75", "1.0"], fontsize=9)
    ax_main.set_xlabel("Barycenter pseudotime", fontsize=10)
    ax_main.set_ylabel("Patient", fontsize=10)

    if ax_gene is None:
        ax_main.set_title(
            title or "Patient pseudotime bin → barycenter mapping (majority cell type)",
            fontsize=11,
        )

    for y in np.arange(0.5, n_patients - 0.5):
        ax_main.axhline(y, color="white", linewidth=0.3, alpha=0.6)

    # ── legend ────────────────────────────────────────────────────────────────
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=mcolors.to_rgba(cell_type_colors.get(ct, "#808080")), label=ct) for ct in sorted(used_celltypes)
    ]
    legend_handles.append(Patch(facecolor=tuple(uncovered_rgba), label="Uncovered (no cells)"))

    ax_legend.legend(
        handles=legend_handles,
        loc="center",
        ncol=min(len(legend_handles), 7),
        frameon=False,
        fontsize=8,
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
