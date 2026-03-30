import matplotlib.pyplot as plt
import numpy as np


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
    # This visually confirms the "chapters" the algorithm will use for subsequence matching
    for i, seg in enumerate(segments):
        start_t = seg["start_time"]
        end_t = seg["end_time"]

        # Alternate between a very faint gray and pure white
        bg_color = "#F0F0F0" if i % 2 == 0 else "#FFFFFF"
        ax.axvspan(start_t, end_t, facecolor=bg_color, alpha=0.5, zorder=0)

        # Add segment number labels near the top of the plot
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

    # Move legend outside the plot so it doesn't cover the data
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
