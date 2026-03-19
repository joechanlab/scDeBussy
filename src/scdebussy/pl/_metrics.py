import matplotlib.pyplot as plt


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
