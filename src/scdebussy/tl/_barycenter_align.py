import anndata as ad
import numpy as np
import pandas as pd
from tqdm import tqdm
from tslearn.barycenters import softdtw_barycenter
from tslearn.metrics import soft_dtw_alignment


def interpolate_expression(exp_data, traj_cond, traj_val_new_pts, win_sz):
    """
    Interpolates expression data along pseudotime.

    Parameters
    ----------
    exp_data : pd.DataFrame
        Expression data for a single sample.
    traj_cond : np.array
        Pseudotime values corresponding to `exp_data`.
    traj_val_new_pts : np.array
        New pseudotime grid points.
    win_sz : float
        Window size for Gaussian-weighted interpolation.

    Returns
    -------
    np.array
        Interpolated expression values at `traj_val_new_pts`.
    """
    return np.array(
        [
            np.sum(exp_data.values * np.exp(-((x - traj_cond) ** 2) / (win_sz**2))[:, np.newaxis], axis=0)
            / np.sum(np.exp(-((x - traj_cond) ** 2) / (win_sz**2)))
            for x in traj_val_new_pts
        ]
    )


def compute_barycenter(interpolated_arrays, method="soft_dtw", gamma=10, max_iter=100, tol=1e-3, verbose=True):
    """
    Computes the barycenter trajectory using DTW.

    Parameters
    ----------
    interpolated_arrays : np.array
        Interpolated gene expression arrays for each sample.
    method : str, optional
        'soft_dtw' or 'dba' (default: 'soft_dtw').
    gamma : float, optional
        Gamma parameter for soft-DTW (default: 10).
    max_iter : int, optional
        Maximum iterations for DTW barycenter computation (default: 100).
    tol : float, optional
        Convergence tolerance (default: 1e-3).
    verbose : bool, optional
        Whether to display progress.

    Returns
    -------
    np.array
        Computed barycenter expression values.
    """
    if method == "soft_dtw":
        if verbose:
            print("Computing barycenter using soft-DTW...")
        return softdtw_barycenter(interpolated_arrays, gamma=gamma, max_iter=max_iter, tol=tol)
    else:
        raise ValueError("Currently only 'soft_dtw' is supported")


def map_cells_to_aligned_pseudotime(adata, pseudotime_key, batch_key, gamma=10, verbose=True):
    """
    Maps individual cells to the aligned pseudotime using soft-DTW to find correspondences.

    Parameters
    ----------
    adata : AnnData
        AnnData object with pseudotime in `adata.obs[pseudotime_key]` and batch labels in `adata.obs[batch_key]`.
    pseudotime_key : str
        Column in `adata.obs` containing original pseudotime values.
    batch_key : str
        Column in `adata.obs` indicating sample/batch labels.
    gamma : float, optional
        Gamma parameter for soft-DTW (default: 10).
    verbose : bool, optional
        Whether to display a progress bar (default: True).

    Returns
    -------
    AnnData
        Updated AnnData object with "aligned_pseudotime" stored in `adata.obs["aligned_pseudotime"]`.
    """
    aligned_pseudotime = np.zeros(len(adata), dtype=float)  # Placeholder for mapped values
    barycenter_pseudotime = adata.uns["barycenter"]["aligned_pseudotime"]  # Interpolated aligned pseudotime

    batches = adata.obs[batch_key].unique()
    batch_iterator = tqdm(batches, desc="Mapping pseudotime", disable=not verbose)

    for batch in batch_iterator:
        # Extract batch-specific data
        mask = adata.obs[batch_key] == batch
        original_pseudotime = adata.obs.loc[mask, pseudotime_key].values
        expression_data = adata.to_df().loc[mask].values
        barycenter_expr = adata.uns["barycenter"]["expression"].values
        barycenter_pseudotime = adata.uns["barycenter"]["aligned_pseudotime"]

        # Compute the soft alignment matrix
        alignment_matrix, sim = soft_dtw_alignment(expression_data, barycenter_expr, gamma=gamma)

        # Get the highest-weighted match for each original pseudotime
        mapped_pseudotime = np.zeros(len(original_pseudotime))

        for orig_idx in range(alignment_matrix.shape[0]):
            # Find the barycenter index with the highest alignment score for this original index
            bary_idx = np.argmax(alignment_matrix[orig_idx])  # Get best match based on alignment strength
            mapped_pseudotime[orig_idx] = barycenter_pseudotime[bary_idx]

        # Handle missing values with nearest-neighbor interpolation
        mapped_series = pd.Series(mapped_pseudotime)
        mapped_series.replace(0, np.nan, inplace=True)  # Replace unset values with NaN
        mapped_series.interpolate(method="nearest", inplace=True)  # Nearest-neighbor interpolation
        mapped_pseudotime = mapped_series.to_numpy()

        # Use the mask to assign the correct values
        aligned_pseudotime[mask] = mapped_pseudotime

    # Store mapped pseudotime in `obs`
    adata.obs["aligned_pseudotime"] = aligned_pseudotime

    if verbose:
        print("Pseudotime mapping completed.")

    return adata


def align_pseudotime(
    adata: ad.AnnData,
    pseudotime_key: str,
    batch_key: str,
    win_sz=0.3,
    num_pts=30,
    gamma=10,
    method="soft_dtw",
    max_iter=100,
    tol=1e-3,
    verbose=True,
):
    """
    Aligns pseudotime across multiple samples and computes a barycenter trajectory.

    Parameters
    ----------
    adata : AnnData
        The input AnnData object with pseudotime stored in `adata.obs[pseudotime_key]`.
    pseudotime_key : str
        Column in `adata.obs` containing pseudotime values.
    batch_key : str
        Column in `adata.obs` indicating sample/batch labels.
    win_sz : float, optional
        Window size for interpolation (default: 0.3).
    num_pts : int, optional
        Number of points for interpolation (default: 30).
    gamma : float, optional
        Gamma parameter for soft-DTW (default: 10).
    method : str, optional
        'soft_dtw' or 'dba' (default: 'soft_dtw').
    max_iter : int, optional
        Maximum iterations for DTW barycenter computation (default: 100).
    tol : float, optional
        Convergence tolerance (default: 1e-3).
    verbose : bool, optional
        Whether to display progress bars (default: True).

    Returns
    -------
    AnnData
        The modified AnnData object with aligned pseudotime in `adata.obs["aligned_pseudotime"]`
        and barycenter expression stored in `adata.uns["barycenter_expr"]`.
    """
    # Prepare data for interpolation
    batches = adata.obs[batch_key].unique()
    gene_expr = adata.to_df()
    traj_val_new_pts = np.linspace(adata.obs[pseudotime_key].min(), adata.obs[pseudotime_key].max(), num_pts)

    interpolated_arrays = []

    if verbose:
        print("Interpolating expression data...")
        batch_iterator = tqdm(batches, desc="Processing batches")
    else:
        batch_iterator = batches

    for batch in batch_iterator:
        mask = adata.obs[batch_key] == batch
        traj_cond = adata.obs.loc[mask, pseudotime_key].values
        exp_data = gene_expr.loc[mask]

        # Perform interpolation
        val_new_pts = interpolate_expression(exp_data, traj_cond, traj_val_new_pts, win_sz)
        interpolated_arrays.append(val_new_pts)

    interpolated_arrays = np.array(interpolated_arrays, dtype=object)

    # Compute barycenter
    barycenter = compute_barycenter(
        interpolated_arrays, method=method, gamma=gamma, max_iter=max_iter, tol=tol, verbose=verbose
    )

    # Store results in `adata.uns`
    adata.uns["barycenter"] = {
        "aligned_pseudotime": traj_val_new_pts,
        "expression": pd.DataFrame(barycenter, columns=adata.var_names),
    }

    # Map individual cells to aligned pseudotime
    adata = map_cells_to_aligned_pseudotime(
        adata, pseudotime_key=pseudotime_key, batch_key=batch_key, gamma=gamma, verbose=verbose
    )

    if verbose:
        print("Alignment and barycenter computation completed.")

    return adata
