import anndata
import numpy as np
import pandas as pd
from tqdm import tqdm
from tslearn.barycenters import softdtw_barycenter
from tslearn.metrics import soft_dtw_alignment


def interpolate_expression(
    exp_data: pd.DataFrame,
    traj_cond: np.ndarray,
    traj_val_new_pts: np.ndarray,
    win_sz: float,
) -> np.ndarray:
    """Interpolates expression data along pseudotime.

    Args:
        exp_data: Expression data for a single sample.
        traj_cond: Pseudotime values corresponding to `exp_data`.
        traj_val_new_pts: New pseudotime grid points.
        win_sz: Window size for Gaussian-weighted interpolation.

    Returns
    -------
        Interpolated expression values at `traj_val_new_pts`.
    """
    return np.array(
        [
            np.sum(exp_data.values * np.exp(-((x - traj_cond) ** 2) / (win_sz**2))[:, np.newaxis], axis=0)
            / np.sum(np.exp(-((x - traj_cond) ** 2) / (win_sz**2)))
            for x in traj_val_new_pts
        ]
    )


def compute_barycenter(
    interpolated_arrays: np.ndarray,
    method: str = "soft_dtw",
    gamma: float = 10,
    max_iter: int = 100,
    tol: float = 1e-3,
    verbose: bool = True,
) -> np.ndarray:
    """Computes the barycenter trajectory using DTW.

    Args:
        interpolated_arrays: Interpolated gene expression arrays for each sample.
        method: Method for computing barycenter, currently only 'soft_dtw' supported.
            Defaults to "soft_dtw".
        gamma: Gamma parameter for soft-DTW. Defaults to 10.
        max_iter: Maximum iterations for DTW barycenter computation. Defaults to 100.
        tol: Convergence tolerance. Defaults to 1e-3.
        verbose: Whether to display progress. Defaults to True.

    Returns
    -------
        Computed barycenter expression values.

    Raises
    ------
        ValueError: If method is not 'soft_dtw'.
    """
    if method == "soft_dtw":
        if verbose:
            print("Computing barycenter using soft-DTW...")
        return softdtw_barycenter(interpolated_arrays, gamma=gamma, max_iter=max_iter, tol=tol)
    else:
        raise ValueError("Currently only 'soft_dtw' is supported")


def map_cells_to_aligned_pseudotime(
    adata: anndata.AnnData,
    pseudotime_key: str,
    batch_key: str,
    gamma: float = 10,
    verbose: bool = True,
) -> anndata.AnnData:
    """Maps individual cells to the aligned pseudotime using soft-DTW.

    Args:
        adata: AnnData object with pseudotime in `adata.obs[pseudotime_key]`
            and batch labels in `adata.obs[batch_key]`.
        pseudotime_key: Column in `adata.obs` containing original pseudotime values.
        batch_key: Column in `adata.obs` indicating sample/batch labels.
        gamma: Gamma parameter for soft-DTW. Defaults to 10.
        verbose: Whether to display a progress bar. Defaults to True.

    Returns
    -------
        Updated AnnData object with "aligned_pseudotime" stored in
        `adata.obs["aligned_pseudotime"]`.
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
    adata: anndata.AnnData,
    pseudotime_key: str,
    batch_key: str,
    win_sz: float = 0.3,
    num_pts: int = 30,
    gamma: float = 10,
    method: str = "soft_dtw",
    max_iter: int = 100,
    tol: float = 1e-3,
    verbose: bool = True,
) -> anndata.AnnData:
    """Aligns pseudotime across multiple samples and computes a barycenter trajectory.

    Parameters
    ----------
    adata
        The input AnnData object with pseudotime stored in `adata.obs[pseudotime_key]`.

    pseudotime_key
        Column in `adata.obs` containing pseudotime values.

    batch_key
        Column in `adata.obs` indicating sample/batch labels.

    win_sz
        Window size for interpolation.
        Defaults to 0.3.

    num_pts
        Number of points for interpolation.
        Defaults to 30.

    gamma
        Gamma parameter for soft-DTW.
        Defaults to 10.

    method
        Method for computing barycenter, either 'soft_dtw' or 'dba'.
        Defaults to "soft_dtw".

    max_iter
        Maximum iterations for DTW barycenter computation.
        Defaults to 100.

    tol
        Convergence tolerance.
        Defaults to 1e-3.

    verbose
        Whether to display progress bars.
        Defaults to True.

    Returns
    -------
    anndata.AnnData
        The modified AnnData object with aligned pseudotime in
        `adata.obs["aligned_pseudotime"]` and barycenter expression stored in
        `adata.uns["barycenter_expr"]`.
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
