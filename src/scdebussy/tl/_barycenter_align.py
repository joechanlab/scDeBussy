import anndata
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
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
    pca_components: int | None = 10,
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
    if "barycenter" not in adata.uns:
        raise KeyError("adata.uns['barycenter'] not found. Run barycenter computation first.")
    barycenter_pseudotime = adata.uns["barycenter"]["aligned_pseudotime"]  # Interpolated aligned pseudotime
    barycenter_expr = np.asarray(adata.uns["barycenter"]["expression"])

    aligned = np.full(len(adata), np.nan, dtype=float)

    batches = adata.obs[batch_key].unique()
    batch_iterator = tqdm(batches, desc="Mapping pseudotime", disable=not verbose)

    for batch in batch_iterator:
        # Extract batch-specific data
        mask = adata.obs[batch_key] == batch
        if mask.sum() == 0:
            continue

        orig_pt = adata.obs.loc[mask, pseudotime_key].values
        expr_df = adata.to_df().loc[mask]
        if expr_df.shape[0] == 0:
            continue
        X = expr_df.values

        # PCA reduction (concatenate to ensure shared subspace with barycenter)
        if (
            pca_components is not None
            and isinstance(pca_components, int)
            and pca_components > 0
            and X.shape[1] > pca_components
        ):
            concat = np.vstack([X, barycenter_expr])
            n_comp = min(pca_components, concat.shape[1])
            pca = PCA(n_components=n_comp)
            concat_p = pca.fit_transform(concat)
            X_p = concat_p[: X.shape[0]]
            Y_p = concat_p[X.shape[0] :]
        else:
            X_p = X
            Y_p = barycenter_expr

        # Sort cells by original pseudotime
        sort_idx = np.argsort(orig_pt)
        X_sorted = X_p[sort_idx]

        # soft-DTW alignment (rows: X_sorted, cols: Y_p)
        alignment_matrix, sim = soft_dtw_alignment(X_sorted, Y_p, gamma=gamma)

        # continuous index (expected barycenter index) per cell
        row_sums = alignment_matrix.sum(axis=1)
        idx_range = np.arange(alignment_matrix.shape[1])
        with np.errstate(invalid="ignore", divide="ignore"):
            cont_idx = (alignment_matrix * idx_range[None, :]).sum(axis=1) / row_sums
        cont_idx[row_sums == 0] = np.nan

        # Map continuous index to barycenter pseudotime
        mapped_sorted = np.full_like(cont_idx, np.nan, dtype=float)
        valid = ~np.isnan(cont_idx)
        if np.any(valid):
            mapped_sorted[valid] = np.interp(
                cont_idx[valid], np.arange(len(barycenter_pseudotime)), barycenter_pseudotime
            )

        # Enforce monotonic non-decreasing mapping to avoid crisscross using isotonic regression
        nan_mask = np.isnan(mapped_sorted)
        if not np.all(nan_mask):
            valid = ~nan_mask
            x_sorted = orig_pt[sort_idx][valid]
            y_sorted = mapped_sorted[valid]
            w = row_sums[valid]

            ir = IsotonicRegression(
                increasing=True,
                out_of_bounds="clip",
                y_min=np.nanmin(barycenter_pseudotime),
                y_max=np.nanmax(barycenter_pseudotime),
            )
            y_fit = ir.fit_transform(x_sorted, y_sorted, sample_weight=w)

            mapped_monotone = mapped_sorted.copy()
            mapped_monotone[valid] = y_fit
        else:
            mapped_monotone = mapped_sorted

        # put back in original order
        mapped_original = np.empty_like(mapped_monotone)
        mapped_original[sort_idx] = mapped_monotone

        aligned[mask] = mapped_original

    adata.obs["aligned_pseudotime"] = aligned
    if verbose:
        print("Mapping complete; added obs['aligned_pseudotime']")
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
