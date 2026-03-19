import numpy as np
import torch
from dtw import dtw
from sklearn.isotonic import IsotonicRegression
from tslearn.barycenters import softdtw_barycenter


def smooth_patient_trajectory(
    adata,
    patient_id,
    patient_key="patient",
    pseudotime_key="s_local",
    n_bins=100,
    bandwidth=0.1,
):
    """
    Smooth expression data for a single patient along local pseudotime using Gaussian kernel regression

    Extracts and smooths expression data for a single patient along local
    pseudotime s_{p,i}.

    Parameters
    ----------
    adata : AnnData

        Annotated data matrix containing all patients.
    patient_id : str
        Patient identifier used to subset adata.
    patient_key : str
        Column in adata.obs identifying patients.
    pseudotime_key : str
        Column in adata.obs containing local pseudotime s_{p,i} in [0, 1].
    n_bins : int
        Number of pseudotime bins for the interpolation grid tau.
    bandwidth : float
        Gaussian kernel bandwidth for expression smoothing.

    Returns
    -------
    x_smoothed : torch.Tensor, shape (n_bins, n_genes)
        Smoothed expression matrix along pseudotime grid.
    density : torch.Tensor, shape (n_bins,)
        Kernel weight sums per bin, used for downstream density diagnostics.
    """
    # Subset and sort cells by local pseudotime s_{p,i}
    p_data = adata[adata.obs[patient_key] == patient_id]
    if p_data.n_obs == 0:
        raise ValueError(f"No cells found for patient_id={patient_id!r}.")

    sort_order = np.argsort(p_data.obs[pseudotime_key].values)
    p_data = p_data[sort_order, :]

    x_raw = p_data.X.toarray() if hasattr(p_data.X, "toarray") else p_data.X
    x_raw = np.asarray(x_raw, dtype=float)
    s_local = p_data.obs[pseudotime_key].values.astype(float)

    if np.any(~np.isfinite(s_local)):
        raise ValueError(f"Non-finite pseudotime values found for patient_id={patient_id!r}.")

    # Uniform pseudotime grid tau in [0, 1]
    tau_grid = np.linspace(0, 1, n_bins)
    sq_dist = (tau_grid[:, None] - s_local[None, :]) ** 2

    # 1. Compute Gaussian kernel weights (unnormalized = density proxy)
    weights = np.exp(-sq_dist / (2 * bandwidth**2))
    density = weights.sum(axis=1, keepdims=True)

    # 2. Normalize weights and compute smoothed expression
    weights_norm = weights / (density + 1e-8)
    x_smoothed = weights_norm @ x_raw

    return (
        torch.tensor(x_smoothed, dtype=torch.float32),
        torch.tensor(density.flatten(), dtype=torch.float32),
    )


def _validate_alignment_inputs(adata, patient_key, pseudotime_key, n_bins, bandwidth):
    if patient_key not in adata.obs:
        raise ValueError(f"{patient_key!r} is missing from adata.obs.")
    if pseudotime_key not in adata.obs:
        raise ValueError(f"{pseudotime_key!r} is missing from adata.obs.")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    pseudotime = np.asarray(adata.obs[pseudotime_key], dtype=float)
    if np.any(~np.isfinite(pseudotime)):
        raise ValueError("Pseudotime values must be finite.")
    if np.any((pseudotime < 0) | (pseudotime > 1)):
        raise ValueError("Pseudotime values must lie within [0, 1].")

    patient_ids = list(adata.obs[patient_key].dropna().unique())
    if not patient_ids:
        raise ValueError(f"No non-null patient ids found in adata.obs[{patient_key!r}].")

    for patient_id in patient_ids:
        n_cells = int((adata.obs[patient_key] == patient_id).sum())
        if n_cells == 0:
            raise ValueError(f"No cells found for patient_id={patient_id!r}.")

    return patient_ids


def _fit_barycenter_from_trajectories(patient_trajectories, n_bins, gamma, max_iter, tol, verbose):
    barycenter = softdtw_barycenter(X=patient_trajectories, gamma=gamma, max_iter=5)
    prev_cost = float("inf")
    warp_paths = []

    for iteration in range(max_iter):
        cropped_trajectories = []
        total_cost = 0.0
        warp_paths = []

        for trajectory in patient_trajectories:
            alignment = dtw(
                x=barycenter,
                y=trajectory,
                dist_method="cosine",
                step_pattern="asymmetric",
                open_begin=True,
                open_end=True,
                window_type="sakoechiba",
                window_args={"window_size": max(1, int(0.2 * n_bins))},
            )

            total_cost += alignment.distance
            warp_path = list(zip(alignment.index1, alignment.index2, strict=False))
            warp_paths.append(warp_path)

            start_idx = alignment.index2.min()
            end_idx = alignment.index2.max()
            cropped_trajectories.append(trajectory[start_idx : end_idx + 1])

        if verbose:
            print(f"Iter {iteration + 1:02d} | Cost: {total_cost:.4f}")

        if abs(prev_cost - total_cost) < tol:
            if verbose:
                print("Convergence reached.")
            break
        prev_cost = total_cost

        barycenter = softdtw_barycenter(
            X=cropped_trajectories,
            gamma=gamma,
            max_iter=1,
            init=barycenter,
        )

    return barycenter, warp_paths


def _warp_path_to_grid_mapping(warp_path, n_bins):
    patient_tau = np.linspace(0, 1, n_bins)
    barycenter_tau = np.linspace(0, 1, n_bins)
    if not warp_path:
        return patient_tau, patient_tau

    path = np.asarray(warp_path, dtype=int)
    bary_indices = path[:, 0]
    patient_indices = path[:, 1]

    unique_patient_indices = np.unique(patient_indices)
    mapped_bary_indices = np.array(
        [bary_indices[patient_indices == idx].mean() for idx in unique_patient_indices],
        dtype=float,
    )
    mapped_bary_tau = np.interp(mapped_bary_indices, np.arange(n_bins), barycenter_tau)
    patient_tau_subset = patient_tau[unique_patient_indices]

    if len(patient_tau_subset) == 1:
        patient_tau_subset = np.array([0.0, 1.0], dtype=float)
        mapped_bary_tau = np.repeat(mapped_bary_tau[0], 2)

    return patient_tau_subset, mapped_bary_tau


def _map_cells_to_aligned_pseudotime(
    adata,
    patient_ids,
    patient_key,
    pseudotime_key,
    warp_paths,
    n_bins,
):
    aligned = np.full(adata.n_obs, np.nan, dtype=float)

    for patient_id, warp_path in zip(patient_ids, warp_paths, strict=False):
        mask = adata.obs[patient_key] == patient_id
        if int(mask.sum()) == 0:
            continue

        patient_pseudotime = adata.obs.loc[mask, pseudotime_key].to_numpy(dtype=float)
        sort_idx = np.argsort(patient_pseudotime)
        sorted_pseudotime = patient_pseudotime[sort_idx]

        patient_tau_subset, mapped_bary_tau = _warp_path_to_grid_mapping(warp_path, n_bins)
        mapped_sorted = np.interp(sorted_pseudotime, patient_tau_subset, mapped_bary_tau)

        isotonic = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0, y_max=1.0)
        mapped_monotone = isotonic.fit_transform(sorted_pseudotime, mapped_sorted)

        mapped_original = np.empty_like(mapped_monotone)
        mapped_original[sort_idx] = mapped_monotone
        aligned[np.flatnonzero(mask.to_numpy())] = mapped_original

    return aligned


def scDeBussy(
    adata,
    patient_key="patient",
    pseudotime_key="s_local",
    n_bins=100,
    bandwidth=0.1,
    gamma=0.1,
    max_iter=20,
    tol=1e-4,
    key_added="aligned_pseudotime",
    barycenter_key="barycenter",
    verbose=False,
):
    """
    Align per-patient pseudotime trajectories onto a shared barycenter.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing all patients.
    patient_key : str
        Column in ``adata.obs`` identifying patients or batches.
    pseudotime_key : str
        Column in ``adata.obs`` containing local pseudotime values in ``[0, 1]``.
    n_bins : int
        Number of pseudotime bins used to smooth each patient trajectory.
    bandwidth : float
        Gaussian kernel bandwidth used during trajectory smoothing.
    gamma : float
        Soft-DTW gamma parameter for barycenter averaging.
    max_iter : int
        Maximum number of outer EM iterations.
    tol : float
        Convergence tolerance for the outer EM loop.
    key_added : str
        Key in ``adata.obs`` where aligned pseudotime is stored.
    barycenter_key : str
        Key in ``adata.uns`` where barycenter metadata is stored.
    verbose : bool
        Whether to print EM progress.

    Returns
    -------
    AnnData
        The same ``adata`` object with aligned results written in place.
    """
    patient_ids = _validate_alignment_inputs(adata, patient_key, pseudotime_key, n_bins, bandwidth)

    patient_trajectories = []
    patient_densities = {}
    for patient_id in patient_ids:
        smoothed, density = smooth_patient_trajectory(
            adata,
            patient_id=patient_id,
            patient_key=patient_key,
            pseudotime_key=pseudotime_key,
            n_bins=n_bins,
            bandwidth=bandwidth,
        )
        patient_trajectories.append(smoothed.detach().cpu().numpy())
        patient_densities[str(patient_id)] = density.detach().cpu().numpy()

    barycenter, warp_paths = _fit_barycenter_from_trajectories(
        patient_trajectories=patient_trajectories,
        n_bins=n_bins,
        gamma=gamma,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
    )

    adata.obs[key_added] = _map_cells_to_aligned_pseudotime(
        adata,
        patient_ids=patient_ids,
        patient_key=patient_key,
        pseudotime_key=pseudotime_key,
        warp_paths=warp_paths,
        n_bins=n_bins,
    )

    adata.uns[barycenter_key] = {
        "aligned_pseudotime": np.linspace(0, 1, n_bins),
        "expression": barycenter,
        "patient_ids": list(patient_ids),
        "warp_paths": [[(int(i), int(j)) for i, j in warp_path] for warp_path in warp_paths],
        "patient_densities": patient_densities,
        "params": {
            "patient_key": patient_key,
            "pseudotime_key": pseudotime_key,
            "n_bins": n_bins,
            "bandwidth": bandwidth,
            "gamma": gamma,
            "max_iter": max_iter,
            "tol": tol,
            "key_added": key_added,
        },
    }

    return adata


__all__ = ["scDeBussy", "smooth_patient_trajectory"]
