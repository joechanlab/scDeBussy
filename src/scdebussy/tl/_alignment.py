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


def _validate_advanced_alignment_inputs(
    dtw_dist_method,
    dtw_step_pattern,
    dtw_open_begin,
    dtw_open_end,
    dtw_window_fraction,
    barycenter_init_iter,
    barycenter_update_iter,
):
    supported_dist_methods = {"cosine", "euclidean", "cityblock", "sqeuclidean"}
    supported_step_patterns = {
        "symmetric1",
        "symmetric2",
        "asymmetric",
        "symmetricP0",
        "asymmetricP0",
        "symmetricP05",
        "asymmetricP05",
        "symmetricP1",
        "asymmetricP1",
        "symmetricP2",
        "asymmetricP2",
        "typeIa",
        "typeIb",
        "typeIc",
        "typeId",
        "typeIas",
        "typeIbs",
        "typeIcs",
        "typeIds",
        "typeIIa",
        "typeIIb",
        "typeIIc",
        "typeIId",
        "typeIIas",
        "typeIIbs",
        "typeIIcs",
        "typeIIds",
        "typeIIIc",
        "typeIVc",
        "mori2006",
        "rigid",
    }

    if dtw_dist_method not in supported_dist_methods:
        raise ValueError(f"dtw_dist_method must be one of {sorted(supported_dist_methods)}, got {dtw_dist_method!r}.")

    if dtw_step_pattern not in supported_step_patterns:
        raise ValueError(
            f"dtw_step_pattern must be one of {sorted(supported_step_patterns)}, got {dtw_step_pattern!r}."
        )

    if not isinstance(dtw_open_begin, (bool, np.bool_)):
        raise ValueError("dtw_open_begin must be a boolean.")
    if not isinstance(dtw_open_end, (bool, np.bool_)):
        raise ValueError("dtw_open_end must be a boolean.")

    if not np.isfinite(dtw_window_fraction) or dtw_window_fraction <= 0 or dtw_window_fraction > 1:
        raise ValueError("dtw_window_fraction must be finite and lie in (0, 1].")

    if int(barycenter_init_iter) < 1:
        raise ValueError("barycenter_init_iter must be at least 1.")
    if int(barycenter_update_iter) < 1:
        raise ValueError("barycenter_update_iter must be at least 1.")


def _pad_trajectory_to_barycenter(trajectory, barycenter, bary_start, bary_end, pat_start, pat_end):
    """Resample patient crop onto the barycenter's aligned span and pad non-covered bins.

    Returns a full-length (n_bins, n_genes) array where the aligned region contains the
    patient's expression (resampled if lengths differ) and non-covered bins retain the
    current barycenter values.  This prevents tslearn's softdtw_barycenter from receiving
    variable-length sequences and applying its own boundary-anchored stretch (double warp).
    """
    padded = barycenter.copy()
    pat_crop = trajectory[pat_start : pat_end + 1]
    target_len = bary_end - bary_start + 1
    if target_len <= 0:
        return padded
    if len(pat_crop) != target_len:
        xs_src = np.linspace(0, 1, len(pat_crop))
        xs_dst = np.linspace(0, 1, target_len)
        pat_crop = np.stack(
            [np.interp(xs_dst, xs_src, pat_crop[:, g]) for g in range(pat_crop.shape[1])],
            axis=1,
        )
    padded[bary_start : bary_end + 1] = pat_crop
    return padded


def _fit_barycenter_from_trajectories(
    patient_trajectories,
    n_bins,
    gamma,
    max_iter,
    tol,
    verbose,
    dtw_dist_method,
    dtw_step_pattern,
    dtw_open_begin,
    dtw_open_end,
    dtw_window_fraction,
    barycenter_init_iter,
    barycenter_update_iter,
    patient_weights=None,
    iteration_callback=None,
):
    barycenter = softdtw_barycenter(
        X=patient_trajectories, gamma=gamma, max_iter=barycenter_init_iter, weights=patient_weights
    )
    prev_cost = float("inf")
    warp_paths = []
    iteration_costs: list[float] = []
    converged = False
    coverage_per_iteration: list[list[dict]] = []

    for iteration in range(max_iter):
        padded_trajectories = []
        total_cost = 0.0
        warp_paths = []
        coverage_this_iter: list[dict] = []

        for trajectory in patient_trajectories:
            alignment = dtw(
                x=barycenter,
                y=trajectory,
                dist_method=dtw_dist_method,
                step_pattern=dtw_step_pattern,
                open_begin=dtw_open_begin,
                open_end=dtw_open_end,
                window_type="sakoechiba",
                window_args={"window_size": max(1, int(dtw_window_fraction * n_bins))},
            )

            total_cost += alignment.distance
            warp_path = list(zip(alignment.index1, alignment.index2, strict=False))
            warp_paths.append(warp_path)

            bary_start = int(alignment.index1.min())
            bary_end = int(alignment.index1.max())
            pat_start = int(alignment.index2.min())
            pat_end = int(alignment.index2.max())
            coverage_this_iter.append(
                {"bary_start": bary_start, "bary_end": bary_end, "pat_start": pat_start, "pat_end": pat_end}
            )

            padded_trajectories.append(
                _pad_trajectory_to_barycenter(trajectory, barycenter, bary_start, bary_end, pat_start, pat_end)
            )

        iteration_costs.append(float(total_cost))
        coverage_per_iteration.append(coverage_this_iter)

        if iteration_callback is not None:
            try:
                iteration_callback(
                    iteration=int(iteration + 1),
                    total_cost=float(total_cost),
                    warp_paths=warp_paths,
                )
            except (ValueError, TypeError, RuntimeError, FloatingPointError, KeyError, IndexError):
                # Diagnostics callbacks must never interrupt alignment.
                pass

        if verbose:
            print(f"Iter {iteration + 1:02d} | Cost: {total_cost:.4f}")

        if abs(prev_cost - total_cost) < tol:
            converged = True
            if verbose:
                print("Convergence reached.")
            break
        prev_cost = total_cost

        barycenter = softdtw_barycenter(
            X=padded_trajectories,
            gamma=gamma,
            max_iter=barycenter_update_iter,
            init=barycenter,
            weights=patient_weights,
        )

    # Summarise final-iteration coverage per patient
    final_coverage = coverage_per_iteration[-1] if coverage_per_iteration else []

    diagnostics = {
        "iteration_costs": iteration_costs,
        "final_iteration": int(len(iteration_costs)),
        "converged": bool(converged),
        "final_cost": float(iteration_costs[-1]) if iteration_costs else None,
        "tolerance": float(tol),
        "max_iterations": int(max_iter),
        "coverage": final_coverage,
    }
    return barycenter, warp_paths, diagnostics


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


def _interp_with_linear_extrap(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Interpolate like np.interp but linearly extrapolate outside [xp[0], xp[-1]].

    Values are clipped to [0, 1] afterwards; the downstream isotonic regression
    enforces monotonicity and the same range, so this is just a better starting point
    than constant-clip for cells that fall outside the DTW warp-path coverage.
    """
    result = np.interp(x, xp, fp)
    if len(xp) >= 2:
        left_mask = x < xp[0]
        if np.any(left_mask):
            slope = (fp[1] - fp[0]) / (xp[1] - xp[0] + 1e-12)
            result[left_mask] = fp[0] + slope * (x[left_mask] - xp[0])
        right_mask = x > xp[-1]
        if np.any(right_mask):
            slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2] + 1e-12)
            result[right_mask] = fp[-1] + slope * (x[right_mask] - xp[-1])
    return np.clip(result, 0.0, 1.0)


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
        mapped_sorted = _interp_with_linear_extrap(sorted_pseudotime, patient_tau_subset, mapped_bary_tau)

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
    dtw_dist_method="cosine",
    dtw_step_pattern="asymmetric",
    dtw_open_begin=True,
    dtw_open_end=True,
    dtw_window_fraction=0.2,
    barycenter_init_iter=5,
    barycenter_update_iter=1,
    group_key=None,
    rmse_truth_key="tau_global",
    log_iteration_rmse_if_available=True,
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
    dtw_dist_method : str
        Distance metric used inside DTW (e.g. ``"cosine"`` or ``"euclidean"``).
    dtw_step_pattern : str
        Step pattern used by DTW (e.g. ``"asymmetric"``, ``"symmetric1"``, ``"symmetric2"``).
    dtw_open_begin : bool
        Whether DTW alignment is allowed to start from an open boundary.
    dtw_open_end : bool
        Whether DTW alignment is allowed to end at an open boundary.
    dtw_window_fraction : float
        Fraction of ``n_bins`` used as Sakoe-Chiba window size; must lie in ``(0, 1]``.
    barycenter_init_iter : int
        Number of Soft-DTW barycenter iterations used for initialization.
    barycenter_update_iter : int
        Number of Soft-DTW barycenter iterations used in each EM M-step.
    group_key : str or None
        Optional column in ``adata.obs`` identifying patient groups (e.g. cancer subtype).
        When provided, per-patient weights are set to the inverse of each group's sample
        count, normalised to sum to 1.  This counteracts sample-count imbalance between
        groups so that a majority group does not dominate the barycenter.
    rmse_truth_key : str
        Optional ground-truth pseudotime key in ``adata.obs`` used to log
        per-iteration RMSE diagnostics when available (synthetic runs).
    log_iteration_rmse_if_available : bool
        Whether to compute and record per-iteration RMSE to ``rmse_truth_key``
        during EM iterations when the truth key is present.

    Returns
    -------
    AnnData
        The same ``adata`` object with aligned results written in place.
    """
    patient_ids = _validate_alignment_inputs(adata, patient_key, pseudotime_key, n_bins, bandwidth)
    _validate_advanced_alignment_inputs(
        dtw_dist_method=dtw_dist_method,
        dtw_step_pattern=dtw_step_pattern,
        dtw_open_begin=dtw_open_begin,
        dtw_open_end=dtw_open_end,
        dtw_window_fraction=dtw_window_fraction,
        barycenter_init_iter=barycenter_init_iter,
        barycenter_update_iter=barycenter_update_iter,
    )

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

    # Compute inverse-frequency weights by group to counteract sample imbalance.
    patient_weights = None
    group_assignments = None
    if group_key is not None:
        if group_key not in adata.obs.columns:
            raise ValueError(f"group_key={group_key!r} not found in adata.obs.")
        # Map each patient to its group (use the first cell's value)
        group_assignments = {}
        for pid in patient_ids:
            grp = adata.obs.loc[adata.obs[patient_key] == pid, group_key].iloc[0]
            group_assignments[str(pid)] = str(grp)
        group_counts: dict[str, int] = {}
        for grp in group_assignments.values():
            group_counts[grp] = group_counts.get(grp, 0) + 1
        raw_weights = np.array([1.0 / group_counts[group_assignments[str(pid)]] for pid in patient_ids])
        patient_weights = raw_weights / raw_weights.sum()

    iteration_rmse_to_truth: list[float] = []
    baseline_rmse_to_truth = None
    truth_values = None
    iteration_callback = None

    if log_iteration_rmse_if_available and rmse_truth_key and rmse_truth_key in adata.obs:
        truth_values = adata.obs[rmse_truth_key].to_numpy(dtype=float)
        baseline_values = adata.obs[pseudotime_key].to_numpy(dtype=float)
        baseline_mask = np.isfinite(truth_values) & np.isfinite(baseline_values)
        if np.any(baseline_mask):
            baseline_rmse_to_truth = float(
                np.sqrt(np.mean((baseline_values[baseline_mask] - truth_values[baseline_mask]) ** 2))
            )

        def _iteration_callback(iteration, total_cost, warp_paths):
            aligned_iter = _map_cells_to_aligned_pseudotime(
                adata,
                patient_ids=patient_ids,
                patient_key=patient_key,
                pseudotime_key=pseudotime_key,
                warp_paths=warp_paths,
                n_bins=n_bins,
            )
            mask = np.isfinite(truth_values) & np.isfinite(aligned_iter)
            if np.any(mask):
                rmse = float(np.sqrt(np.mean((aligned_iter[mask] - truth_values[mask]) ** 2)))
            else:
                rmse = float("nan")
            iteration_rmse_to_truth.append(rmse)

        iteration_callback = _iteration_callback

    barycenter, warp_paths, em_convergence = _fit_barycenter_from_trajectories(
        patient_trajectories=patient_trajectories,
        n_bins=n_bins,
        gamma=gamma,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        dtw_dist_method=dtw_dist_method,
        dtw_step_pattern=dtw_step_pattern,
        dtw_open_begin=dtw_open_begin,
        dtw_open_end=dtw_open_end,
        dtw_window_fraction=dtw_window_fraction,
        barycenter_init_iter=barycenter_init_iter,
        barycenter_update_iter=barycenter_update_iter,
        patient_weights=patient_weights,
        iteration_callback=iteration_callback,
    )

    adata.obs[key_added] = _map_cells_to_aligned_pseudotime(
        adata,
        patient_ids=patient_ids,
        patient_key=patient_key,
        pseudotime_key=pseudotime_key,
        warp_paths=warp_paths,
        n_bins=n_bins,
    )

    final_rmse_to_truth = None
    if truth_values is not None:
        aligned_values = adata.obs[key_added].to_numpy(dtype=float)
        final_mask = np.isfinite(truth_values) & np.isfinite(aligned_values)
        if np.any(final_mask):
            final_rmse_to_truth = float(np.sqrt(np.mean((aligned_values[final_mask] - truth_values[final_mask]) ** 2)))

    if iteration_rmse_to_truth:
        em_convergence["rmse_truth_key"] = rmse_truth_key
        em_convergence["iteration_rmse_to_truth"] = [float(v) for v in iteration_rmse_to_truth]
        em_convergence["baseline_rmse_to_truth"] = baseline_rmse_to_truth
        em_convergence["final_rmse_to_truth"] = final_rmse_to_truth

    # Build per-patient coverage summary from the final EM iteration.
    final_coverage = em_convergence.pop("coverage", [])
    coverage_summary: dict = {}
    for pid, cov in zip(patient_ids, final_coverage, strict=False):
        coverage_summary[str(pid)] = {
            "bary_start": cov["bary_start"],
            "bary_end": cov["bary_end"],
            "pat_start": cov["pat_start"],
            "pat_end": cov["pat_end"],
            "bary_fraction": (cov["bary_end"] - cov["bary_start"] + 1) / n_bins,
        }

    adata.uns[barycenter_key] = {
        "aligned_pseudotime": np.linspace(0, 1, n_bins),
        "expression": barycenter,
        "patient_ids": list(patient_ids),
        "warp_paths": [[(int(i), int(j)) for i, j in warp_path] for warp_path in warp_paths],
        "patient_densities": patient_densities,
        "coverage": coverage_summary,
        "em_convergence": em_convergence,
        "params": {
            "patient_key": patient_key,
            "pseudotime_key": pseudotime_key,
            "n_bins": n_bins,
            "bandwidth": bandwidth,
            "gamma": gamma,
            "max_iter": max_iter,
            "tol": tol,
            "key_added": key_added,
            "dtw_dist_method": dtw_dist_method,
            "dtw_step_pattern": dtw_step_pattern,
            "dtw_open_begin": dtw_open_begin,
            "dtw_open_end": dtw_open_end,
            "dtw_window_fraction": dtw_window_fraction,
            "barycenter_init_iter": barycenter_init_iter,
            "barycenter_update_iter": barycenter_update_iter,
            "group_key": group_key,
            "patient_weights": patient_weights.tolist() if patient_weights is not None else None,
            "group_assignments": group_assignments,
        },
    }

    return adata


__all__ = ["scDeBussy", "smooth_patient_trajectory"]
