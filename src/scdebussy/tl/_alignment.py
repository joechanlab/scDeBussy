import numpy as np
import torch
from dtw import dtw
from scipy.ndimage import gaussian_filter1d as _gaussian_filter1d
from scipy.special import logsumexp as _logsumexp
from sklearn.isotonic import IsotonicRegression
from tslearn.barycenters import softdtw_barycenter


def smooth_patient_trajectory(
    adata,
    patient_id,
    patient_key="patient",
    pseudotime_key="s_local",
    n_bins=100,
    bandwidth=0.1,
    method="llr",
    adaptive_bandwidth=False,
    knn_k=10,
):
    """
    Smooth expression data for a single patient along local pseudotime.

    Extracts and smooths expression data for a single patient along local
    pseudotime s_{p,i} using either Nadaraya-Watson kernel regression (``"nw"``)
    or local linear regression (``"llr"``).  Local linear regression eliminates
    boundary bias at tau=0 and tau=1 and returns the slope coefficient as a
    free byproduct, enabling more accurate endpoint estimates.

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
        Gaussian kernel bandwidth.  When ``adaptive_bandwidth=True`` this
        acts as a minimum floor for the per-bin bandwidth.
    method : str
        Interpolation method.  ``"nw"`` for Nadaraya-Watson (degree-0 local
        polynomial); ``"llr"`` for local linear regression (degree-1), which
        is asymptotically unbiased at the boundaries.  Default ``"llr"``.
    adaptive_bandwidth : bool
        When ``True``, replace the global bandwidth with a per-bin balloon
        estimator: h_t = distance to the knn_k-th nearest cell pseudotime
        from tau_t, floored at ``bandwidth``.  Default ``False``.
    knn_k : int
        Number of nearest neighbours used when ``adaptive_bandwidth=True``.
        Default 10.

    Returns
    -------
    x_smoothed : torch.Tensor, shape (n_bins, n_genes)
        Smoothed expression matrix along pseudotime grid.
    density : torch.Tensor, shape (n_bins,)
        Kernel weight sums per bin, used for downstream density diagnostics.
    x_variance : torch.Tensor, shape (n_bins, n_genes)
        Weighted residual variance per bin per gene.  Captures per-bin
        uncertainty for downstream uncertainty-weighted alignment.
    """
    if method not in {"nw", "llr"}:
        raise ValueError(f"method must be 'nw' or 'llr', got {method!r}.")

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

    tau_grid = np.linspace(0, 1, n_bins)
    centered_s = s_local[None, :] - tau_grid[:, None]  # (T, N_p)
    sq_dist = centered_s**2

    if adaptive_bandwidth:
        k_idx = min(knn_k - 1, x_raw.shape[0] - 1)
        h_t = np.maximum(np.sort(np.abs(centered_s), axis=1)[:, k_idx], bandwidth)
        weights = np.exp(-sq_dist / (2 * h_t[:, None] ** 2))
    else:
        weights = np.exp(-sq_dist / (2 * bandwidth**2))

    density = weights.sum(axis=1, keepdims=True)  # (T, 1)
    weights_norm = weights / (density + 1e-8)  # (T, N_p)

    if method == "nw":
        x_smoothed = weights_norm @ x_raw  # (T, G)
        x_variance = weights_norm @ (x_raw**2) - x_smoothed**2  # (T, G)

    else:  # llr
        # Local linear regression: fit intercept + slope at each grid point.
        # Z[t] = [1, s_i - tau_t] — design matrix (T, N_p, 2)
        Z = np.stack([np.ones_like(centered_s), centered_s], axis=2)
        WZ = weights[:, :, None] * Z  # (T, N_p, 2)
        ZtWZ = np.einsum("tni,tnj->tij", WZ, Z) + 1e-8 * np.eye(2)  # (T, 2, 2)
        ZtWX = np.einsum("tni,ng->tig", WZ, x_raw)  # (T, 2, G)
        beta = np.linalg.solve(ZtWZ, ZtWX)  # (T, 2, G)
        x_smoothed = beta[:, 0, :]  # (T, G) intercept

        # Weighted residual variance via algebraic expansion of E_w[r^2]:
        # r_{t,i,g} = X_ig - beta_0[t,g] - beta_1[t,g]*(s_i - tau_t)
        b0 = beta[:, 0, :]  # (T, G)
        b1 = beta[:, 1, :]  # (T, G)
        E_w_X2 = weights_norm @ (x_raw**2)  # (T, G)
        E_w_X = weights_norm @ x_raw  # (T, G)
        E_w_Xc = (weights_norm * centered_s) @ x_raw  # (T, G)
        E_w_c = (weights_norm * centered_s).sum(axis=1)  # (T,)
        E_w_c2 = (weights_norm * centered_s**2).sum(axis=1)  # (T,)
        x_variance = (
            E_w_X2 - 2 * b0 * E_w_X - 2 * b1 * E_w_Xc + b0**2 + 2 * b0 * b1 * E_w_c[:, None] + b1**2 * E_w_c2[:, None]
        )

    x_variance = np.maximum(x_variance, 0.0)  # guard against tiny negative values from floating point

    return (
        torch.tensor(x_smoothed, dtype=torch.float32),
        torch.tensor(density.flatten(), dtype=torch.float32),
        torch.tensor(x_variance, dtype=torch.float32),
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

    _window_size = max(1, int(dtw_window_fraction * n_bins))

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
            traj_arr = np.asarray(trajectory)
            alignment = dtw(
                x=traj_arr,
                y=barycenter,
                dist_method=dtw_dist_method,
                step_pattern=dtw_step_pattern,
                open_begin=dtw_open_begin,
                open_end=dtw_open_end,
                window_type="sakoechiba",
                window_args={"window_size": _window_size},
            )
            pat_start = int(alignment.index1.min())
            pat_end = int(alignment.index1.max())
            bary_start = int(alignment.index2.min())
            bary_end = int(alignment.index2.max())
            warp_path = list(zip(alignment.index1, alignment.index2, strict=False))

            total_cost += alignment.distance
            warp_paths.append(warp_path)

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


def _compute_distance_matrix(A: np.ndarray, B: np.ndarray, dist_method: str) -> np.ndarray:
    """Compute T×T pairwise distance matrix between rows of A (T,K) and B (T,K)."""
    if dist_method in {"sqeuclidean", "euclidean"}:
        A_sq = (A**2).sum(axis=1)
        B_sq = (B**2).sum(axis=1)
        D = np.maximum(A_sq[:, None] + B_sq[None, :] - 2.0 * (A @ B.T), 0.0)
        return D if dist_method == "sqeuclidean" else np.sqrt(D)
    elif dist_method == "cosine":
        A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
        return np.maximum(1.0 - A_n @ B_n.T, 0.0)
    elif dist_method == "cityblock":
        return np.abs(A[:, None, :] - B[None, :, :]).sum(axis=2)
    else:
        raise ValueError(f"Unknown dist_method: {dist_method!r}")


def _forward_backward_open(
    delta: np.ndarray,
    gamma: float,
    window_size: int,
    bary_min: int = 0,
    bary_max: int | None = None,
    advance_mask: np.ndarray | None = None,
) -> tuple:
    """Expected alignment matrix for asymmetric open-ended soft-DTW.

    Transitions available at each patient bin t:
      (1,0) always valid — patient advances, barycenter stays.
      (1,1), (1,2), (1,3) — both advance (at different speeds).  Only valid when
            advance_mask[t] is True (or advance_mask is None).  This allows the caller
            to block advance steps for LLR-extrapolated bins (no real cells) so that
            the path cannot gallop through barycenter regions those bins never occupied.
    The skip steps (1,2)/(1,3) let patients with a short phase stretch across the full
    barycenter phase, and patients missing a middle segment skip it cleanly.
    Open begin/end: path may start/end at any barycenter bin.
    Every patient bin is visited exactly once (all steps advance t by 1), so rows
    of A_bar sum to 1.

    Parameters
    ----------
    delta : (T, T) pairwise distance matrix Delta[t, t'] = d(X_p(t), B(t')).
    gamma : soft-DTW temperature parameter.
    window_size : Sakoe-Chiba half-width in bins.
    bary_min, bary_max : optional column masking for per-patient coverage windows.
    advance_mask : optional bool array of shape (T,).  When False at bin t, only the
        (1,0) stay-step is valid; (1,1)/(1,2)/(1,3) are blocked.  Use this to prevent
        LLR-extrapolated bins from advancing the barycenter.

    Returns
    -------
    A_bar : (T, T) expected alignment matrix; A_bar[t, t'] = P((t,t') on path).
    log_Z : float, log partition function.
    """
    T = delta.shape[0]
    if bary_max is None:
        bary_max = T - 1

    # Log-probability of visiting each cell; mask out-of-band cells
    log_p = -delta / gamma
    t_idx = np.arange(T)
    out_of_band = np.abs(t_idx[:, None] - t_idx[None, :]) > window_size
    log_p[out_of_band] = -np.inf

    # Per-patient barycenter window: mask columns outside [bary_min, bary_max]
    if bary_min > 0:
        log_p[:, :bary_min] = -np.inf
    if bary_max < T - 1:
        log_p[:, bary_max + 1 :] = -np.inf

    # --- Forward pass ---
    log_alpha = np.full((T, T), -np.inf)
    log_alpha[0] = log_p[0]  # open begin: all barycenter start positions valid

    for t in range(1, T):
        from_vert = log_alpha[t - 1]  # (1,0): always valid
        if advance_mask is None or advance_mask[t]:
            from_diag = np.empty(T)
            from_diag[0] = -np.inf
            from_diag[1:] = log_alpha[t - 1, :-1]  # (1,1)
            from_skip2 = np.empty(T)
            from_skip2[:2] = -np.inf
            from_skip2[2:] = log_alpha[t - 1, :-2]  # (1,2)
            from_skip3 = np.empty(T)
            from_skip3[:3] = -np.inf
            from_skip3[3:] = log_alpha[t - 1, :-3]  # (1,3)
            log_alpha[t] = log_p[t] + np.logaddexp(
                np.logaddexp(from_vert, from_diag),
                np.logaddexp(from_skip2, from_skip3),
            )
        else:
            # LLR-extrapolated bin: only stay-step is valid.
            # The path holds the current barycenter position instead of advancing.
            log_alpha[t] = log_p[t] + from_vert

    # Partition function: sum over all terminal barycenter positions (open end)
    terminal = log_alpha[T - 1]
    finite = terminal[np.isfinite(terminal)]
    if finite.size == 0:
        return np.zeros((T, T)), float(-np.inf)
    log_Z = float(_logsumexp(finite))

    # --- Backward pass ---
    log_beta = np.full((T, T), -np.inf)
    log_beta[T - 1] = 0.0  # open end: all terminal positions weight 1
    log_beta[T - 1, out_of_band[T - 1]] = -np.inf

    for t in range(T - 2, -1, -1):
        # Successor (1,0): always valid — (t, t') -> (t+1, t')
        to_vert = log_beta[t + 1] + log_p[t + 1]
        if advance_mask is None or advance_mask[t + 1]:
            # Successor (1,1): (t, t') -> (t+1, t'+1)
            to_diag = np.empty(T)
            to_diag[T - 1] = -np.inf
            to_diag[:-1] = log_beta[t + 1, 1:] + log_p[t + 1, 1:]
            # Successor (1,2): (t, t') -> (t+1, t'+2)
            to_skip2 = np.empty(T)
            to_skip2[T - 1] = -np.inf
            to_skip2[T - 2] = -np.inf
            to_skip2[:-2] = log_beta[t + 1, 2:] + log_p[t + 1, 2:]
            # Successor (1,3): (t, t') -> (t+1, t'+3)
            to_skip3 = np.empty(T)
            to_skip3[T - 1] = -np.inf
            to_skip3[T - 2] = -np.inf
            to_skip3[T - 3] = -np.inf
            to_skip3[:-3] = log_beta[t + 1, 3:] + log_p[t + 1, 3:]
            log_beta[t] = np.logaddexp(
                np.logaddexp(to_vert, to_diag),
                np.logaddexp(to_skip2, to_skip3),
            )
        else:
            log_beta[t] = to_vert

    # Expected alignment matrix A_bar[t,t'] = alpha[t,t'] * beta[t,t'] / Z
    log_A = log_alpha + log_beta - log_Z
    A_bar = np.exp(np.clip(log_A, -500.0, 0.0))
    A_bar[out_of_band] = 0.0
    return A_bar, log_Z


def _fit_barycenter_soft_em(
    patient_trajectories,
    n_bins,
    gamma,
    max_iter,
    tol,
    verbose,
    dtw_dist_method,
    barycenter_init_iter,
    patient_weights=None,
    iteration_callback=None,
    patient_densities=None,
):
    """Proper soft-EM for DTW barycenter alignment.

    E-step: compute expected alignment matrix A_bar_p via forward-backward on the
    open-ended DTW lattice (replaces hard Viterbi path).  No Sakoe-Chiba band is used:
    the band would force partial patients to advance into barycenter regions they don't
    cover, pulling LUAD cells into the SCLC region.  Without the band the path can stay
    at the coverage boundary via (1,0) steps, which is the correct open-end behaviour.

    When patient_densities is provided, each row of the distance matrix is scaled by
    (median_density / bin_density), capped at 5×.  This inflates the cost for LLR-
    extrapolated bins (low kernel-density, no real cells), causing the path to collapse
    them onto a single barycenter position via (1,0) stay-steps rather than advancing
    through intermediate barycenter regions that the patient never actually traversed.
    Patients with missing middle segments (e.g. LUAD+SCLC-N without SCLC-A) therefore
    skip the SCLC-A barycenter region instead of mapping their interpolated expression
    into it.

    M-step: direct coverage-weighted average — B[t'] = sum_p w_p A_bar_p[:,t'] X_p
    divided by sum_p w_p A_bar_p[:,t'].sum().  Bins not covered by any patient retain
    the previous B value rather than being pulled toward zero.  A Gaussian smoothing
    pass (sigma=2 bins) removes sharp coverage-boundary artefacts.
    The objective -sum_p w_p log Z_p is guaranteed non-increasing each iteration.
    """
    T = n_bins
    P = len(patient_trajectories)
    # Half-length window: large enough that a patient at bin t can stay at any
    # barycenter bin within T//2 steps (so partial patients covering [0, T//2] can
    # always hold position at their coverage boundary via (1,0) stay-steps).
    fb_window = max(T // 2, 1)

    if patient_weights is None:
        patient_weights = np.ones(P) / P

    # Initialise barycenter with a few soft-DTW barycenter iterations
    B = softdtw_barycenter(X=patient_trajectories, gamma=gamma, max_iter=barycenter_init_iter, weights=patient_weights)

    prev_cost = float("inf")
    iteration_costs: list[float] = []
    converged = False
    coverage_per_iteration: list[list[dict]] = []

    for iteration in range(max_iter):
        # E-step: expected alignment matrices
        alignment_matrices = []
        total_neg_log_Z = 0.0

        for p, traj in enumerate(patient_trajectories):
            delta = _compute_distance_matrix(traj, B, dtw_dist_method)
            advance_mask = None
            if patient_densities is not None:
                d = np.asarray(patient_densities[p], dtype=float)
                pos_d = d[d > 1e-6]
                if pos_d.size > 0:
                    ref_d = float(np.median(pos_d))
                    # Inflate cost rows by (ref / density), capped at 5×, to penalise
                    # LLR-extrapolated bins that are far from real cells.
                    row_weights = np.clip(ref_d / (d + 1e-8), 1.0, 5.0)
                    delta = delta * row_weights[:, None]
                    # Hard constraint: bins below 5% of peak density are LLR-only.
                    # Block (1,1)/(1,2)/(1,3) advance steps so the path cannot
                    # gallop through barycenter regions no real cells ever occupied.
                    advance_mask = d >= (d.max() * 0.05)
            A_bar, log_Z = _forward_backward_open(delta, gamma, fb_window, advance_mask=advance_mask)
            alignment_matrices.append(A_bar)
            total_neg_log_Z -= float(patient_weights[p]) * log_Z

        iteration_costs.append(float(total_neg_log_Z))

        # Coverage diagnostics from marginal distributions
        coverage_this_iter: list[dict] = []
        for A_bar in alignment_matrices:
            bary_marg = A_bar.sum(axis=0)
            pat_marg = A_bar.sum(axis=1)
            cb = np.where(bary_marg > 1e-6)[0]
            cp = np.where(pat_marg > 1e-6)[0]
            coverage_this_iter.append(
                {
                    "bary_start": int(cb[0]) if cb.size else 0,
                    "bary_end": int(cb[-1]) if cb.size else T - 1,
                    "pat_start": int(cp[0]) if cp.size else 0,
                    "pat_end": int(cp[-1]) if cp.size else T - 1,
                }
            )
        coverage_per_iteration.append(coverage_this_iter)

        if iteration_callback is not None:
            # Approximate warp paths from argmax of each row for the callback
            approx_paths = [
                list(zip(range(T), A_bar.argmax(axis=1).tolist(), strict=False)) for A_bar in alignment_matrices
            ]
            try:
                iteration_callback(
                    iteration=int(iteration + 1),
                    total_cost=total_neg_log_Z,
                    warp_paths=approx_paths,
                )
            except Exception:  # noqa: BLE001
                pass

        if verbose:
            print(f"Iter {iteration + 1:02d} | -log Z: {total_neg_log_Z:.4f}")

        if abs(prev_cost - total_neg_log_Z) < tol:
            converged = True
            if verbose:
                print("Convergence reached.")
            break
        prev_cost = total_neg_log_Z

        # M-step: direct coverage-weighted average.
        # B[t'] = (sum_p w_p * A_bar_p[:, t'].T @ X_p) / (sum_p w_p * A_bar_p[:, t'].sum())
        # Bins where total weighted coverage is negligible retain the previous B value
        # so that uncovered patients don't drag those bins toward zero expression.
        K = B.shape[1]
        numerator = np.zeros((T, K))
        denominator = np.zeros(T)
        for p, (traj, A_bar) in enumerate(zip(patient_trajectories, alignment_matrices, strict=False)):
            w = float(patient_weights[p])
            numerator += w * (A_bar.T @ np.asarray(traj))
            denominator += w * A_bar.sum(axis=0)
        covered = denominator > (1.0 / (P * T))
        B = np.where(covered[:, None], numerator / (denominator[:, None] + 1e-8), B)

    # Smooth the converged barycenter to remove sharp coverage-boundary artefacts.
    B = _gaussian_filter1d(B, sigma=2, axis=0)

    # Derive warp paths from the last E-step A_bar.  Use the expected barycenter bin
    # (weighted mean over t') rather than argmax: it's smoother, approximately monotone,
    # and avoids discrete jumps from ties or diffuse probability near coverage boundaries.
    coverage_threshold = 1.0 / (P * T)
    warp_paths = []
    final_coverage: list[dict] = []
    for A_bar in alignment_matrices:
        row_sums = A_bar.sum(axis=1) + 1e-8
        expected_bary = (A_bar * np.arange(T)[None, :]).sum(axis=1) / row_sums
        bary_bins = np.clip(np.round(expected_bary).astype(int), 0, T - 1).tolist()
        warp_paths.append(list(zip(range(T), bary_bins, strict=False)))
        bary_marg = A_bar.sum(axis=0)
        covered_cols = np.where(bary_marg > coverage_threshold)[0]
        final_coverage.append(
            {
                "bary_start": int(covered_cols[0]) if covered_cols.size else 0,
                "bary_end": int(covered_cols[-1]) if covered_cols.size else T - 1,
                "pat_start": 0,
                "pat_end": T - 1,
            }
        )

    diagnostics = {
        "iteration_costs": iteration_costs,
        "final_iteration": int(len(iteration_costs)),
        "converged": bool(converged),
        "final_cost": float(iteration_costs[-1]) if iteration_costs else None,
        "tolerance": float(tol),
        "max_iterations": int(max_iter),
        "coverage": final_coverage,
    }
    return B, warp_paths, diagnostics


def _warp_path_to_grid_mapping(warp_path, n_bins):
    patient_tau = np.linspace(0, 1, n_bins)
    barycenter_tau = np.linspace(0, 1, n_bins)
    if not warp_path:
        return patient_tau, patient_tau

    path = np.asarray(warp_path, dtype=int)
    # Column 0 = patient (index1/x), column 1 = barycenter (index2/y).
    patient_indices = path[:, 0]
    bary_indices = path[:, 1]

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
    gene_subset_key=None,
    method="llr",
    adaptive_bandwidth=False,
    knn_k=10,
    use_soft_em=False,
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
    gene_subset_key : str or None
        Column in ``adata.var`` whose boolean values indicate which genes to use
        for trajectory smoothing and DTW alignment.  When ``None`` all genes are
        used.  Intended for use with :func:`filter_recurrent_hvgs` (key
        ``"recurrent_hvg"``) or :func:`filter_informative_genes` (key
        ``"informative_gene"``).  All genes remain accessible in ``adata``
        regardless of this selection.
    method : str
        Interpolation method passed to :func:`smooth_patient_trajectory`.
        ``"llr"`` (local linear regression, default) eliminates boundary bias
        at tau=0 and tau=1.  ``"nw"`` uses the legacy Nadaraya-Watson estimator.
    adaptive_bandwidth : bool
        When ``True``, use a per-bin balloon bandwidth based on the knn_k-th
        nearest cell pseudotime.  Default ``False``.
    knn_k : int
        Number of nearest neighbours for the balloon bandwidth estimator.
        Only used when ``adaptive_bandwidth=True``.  Default 10.
    use_soft_em : bool
        When ``True``, replace the hard-Viterbi E-step with a proper soft E-step
        that computes the expected alignment matrix via the forward-backward
        algorithm on the open-ended DTW lattice, and replaces the soft-DTW
        barycenter M-step with a closed-form weighted average.  This makes the
        algorithm a proper EM with a guaranteed non-increasing objective
        ``-sum_p w_p log Z_p`` each iteration, and correctly handles the
        open-begin/open-end assumption in the M-step without padding heuristics.
        Default ``False`` (retains legacy behaviour).

    Returns
    -------
    AnnData
        The same ``adata`` object with aligned results written in place.
    """
    patient_ids = _validate_alignment_inputs(adata, patient_key, pseudotime_key, n_bins, bandwidth)

    # Restrict trajectory computation to a gene subset when requested.
    adata_for_trajectories = adata
    if gene_subset_key is not None:
        if gene_subset_key not in adata.var.columns:
            raise ValueError(f"gene_subset_key={gene_subset_key!r} not found in adata.var.")
        gene_mask = adata.var[gene_subset_key].astype(bool).values
        if not np.any(gene_mask):
            raise ValueError(f"gene_subset_key={gene_subset_key!r} selects zero genes.")
        adata_for_trajectories = adata[:, gene_mask]
    _validate_advanced_alignment_inputs(
        dtw_dist_method=dtw_dist_method,
        dtw_step_pattern=dtw_step_pattern,
        dtw_open_begin=dtw_open_begin,
        dtw_open_end=dtw_open_end,
        dtw_window_fraction=dtw_window_fraction,
        barycenter_init_iter=barycenter_init_iter,
        barycenter_update_iter=barycenter_update_iter,
    )

    if method not in {"nw", "llr"}:
        raise ValueError(f"method must be 'nw' or 'llr', got {method!r}.")
    if not isinstance(adaptive_bandwidth, (bool, np.bool_)):
        raise ValueError("adaptive_bandwidth must be a boolean.")
    if int(knn_k) < 1:
        raise ValueError("knn_k must be at least 1.")

    patient_trajectories = []
    patient_densities = {}
    patient_variances = {}
    for patient_id in patient_ids:
        smoothed, density, variance = smooth_patient_trajectory(
            adata_for_trajectories,
            patient_id=patient_id,
            patient_key=patient_key,
            pseudotime_key=pseudotime_key,
            n_bins=n_bins,
            bandwidth=bandwidth,
            method=method,
            adaptive_bandwidth=adaptive_bandwidth,
            knn_k=knn_k,
        )
        patient_trajectories.append(smoothed.detach().cpu().numpy())
        patient_densities[str(patient_id)] = density.detach().cpu().numpy()
        patient_variances[str(patient_id)] = variance.detach().cpu().numpy()

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

    if use_soft_em:
        barycenter, warp_paths, em_convergence = _fit_barycenter_soft_em(
            patient_trajectories=patient_trajectories,
            n_bins=n_bins,
            gamma=gamma,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            dtw_dist_method=dtw_dist_method,
            barycenter_init_iter=barycenter_init_iter,
            patient_weights=patient_weights,
            iteration_callback=iteration_callback,
            patient_densities=[patient_densities[str(pid)] for pid in patient_ids],
        )
    else:
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
        "warp_paths": {
            str(pid): np.array([(int(i), int(j)) for i, j in warp_path], dtype=np.int32)
            for pid, warp_path in zip(patient_ids, warp_paths, strict=False)
        },
        "patient_densities": patient_densities,
        "patient_variances": patient_variances,
        "coverage": coverage_summary,
        "em_convergence": em_convergence,
        "params": {
            "patient_key": patient_key,
            "pseudotime_key": pseudotime_key,
            "genes_used": list(adata_for_trajectories.var_names.astype(str)),
            "gene_subset_key": gene_subset_key,
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
            "method": method,
            "adaptive_bandwidth": adaptive_bandwidth,
            "knn_k": int(knn_k),
            "use_soft_em": bool(use_soft_em),
        },
    }

    return adata


__all__ = ["scDeBussy", "smooth_patient_trajectory"]
