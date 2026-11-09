import anndata as ad
import numpy as np


def rbf_kernel(t, lengthscale, variance):
    """Compute the RBF (squared exponential) kernel matrix.

    Parameters
    ----------
    t : np.ndarray
        1D array of input time points of shape (N,).
    lengthscale : float
        Length scale controlling the smoothness of the kernel.
    variance : float
        Signal variance (output scale) of the kernel.

    Returns
    -------
    K : np.ndarray
        Kernel matrix of shape (N, N).
    """
    t = t[:, None]  # Turn to column vector (N x 1)
    sqdist = (t - t.T) ** 2  # Pairwise squared distances
    return variance * np.exp(-0.5 * sqdist / lengthscale**2)  # Squared Exponential Kernel


def sample_gp(t, lengthscale, variance, rng, jitter=1e-6):
    """Sample a single function from a Gaussian Process with RBF kernel.

    Parameters
    ----------
    t : np.ndarray
        1D array of input time points of shape (N,).
    lengthscale : float
        Length scale of the RBF kernel.
    variance : float
        Signal variance of the RBF kernel.
    rng : np.random.Generator
        NumPy random generator.
    jitter : float, optional
        Small diagonal term added for numerical stability. Default is 1e-6.

    Returns
    -------
    sample : np.ndarray
        GP sample of shape (N,).
    """
    K = rbf_kernel(t, lengthscale, variance) + jitter * np.eye(
        len(t)
    )  # Covariance matrix with jitter for numerical stability
    return rng.multivariate_normal(np.zeros(len(t)), K)  # Parameters are mean=0 and covariance matrix K from RBF Kernel


def sample_structured_latent_factors(time_grid, rng=None):
    """Sample four structured latent factors over a pseudotime grid.

    The four factors correspond to: monotonic up, monotonic down,
    Gaussian bump, and flat — each with optional small GP noise added
    for realism.

    Parameters
    ----------
    time_grid : np.ndarray
        1D array of pseudotime grid points of shape (T,).
    rng : np.random.Generator, optional
        NumPy random generator. A new one is created if not provided.

    Returns
    -------
    U : np.ndarray
        Latent factor matrix of shape (4, T).
    """
    if rng is None:
        rng = np.random.default_rng()

    T = len(time_grid)

    # --- 1. Monotonic up
    u_up = np.linspace(0, 1, T)

    # --- 2. Monotonic down
    u_down = 1 - u_up

    # --- 3. Bump (Gaussian-shaped)
    center = 0.5
    width = 0.15
    u_bump = np.exp(-0.5 * ((time_grid - center) / width) ** 2)

    # --- 4. Flat
    u_flat = np.zeros(T)

    # OPTIONAL: add small GP noise for realism
    def gp_noise(scale=0.1):
        return rng.normal(0, scale, T)

    U = np.vstack(
        [
            u_up + 0.1 * gp_noise(0.1),
            u_down + 0.1 * gp_noise(0.1),
            u_bump + 0.1 * gp_noise(0.1),
            u_flat + 0.05 * gp_noise(0.05),
        ]
    )

    return U


def construct_global_trajectory(U, W):
    """Compute global gene expression trajectories from latent factors and loadings.

    Parameters
    ----------
    U : np.ndarray
        Latent factor matrix of shape (M, T).
    W : np.ndarray
        Gene loading matrix of shape (K, M).

    Returns
    -------
    f_global : np.ndarray
        Global gene expression trajectories of shape (K, T).
    """
    return W @ U  # (K,M) @ (M,T) = (K,T)


def sample_patient_deviations(time_grid, K, deviation_kernel, rng):
    """Sample patient-specific GP deviations for all genes.

    Parameters
    ----------
    time_grid : np.ndarray
        1D array of pseudotime grid points of shape (T,).
    K : int
        Number of genes.
    deviation_kernel : dict
        Kernel parameters with keys ``'lengthscale'`` and ``'variance'``.
    rng : np.random.Generator
        NumPy random generator.

    Returns
    -------
    delta : np.ndarray
        Deviation matrix of shape (K, T) for one patient.
    """
    T = len(time_grid)  # Number of points in pseudotime grid
    delta = np.zeros((K, T))  # Initialize deviation matrix
    for k in range(K):
        delta[k] = sample_gp(
            time_grid,
            deviation_kernel["lengthscale"],
            deviation_kernel["variance"],
            rng,
        )
    return delta


def construct_patient_trajectories(f_global, time_grid, K, P, deviation_kernel, rng):
    """Build patient-specific gene expression trajectories by adding GP deviations.

    Parameters
    ----------
    f_global : np.ndarray
        Global gene expression trajectories of shape (K, T).
    time_grid : np.ndarray
        1D array of pseudotime grid points of shape (T,).
    K : int
        Number of genes.
    P : int
        Number of patients.
    deviation_kernel : dict
        Kernel parameters with keys ``'lengthscale'`` and ``'variance'``.
    rng : np.random.Generator
        NumPy random generator.

    Returns
    -------
    f_p : list of np.ndarray
        List of P arrays, each of shape (K, T).
    """
    f_p = []  # Initialize patient-specific trajectories
    for _ in range(P):
        delta_p = sample_patient_deviations(
            time_grid, K, deviation_kernel, rng
        )  # Per patient and per gene independent deviation (K,T)
        f_p.append(f_global + delta_p)
    return f_p  # list of P arrays, each (K,T)


def sample_cell_counts(P, lambda_cells, rng):
    """Sample the number of cells per patient from a Poisson distribution.

    Parameters
    ----------
    P : int
        Number of patients.
    lambda_cells : float
        Expected number of cells per patient (Poisson rate).
    rng : np.random.Generator
        NumPy random generator.

    Returns
    -------
    N_cells : np.ndarray
        Integer array of shape (P,) with cell counts per patient.
    """
    return rng.poisson(lambda_cells, size=P)


def truncated_normal(rng, size, mean, std, low=0, high=1):
    """Sample from a truncated normal distribution using rejection sampling.

    Parameters
    ----------
    rng : np.random.Generator
        NumPy random generator.
    size : int
        Number of samples to draw.
    mean : float
        Mean of the underlying normal distribution.
    std : float
        Standard deviation of the underlying normal distribution.
    low : float, optional
        Lower bound of the truncation interval. Default is 0.
    high : float, optional
        Upper bound of the truncation interval. Default is 1.

    Returns
    -------
    out : np.ndarray
        Array of shape (size,) with samples in [low, high].
    """
    out = []
    while len(out) < size:
        x = rng.normal(mean, std)
        if low <= x <= high:  # Accept only if within bounds
            out.append(x)
    return np.array(out)


def sample_tau_group(N, group, rng, eps=0.05):
    """Sample global pseudotimes for cells in a given patient group.

    Parameters
    ----------
    N : int
        Number of cells to sample.
    group : str
        Pseudotime distribution type. One of ``'early'``, ``'late'``,
        ``'transition'``, ``'bimodal'``, or ``'full'``.
    rng : np.random.Generator
        NumPy random generator.
    eps : float, optional
        Fraction of cells assigned uniform noise pseudotimes. Default is 0.05.

    Returns
    -------
    tau : np.ndarray
        Global pseudotime array of shape (N,) with values in [0, 1].
    """
    # Noise samples (uniform across the whole range)
    noise = rng.uniform(0, 1, size=N)

    # Main distribution
    if group == "early":
        main = truncated_normal(rng, N, 0.1, 0.05)
    elif group == "late":
        main = truncated_normal(rng, N, 0.9, 0.05)
    elif group == "transition":
        main = truncated_normal(rng, N, 0.5, 0.05)
    elif group == "bimodal":
        w = rng.beta(2, 2)  # Random weight for first mode between 0 and 1

        N1 = int(N * w)
        N2 = N - N1  # Complement weight for second mode

        peak1 = truncated_normal(rng, N1, 0.1, 0.05)
        peak2 = truncated_normal(rng, N2, 0.9, 0.05)

        main = np.concatenate([peak1, peak2])
    elif group == "full":
        main = rng.uniform(0, 1, size=N)
    else:
        raise ValueError(f"Unknown group: {group!r}")

    # Mix with noise
    mask = rng.random(N) > eps
    tau = np.where(mask, main, noise)
    return tau


def sample_global_pseudotimes(patient_groups, N_cells, rng, eps=0.05):
    """Sample global pseudotimes for all patients.

    Parameters
    ----------
    patient_groups : list of str
        Pseudotime distribution type for each patient
        (``'early'``, ``'late'``, ``'transition'``, ``'bimodal'``, ``'full'``).
    N_cells : np.ndarray
        Number of cells per patient of shape (P,).
    rng : np.random.Generator
        NumPy random generator.
    eps : float, optional
        Fraction of cells assigned uniform noise pseudotimes. Default is 0.05.

    Returns
    -------
    tau_global : list of np.ndarray
        List of P arrays, each containing global pseudotimes for one patient.
    """
    tau_global = []
    for p, group in enumerate(patient_groups):
        tau_global.append(
            sample_tau_group(N_cells[p], group, rng, eps=eps)
        )  # Sampled pseudotimes for patient p in specified group
    return tau_global


def _safe_minmax_scale(values):
    """Scale values to [0, 1] with safe handling for constant arrays."""
    v_min = np.min(values)
    v_max = np.max(values)
    if v_max == v_min:
        return np.zeros_like(values)
    return (values - v_min) / (v_max - v_min)


def _random_monotone_warp(tau, rng, warp_strength=0.2, n_knots=7):
    """Apply a smooth random monotone warping to pseudotime values."""
    if warp_strength <= 0:
        return tau

    x_knots = np.linspace(0.0, 1.0, n_knots)
    noise = rng.normal(loc=0.0, scale=0.2 * warp_strength, size=n_knots)
    y_knots = np.clip(x_knots + noise, 0.0, 1.0)
    y_knots[0] = 0.0
    y_knots[-1] = 1.0

    # Enforce monotonicity and normalize to preserve [0, 1] endpoints.
    y_knots = np.maximum.accumulate(y_knots)
    y_knots = _safe_minmax_scale(y_knots)
    y_knots[0] = 0.0
    y_knots[-1] = 1.0

    return np.interp(tau, x_knots, y_knots)


def _group_warp(tau, group, rng, warp_strength=0.2):
    """Apply a group-specific monotone warping profile."""
    if warp_strength <= 0:
        return tau

    s = float(warp_strength)
    if group == "early":
        alpha = 1.0 + 2.0 * s
        warped = np.power(tau, alpha)
    elif group == "late":
        alpha = 1.0 + 2.0 * s
        warped = 1.0 - np.power(1.0 - tau, alpha)
    elif group == "transition":
        steepness = max(0.15, 0.25 - 0.12 * s)
        warped = 0.5 * (np.tanh((tau - 0.5) / steepness) + 1.0)
    elif group in {"bimodal", "full"}:
        warped = _random_monotone_warp(tau, rng=rng, warp_strength=s)
    else:
        warped = tau

    return np.clip(warped, 0.0, 1.0)


def _sigmoid_warp(tau, warp_strength):
    """Apply sigmoid warping to pseudotime values.

    Squeezes the center of the trajectory, making transitions appear faster
    mid-pseudotime and slower at the boundaries.

    Parameters
    ----------
    tau : np.ndarray
        Pseudotime values in [0, 1].
    warp_strength : float
        Non-negative strength of warping. Larger values produce sharper sigmoid.

    Returns
    -------
    warped : np.ndarray
        Warped pseudotime values in [0, 1].
    """
    if warp_strength <= 0:
        return tau
    steepness = 3.0 + float(warp_strength) * 10.0
    sig = 1.0 / (1.0 + np.exp(-steepness * (2.0 * tau - 1.0)))
    sig_min = 1.0 / (1.0 + np.exp(steepness))
    sig_max = 1.0 / (1.0 + np.exp(-steepness))
    warped = (sig - sig_min) / (sig_max - sig_min + 1e-12)
    return np.clip(warped, 0.0, 1.0)


def _nonlinear_warp(tau, warp_strength):
    """Apply power-law (convex) warping to pseudotime values.

    Stretches the early part of the trajectory and compresses the late part.

    Parameters
    ----------
    tau : np.ndarray
        Pseudotime values in [0, 1].
    warp_strength : float
        Non-negative strength of warping.

    Returns
    -------
    warped : np.ndarray
        Warped pseudotime values in [0, 1].
    """
    if warp_strength <= 0:
        return tau
    power = 1.0 + float(warp_strength) * 2.0
    return np.clip(np.power(tau, power), 0.0, 1.0)


def _nonlinear_inverse_warp(tau, warp_strength):
    """Apply inverse power-law (concave) warping to pseudotime values.

    Compresses the early part of the trajectory and stretches the late part.

    Parameters
    ----------
    tau : np.ndarray
        Pseudotime values in [0, 1].
    warp_strength : float
        Non-negative strength of warping.

    Returns
    -------
    warped : np.ndarray
        Warped pseudotime values in [0, 1].
    """
    if warp_strength <= 0:
        return tau
    power = 1.0 / (1.0 + float(warp_strength))
    return np.clip(np.power(tau, power), 0.0, 1.0)


_WARP_DISPATCH = {
    "none": lambda tau, s, rng, group: tau,
    "random_monotone": lambda tau, s, rng, group: _random_monotone_warp(tau, rng=rng, warp_strength=s),
    "patient_group": lambda tau, s, rng, group: _group_warp(tau, group=group, rng=rng, warp_strength=s),
    "sigmoid": lambda tau, s, rng, group: _sigmoid_warp(tau, s),
    "nonlinear": lambda tau, s, rng, group: _nonlinear_warp(tau, s),
    "nonlinear_inverse": lambda tau, s, rng, group: _nonlinear_inverse_warp(tau, s),
}

_ALLOWED_WARP_MODES = frozenset(_WARP_DISPATCH) | {"mixed"}


def compute_local_pseudotime(
    tau_global,
    patient_groups=None,
    warp_mode="none",
    warp_strength=0.2,
    warp_types=None,
    rng=None,
):
    """Compute per-patient local pseudotimes with optional monotone warping.

    Parameters
    ----------
    tau_global : list of np.ndarray
        List of P arrays of global pseudotimes, each of shape (N_p,).
    patient_groups : list of str, optional
        Per-patient group labels used when ``warp_mode='patient_group'``.
    warp_mode : str, optional
        Warping mode. One of ``'none'``, ``'random_monotone'``,
        ``'patient_group'``, ``'sigmoid'``, ``'nonlinear'``,
        ``'nonlinear_inverse'``, or ``'mixed'``. Default is ``'none'``.
        When ``'mixed'``, ``warp_types`` must be provided and each patient
        is warped with its own mode from that list.
    warp_strength : float, optional
        Non-negative strength of warping. Ignored when ``warp_mode='none'``.
    warp_types : list of str, optional
        Per-patient warp mode names used when ``warp_mode='mixed'``.
        Length must equal ``len(tau_global)``. Each entry must be one of the
        non-``'mixed'`` warp modes.
    rng : np.random.Generator, optional
        NumPy random generator used by stochastic warping modes.

    Returns
    -------
    s_local : list of np.ndarray
        List of P arrays of local pseudotimes normalized to [0, 1].
    """
    if warp_mode not in _ALLOWED_WARP_MODES:
        raise ValueError(f"Unknown warp_mode: {warp_mode!r}. Allowed: {sorted(_ALLOWED_WARP_MODES)}")
    if warp_strength < 0:
        raise ValueError("warp_strength must be non-negative.")

    if rng is None:
        rng = np.random.default_rng()

    if warp_mode == "patient_group":
        if patient_groups is None:
            raise ValueError("patient_groups must be provided when warp_mode='patient_group'.")
        if len(patient_groups) != len(tau_global):
            raise ValueError("Length of patient_groups must match tau_global.")

    if warp_mode == "mixed":
        if warp_types is None:
            raise ValueError("warp_types must be provided when warp_mode='mixed'.")
        if len(warp_types) != len(tau_global):
            raise ValueError("Length of warp_types must match tau_global.")
        invalid = [m for m in warp_types if m not in _WARP_DISPATCH]
        if invalid:
            raise ValueError(f"Invalid entries in warp_types: {invalid}. Allowed: {sorted(_WARP_DISPATCH)}")

    s_local = []
    for p, tau in enumerate(tau_global):
        tau_scaled = _safe_minmax_scale(tau)

        if warp_mode == "none" or warp_strength == 0:
            tau_warped = tau_scaled
        elif warp_mode == "mixed":
            mode_p = warp_types[p]
            group_p = patient_groups[p] if patient_groups is not None else None
            tau_warped = _WARP_DISPATCH[mode_p](tau_scaled, warp_strength, rng, group_p)
        else:
            group_p = patient_groups[p] if patient_groups is not None else None
            tau_warped = _WARP_DISPATCH[warp_mode](tau_scaled, warp_strength, rng, group_p)

        # Final normalization keeps output in [0, 1] regardless of transform details.
        s_local.append(_safe_minmax_scale(tau_warped))

    return s_local


def _validate_simulation_options(eps, warp_mode, warp_strength, warp_types=None, tau_global_len=None):
    """Validate simulator options for pseudotime noise and warping."""
    if not (0.0 <= eps <= 1.0):
        raise ValueError("eps must be in [0, 1].")

    if warp_mode not in _ALLOWED_WARP_MODES:
        raise ValueError(f"Unknown warp_mode: {warp_mode!r}. Allowed: {sorted(_ALLOWED_WARP_MODES)}")

    if warp_strength < 0:
        raise ValueError("warp_strength must be non-negative.")

    if warp_mode == "mixed":
        if warp_types is None:
            raise ValueError("warp_types must be provided when warp_mode='mixed'.")
        if tau_global_len is not None and len(warp_types) != tau_global_len:
            raise ValueError("Length of warp_types must equal number of patients P.")
        invalid = [m for m in warp_types if m not in _WARP_DISPATCH]
        if invalid:
            raise ValueError(f"Invalid entries in warp_types: {invalid}. Allowed: {sorted(_WARP_DISPATCH)}")


def generate_observed_expression(time_grid, patient_trajectories, tau_global, sigma_noise, rng, noise_settings=None):
    """Generate noisy observed gene expression by interpolating patient trajectories.

    Parameters
    ----------
    time_grid : np.ndarray
        1D array of pseudotime grid points of shape (T,).
    patient_trajectories : list of np.ndarray
        List of P arrays of shape (K, T) with patient-specific trajectories.
    tau_global : list of np.ndarray
        List of P arrays of global pseudotimes, each of shape (N_p,).
    sigma_noise : float
        Standard deviation of the Gaussian observation noise.
    rng : np.random.Generator
        NumPy random generator.
    noise_settings : dict or None, optional
        Optional dictionary controlling the noise model. Supported keys:

        - ``'model'`` : ``'gaussian'`` | ``'student_t'`` | ``'laplace'``
          (default ``'gaussian'``).
        - ``'df'`` : int, degrees of freedom for ``'student_t'`` (default 3).
        - ``'heteroscedastic'`` : bool, scale noise by distance from centre
          (default ``False``).
        - ``'hetero_base'`` : float, base scale multiplier (default 0.8).
        - ``'hetero_amp'`` : float, amplitude of scale variation (default 1.6).
        - ``'outlier_frac'`` : float in [0, 1), fraction of cells receiving
          additional Laplace spike noise (default 0.0).
        - ``'dropout_frac'`` : float in [0, 1), fraction of cells zeroed out
          per gene (default 0.0).
        - ``'patient_offset_scale'`` : float, std. dev. of additive per-patient
          per-gene offset (default 0.0).

        ``None`` uses simple isotropic Gaussian noise (prior behaviour).

    Returns
    -------
    X : list of np.ndarray
        List of P arrays, each of shape (N_p, K), with observed expression values.
    """
    if noise_settings is None:
        noise_settings = {}

    noise_model = noise_settings.get("model", "gaussian")
    df = int(noise_settings.get("df", 3))
    heteroscedastic = bool(noise_settings.get("heteroscedastic", False))
    hetero_base = float(noise_settings.get("hetero_base", 0.8))
    hetero_amp = float(noise_settings.get("hetero_amp", 1.6))
    outlier_frac = float(noise_settings.get("outlier_frac", 0.0))
    dropout_frac = float(noise_settings.get("dropout_frac", 0.0))
    offset_scale = float(noise_settings.get("patient_offset_scale", 0.0))

    if noise_model not in {"gaussian", "student_t", "laplace"}:
        raise ValueError(f"Unknown noise model: {noise_model!r}. Allowed: 'gaussian', 'student_t', 'laplace'.")

    P = len(patient_trajectories)
    X = []

    for p in range(P):
        f_p = patient_trajectories[p]  # (K, T)
        tau_p = tau_global[p]  # (N_p,)
        n_cells = len(tau_p)
        X_p = np.zeros((n_cells, f_p.shape[0]))

        per_gene_offset = rng.normal(0, offset_scale, size=f_p.shape[0]) if offset_scale > 0 else np.zeros(f_p.shape[0])

        for g in range(f_p.shape[0]):
            mean_vals = np.interp(tau_p, time_grid, f_p[g]) + per_gene_offset[g]

            if heteroscedastic:
                scale_vec = sigma_noise * (hetero_base + hetero_amp * np.abs(tau_p - 0.5))
            else:
                scale_vec = np.full(n_cells, sigma_noise)

            if noise_model == "gaussian":
                noise = rng.normal(0, 1, size=n_cells) * scale_vec
            elif noise_model == "student_t":
                noise = rng.standard_t(df, size=n_cells) * scale_vec
            else:  # laplace
                noise = rng.laplace(0, 1, size=n_cells) * scale_vec

            if outlier_frac > 0:
                spike_mask = rng.random(n_cells) < outlier_frac
                noise[spike_mask] += rng.laplace(0, sigma_noise * 3.0, size=int(spike_mask.sum()))

            vals = mean_vals + noise

            if dropout_frac > 0:
                drop_mask = rng.random(n_cells) < dropout_frac
                vals[drop_mask] = 0.0

            X_p[:, g] = vals

        X.append(X_p)

    return X  # list of (N_p, K)


def initialize_structured_loadings(K, M=4, rng=None, strength=1.0):
    """Create a structured loading matrix W (K x M).

    Genes are evenly divided into monotonic up, monotonic down, bump, and
    flat groups. M should be 4 for the four factors: [up, down, bump, flat].

    Parameters
    ----------
    K : int
        Number of genes.
    M : int
        Number of latent factors (must be 4).
    rng : np.random.Generator, optional
        NumPy random generator. A new one is created if not provided.
    strength : float, optional
        Controls the baseline magnitude of the factor loadings. Default is 1.0.

    Returns
    -------
    W : np.ndarray
        Structured loading matrix of shape (K, M).
    gene_categories : list of str
        Category assignment for each gene (``'up'``, ``'down'``,
        ``'bump'``, or ``'flat'``).
    """
    if rng is None:
        rng = np.random.default_rng()

    assert M == 4, "This structured initializer assumes 4 latent factors."

    # allocate
    W = np.zeros((K, M))
    gene_categories = []

    # number of genes per group
    group_size = K // 4
    extra = K % 4  # in case K is not divisible by 4

    categories = ["up", "down", "bump", "flat"]

    # assign genes
    start = 0
    for i, cat in enumerate(categories):
        end = start + group_size + (1 if i < extra else 0)

        if cat == "up":
            # high positive loadings on factor 0
            W[start:end, 0] = rng.normal(strength, 0.1, end - start)
        elif cat == "down":
            # high *negative* loadings on factor 0 OR positive on factor 1
            # here we use factor 1 for clarity
            W[start:end, 1] = rng.normal(strength, 0.1, end - start)
        elif cat == "bump":
            # load on factor 2
            W[start:end, 2] = rng.normal(strength, 0.1, end - start)
        elif cat == "flat":
            # very small loadings on all factors
            W[start:end] = rng.normal(0, 0.05, size=(end - start, M))

        gene_categories.extend([cat] * (end - start))

        start = end

    return W, gene_categories


def simulate_LF_MOGP(
    K,
    P,
    M,
    time_grid,
    factor_kernels,
    deviation_kernel,
    W,
    gene_categories,
    patient_groups,
    lambda_cells,
    sigma_noise,
    rng=None,
    eps=0.0,
    warp_mode="none",
    warp_strength=0.2,
    warp_types=None,
    noise_settings=None,
):
    """Full generative model for synthetic single-cell data via LF-MOGP.

    Executes the complete simulation pipeline: sample latent factors,
    build global and patient-specific trajectories, assign pseudotimes,
    and generate noisy observed expression, returning an AnnData object.

    Parameters
    ----------
    K : int
        Number of genes.
    P : int
        Number of patients.
    M : int
        Number of latent factors.
    time_grid : np.ndarray
        1D array of pseudotime grid points of shape (T,).
    factor_kernels : list of dict
        Per-factor kernel parameters (unused directly; latent factors are
        structured via :func:`sample_structured_latent_factors`).
    deviation_kernel : dict
        Kernel parameters with keys ``'lengthscale'`` and ``'variance'``
        for patient-specific GP deviations.
    W : np.ndarray
        Gene loading matrix of shape (K, M).
    gene_categories : list of str
        Category label per gene (``'up'``, ``'down'``, ``'bump'``, ``'flat'``).
    patient_groups : list of str
        Pseudotime distribution type per patient
        (``'early'``, ``'late'``, ``'transition'``, ``'bimodal'``, ``'full'``).
    lambda_cells : float or list of float
        Expected number of cells per patient (Poisson rate). When a list is
        supplied, each entry is used as the rate for the corresponding patient.
    sigma_noise : float
        Standard deviation of the Gaussian observation noise.
    rng : np.random.Generator, optional
        NumPy random generator. A new one is created if not provided.
    eps : float, optional
        Fraction of cells mixed with uniform pseudotime noise during global
        pseudotime sampling. Default is 0.0 to preserve prior behavior.
    warp_mode : str, optional
        Optional local pseudotime warping mode. One of ``'none'``,
        ``'random_monotone'``, ``'patient_group'``, ``'sigmoid'``,
        ``'nonlinear'``, ``'nonlinear_inverse'``, or ``'mixed'``.
        Default is ``'none'``. When ``'mixed'``, ``warp_types`` must be
        provided.
    warp_strength : float, optional
        Non-negative warping strength used when ``warp_mode != 'none'``.
    warp_types : list of str or None, optional
        Per-patient warp mode names used when ``warp_mode='mixed'``.
        Length must equal ``P``.
    noise_settings : dict or None, optional
        Optional noise configuration passed to
        :func:`generate_observed_expression`. ``None`` uses simple isotropic
        Gaussian noise.

    Returns
    -------
    adata : anndata.AnnData
        Simulated dataset. ``adata.X`` holds observed expression
        (total_cells × K). ``adata.obs`` contains ``'patient'``,
        ``'tau_global'``, ``'s_local'``, ``'cell_index_within_patient'``,
        and ``'patient_group'``. ``adata.var`` contains ``'gene_ids'`` and
        ``'gene_category'``. ``adata.uns`` stores simulation components
        including latent factors, trajectories, and loading matrix.
    """
    if rng is None:
        rng = np.random.default_rng()

    _validate_simulation_options(
        eps=eps,
        warp_mode=warp_mode,
        warp_strength=warp_strength,
        warp_types=warp_types,
        tau_global_len=P,
    )

    # 1. Latent factors
    U = sample_structured_latent_factors(time_grid, rng)

    # 2. Global trajectory f(t)
    f_global = construct_global_trajectory(U, W)

    # 3. Patient trajectories f_p(t)
    f_p = construct_patient_trajectories(f_global, time_grid, K, P, deviation_kernel, rng)

    # 4. Cell counts — lambda_cells may be scalar or per-patient list
    if np.isscalar(lambda_cells):
        N_cells = sample_cell_counts(P, lambda_cells, rng)
    else:
        lambda_cells = list(lambda_cells)
        if len(lambda_cells) != P:
            raise ValueError(f"lambda_cells list length ({len(lambda_cells)}) must equal P ({P}).")
        N_cells = np.array([rng.poisson(lam) for lam in lambda_cells], dtype=int)

    # 5. Global pseudotimes τ
    tau_global = sample_global_pseudotimes(patient_groups, N_cells, rng, eps=eps)

    # 6. Local pseudotimes s
    s_local = compute_local_pseudotime(
        tau_global,
        patient_groups=patient_groups,
        warp_mode=warp_mode,
        warp_strength=warp_strength,
        warp_types=warp_types,
        rng=rng,
    )

    # 7. Observed expression
    X = generate_observed_expression(time_grid, f_p, tau_global, sigma_noise, rng, noise_settings=noise_settings)

    # 8. Build an AnnData object

    # Concatenate all patients' matrices
    X_concat = np.vstack(X)  # total_cells × K

    # Construct obs dataframe-like mapping
    obs_patient = []
    obs_tau = []
    obs_s = []
    obs_cell_within = []
    obs_patient_group = []

    for p in range(P):
        np_p = N_cells[p]
        obs_patient.extend([f"patient_{p}"] * np_p)
        obs_tau.extend(tau_global[p])
        obs_s.extend(s_local[p])
        obs_cell_within.extend(list(range(np_p)))
        obs_patient_group.extend([patient_groups[p]] * np_p)

    obs = {
        "patient": np.array(obs_patient, dtype=str),
        "tau_global": np.array(obs_tau, dtype=float),
        "s_local": np.array(obs_s, dtype=float),
        "cell_index_within_patient": np.array(obs_cell_within, dtype=int),
        "patient_group": np.array(obs_patient_group, dtype=str),
    }

    # gene-level metadata
    var = {
        "gene_ids": np.array([f"gene_{k}" for k in range(K)], dtype=str),
        "gene_category": np.array(gene_categories, dtype=str),
    }

    # Additional model components saved in .uns
    uns = {
        "time_grid": np.array(time_grid),
        "latent_factors": U,  # (M,T)
        "global_trajectory": f_global,  # (K,T)
        "warp_types": list(warp_types) if warp_types is not None else None,
        "patient_trajectories": f_p,  # list length P
        "loading_matrix": W,
        "patient_groups": np.array(patient_groups, dtype=str),
        "N_cells_per_patient": N_cells,
        "simulation_options": {
            "eps": float(eps),
            "warp_mode": warp_mode,
            "warp_strength": float(warp_strength),
        },
    }

    # Construct AnnData
    adata = ad.AnnData(
        X=X_concat,
        obs=obs,
        var=var,
        uns=uns,
    )

    return adata
