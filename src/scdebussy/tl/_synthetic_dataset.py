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


def compute_local_pseudotime(
    tau_global,
    patient_groups=None,
    warp_mode="none",
    warp_strength=0.2,
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
        or ``'patient_group'``. Default is ``'none'``.
    warp_strength : float, optional
        Non-negative strength of warping. Ignored when ``warp_mode='none'``.
    rng : np.random.Generator, optional
        NumPy random generator used by stochastic warping modes.

    Returns
    -------
    s_local : list of np.ndarray
        List of P arrays of local pseudotimes normalized to [0, 1].
    """
    allowed_modes = {"none", "random_monotone", "patient_group"}
    if warp_mode not in allowed_modes:
        raise ValueError(f"Unknown warp_mode: {warp_mode!r}. Allowed: {sorted(allowed_modes)}")
    if warp_strength < 0:
        raise ValueError("warp_strength must be non-negative.")

    if rng is None:
        rng = np.random.default_rng()

    if warp_mode == "patient_group":
        if patient_groups is None:
            raise ValueError("patient_groups must be provided when warp_mode='patient_group'.")
        if len(patient_groups) != len(tau_global):
            raise ValueError("Length of patient_groups must match tau_global.")

    s_local = []
    for p, tau in enumerate(tau_global):
        tau_scaled = _safe_minmax_scale(tau)

        if warp_mode == "none" or warp_strength == 0:
            tau_warped = tau_scaled
        elif warp_mode == "random_monotone":
            tau_warped = _random_monotone_warp(tau_scaled, rng=rng, warp_strength=warp_strength)
        else:
            tau_warped = _group_warp(
                tau_scaled,
                group=patient_groups[p],
                rng=rng,
                warp_strength=warp_strength,
            )

        # Final normalization keeps output in [0, 1] regardless of transform details.
        s_local.append(_safe_minmax_scale(tau_warped))

    return s_local


def _validate_simulation_options(eps, warp_mode, warp_strength):
    """Validate simulator options for pseudotime noise and warping."""
    if not (0.0 <= eps <= 1.0):
        raise ValueError("eps must be in [0, 1].")

    allowed_modes = {"none", "random_monotone", "patient_group"}
    if warp_mode not in allowed_modes:
        raise ValueError(f"Unknown warp_mode: {warp_mode!r}. Allowed: {sorted(allowed_modes)}")

    if warp_strength < 0:
        raise ValueError("warp_strength must be non-negative.")


def generate_observed_expression(time_grid, patient_trajectories, tau_global, sigma_noise, rng):
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

    Returns
    -------
    X : list of np.ndarray
        List of P arrays, each of shape (N_p, K), with observed expression values.
    """
    P = len(patient_trajectories)  # Number of patients
    X = []  # Initialize general observed expression data

    for p in range(P):
        f_p = patient_trajectories[p]  # (K,T)
        tau_p = tau_global[p]  # (N_p,)

        # Interpolate each gene across τ
        X_p = np.zeros((len(tau_p), f_p.shape[0]))  # Initialize observed expression matrix per patient (N_p, K)

        for g in range(f_p.shape[0]):  # For each gene
            mean_vals = np.interp(
                tau_p, time_grid, f_p[g]
            )  # (Linear) Interpolation makes GP a continuous function over pseudotime given different time grids
            X_p[:, g] = rng.normal(
                mean_vals, sigma_noise
            )  # Even if cells at same pseudotime, gene expression values will differ - add Gaussian noise to simulate measurement noise

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
    lambda_cells : float
        Expected number of cells per patient (Poisson rate).
    sigma_noise : float
        Standard deviation of the Gaussian observation noise.
    rng : np.random.Generator, optional
        NumPy random generator. A new one is created if not provided.
    eps : float, optional
        Fraction of cells mixed with uniform pseudotime noise during global
        pseudotime sampling. Default is 0.0 to preserve prior behavior.
    warp_mode : str, optional
        Optional local pseudotime warping mode. One of ``'none'``,
        ``'random_monotone'``, or ``'patient_group'``. Default is ``'none'``.
    warp_strength : float, optional
        Non-negative warping strength used when ``warp_mode != 'none'``.

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

    _validate_simulation_options(eps=eps, warp_mode=warp_mode, warp_strength=warp_strength)

    # 1. Latent factors
    U = sample_structured_latent_factors(time_grid, rng)

    # 2. Global trajectory f(t)
    f_global = construct_global_trajectory(U, W)

    # 3. Patient trajectories f_p(t)
    f_p = construct_patient_trajectories(f_global, time_grid, K, P, deviation_kernel, rng)

    # 4. Cell counts
    N_cells = sample_cell_counts(P, lambda_cells, rng)

    # 5. Global pseudotimes τ
    tau_global = sample_global_pseudotimes(patient_groups, N_cells, rng, eps=eps)

    # 6. Local pseudotimes s
    s_local = compute_local_pseudotime(
        tau_global,
        patient_groups=patient_groups,
        warp_mode=warp_mode,
        warp_strength=warp_strength,
        rng=rng,
    )

    # 7. Observed expression
    X = generate_observed_expression(time_grid, f_p, tau_global, sigma_noise, rng)

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
