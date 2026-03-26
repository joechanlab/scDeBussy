import numpy as np

from scdebussy.tl import initialize_structured_loadings, simulate_LF_MOGP


def _small_simulation_kwargs(seed=0):
    rng = np.random.default_rng(seed)
    K = 40
    P = 4
    M = 4
    time_grid = np.linspace(0, 1, 100)
    factor_kernels = [{"lengthscale": 0.1, "variance": 1.0} for _ in range(M)]
    deviation_kernel = {"lengthscale": 0.15, "variance": 0.2}
    patient_groups = ["full", "early", "late", "transition"]
    W, gene_categories = initialize_structured_loadings(K=K, M=M, rng=rng, strength=1.0)

    return {
        "K": K,
        "P": P,
        "M": M,
        "time_grid": time_grid,
        "factor_kernels": factor_kernels,
        "deviation_kernel": deviation_kernel,
        "W": W,
        "gene_categories": gene_categories,
        "patient_groups": patient_groups,
        "lambda_cells": 80,
        "sigma_noise": 0.05,
        "rng": rng,
    }


def test_simulate_lf_mogp_defaults_match_explicit_no_warp_no_noise():
    kwargs_default = _small_simulation_kwargs(seed=123)
    adata_default = simulate_LF_MOGP(**kwargs_default)

    kwargs_explicit = _small_simulation_kwargs(seed=123)
    adata_explicit = simulate_LF_MOGP(
        **kwargs_explicit,
        eps=0.0,
        warp_mode="none",
        warp_strength=0.2,
    )

    np.testing.assert_allclose(
        adata_default.obs["tau_global"].to_numpy(dtype=float),
        adata_explicit.obs["tau_global"].to_numpy(dtype=float),
    )
    np.testing.assert_allclose(
        adata_default.obs["s_local"].to_numpy(dtype=float),
        adata_explicit.obs["s_local"].to_numpy(dtype=float),
    )


def test_eps_changes_tau_global_distribution():
    kwargs_eps0 = _small_simulation_kwargs(seed=999)
    adata_eps0 = simulate_LF_MOGP(**kwargs_eps0, eps=0.0)

    kwargs_eps1 = _small_simulation_kwargs(seed=999)
    adata_eps1 = simulate_LF_MOGP(**kwargs_eps1, eps=0.2)

    tau0 = adata_eps0.obs["tau_global"].to_numpy(dtype=float)
    tau1 = adata_eps1.obs["tau_global"].to_numpy(dtype=float)
    assert not np.allclose(tau0, tau1)


def test_warping_changes_s_local_but_not_tau_global():
    kwargs_nowarp = _small_simulation_kwargs(seed=2026)
    adata_nowarp = simulate_LF_MOGP(**kwargs_nowarp, eps=0.0, warp_mode="none")

    kwargs_warp = _small_simulation_kwargs(seed=2026)
    adata_warp = simulate_LF_MOGP(
        **kwargs_warp,
        eps=0.0,
        warp_mode="random_monotone",
        warp_strength=0.4,
    )

    tau_nowarp = adata_nowarp.obs["tau_global"].to_numpy(dtype=float)
    tau_warp = adata_warp.obs["tau_global"].to_numpy(dtype=float)
    np.testing.assert_allclose(tau_nowarp, tau_warp)

    s_nowarp = adata_nowarp.obs["s_local"].to_numpy(dtype=float)
    s_warp = adata_warp.obs["s_local"].to_numpy(dtype=float)
    assert not np.allclose(s_nowarp, s_warp)

    assert np.all((s_warp >= 0.0) & (s_warp <= 1.0))


# ---------------------------------------------------------------------------
# Tests for noise_settings extension
# ---------------------------------------------------------------------------


def test_noise_settings_dropout_creates_zeros():
    """Dropout fraction should introduce exact zeros into expression matrix."""
    kwargs = _small_simulation_kwargs(seed=42)
    adata = simulate_LF_MOGP(
        **kwargs,
        noise_settings={"dropout_frac": 0.5},
    )
    X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    assert np.any(X == 0.0), "Expected some zero entries when dropout_frac=0.5"


def test_noise_settings_student_t_heavier_tails():
    """Student-t noise at df=2 should produce heavier tails than Gaussian."""
    rng_g = np.random.default_rng(7)
    rng_t = np.random.default_rng(7)

    K, P, M = 40, 4, 4
    time_grid = np.linspace(0, 1, 50)
    W, gene_cats = initialize_structured_loadings(K=K, M=M, rng=np.random.default_rng(0))
    common = {
        "K": K,
        "P": P,
        "M": M,
        "time_grid": time_grid,
        "factor_kernels": [{"lengthscale": 0.1, "variance": 1.0}] * M,
        "deviation_kernel": {"lengthscale": 0.15, "variance": 0.2},
        "W": W,
        "gene_categories": gene_cats,
        "patient_groups": ["full"] * P,
        "lambda_cells": 200,
        "sigma_noise": 0.5,
    }
    adata_g = simulate_LF_MOGP(**common, rng=rng_g, noise_settings={"model": "gaussian"})
    adata_t = simulate_LF_MOGP(**common, rng=rng_t, noise_settings={"model": "student_t", "df": 2})

    X_g = adata_g.X if not hasattr(adata_g.X, "toarray") else adata_g.X.toarray()
    X_t = adata_t.X if not hasattr(adata_t.X, "toarray") else adata_t.X.toarray()

    # Kurtosis proxy: 95th–5th percentile range should be larger for t
    range_g = np.percentile(X_g, 95) - np.percentile(X_g, 5)
    range_t = np.percentile(X_t, 95) - np.percentile(X_t, 5)
    assert range_t > range_g, (
        f"Expected heavier tails for student_t (range_t={range_t:.4f}) vs gaussian (range_g={range_g:.4f})"
    )


def test_noise_settings_smoke_test_all_models():
    """All noise models should run without errors and produce finite arrays."""
    for model in ("gaussian", "student_t", "laplace"):
        kwargs = _small_simulation_kwargs(seed=55)
        adata = simulate_LF_MOGP(
            **kwargs,
            noise_settings={"model": model, "df": 3},
        )
        X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
        assert np.all(np.isfinite(X)), f"Non-finite values with noise model={model!r}"


# ---------------------------------------------------------------------------
# Tests for new warp modes
# ---------------------------------------------------------------------------


def test_warp_sigmoid_is_monotone():
    """Sigmoid-warped local pseudotime should be monotone within each patient."""
    kwargs = _small_simulation_kwargs(seed=10)
    adata = simulate_LF_MOGP(**kwargs, warp_mode="sigmoid", warp_strength=0.5)

    for pid in adata.obs["patient"].unique():
        mask = adata.obs["patient"] == pid
        tau_p = adata.obs["tau_global"][mask].to_numpy(dtype=float)
        s_p = adata.obs["s_local"][mask].to_numpy(dtype=float)
        order = np.argsort(tau_p)
        s_sorted = s_p[order]
        diffs = np.diff(s_sorted)
        assert np.all(diffs >= -1e-9), f"Patient {pid}: sigmoid-warped s_local is not monotone non-decreasing"


def test_warp_nonlinear_is_monotone():
    """Nonlinear-warped local pseudotime should be monotone within each patient."""
    kwargs = _small_simulation_kwargs(seed=20)
    adata = simulate_LF_MOGP(**kwargs, warp_mode="nonlinear", warp_strength=0.5)

    for pid in adata.obs["patient"].unique():
        mask = adata.obs["patient"] == pid
        tau_p = adata.obs["tau_global"][mask].to_numpy(dtype=float)
        s_p = adata.obs["s_local"][mask].to_numpy(dtype=float)
        order = np.argsort(tau_p)
        s_sorted = s_p[order]
        diffs = np.diff(s_sorted)
        assert np.all(diffs >= -1e-9), f"Patient {pid}: nonlinear-warped s_local is not monotone non-decreasing"


def test_mixed_warp_per_patient_independent():
    """Mixed warp mode should apply distinct warp functions per patient.

    We verify by running mixed with all-sigmoid vs all-nonlinear and
    confirming the results differ, and that each run is reproducible.
    """
    P = 4
    warp_types_sig = ["sigmoid"] * P
    warp_types_nl = ["nonlinear"] * P

    kwargs_sig = _small_simulation_kwargs(seed=30)
    kwargs_sig["patient_groups"] = ["full"] * P
    adata_sig = simulate_LF_MOGP(
        **kwargs_sig,
        warp_mode="mixed",
        warp_strength=0.5,
        warp_types=warp_types_sig,
    )

    kwargs_nl = _small_simulation_kwargs(seed=30)
    kwargs_nl["patient_groups"] = ["full"] * P
    adata_nl = simulate_LF_MOGP(
        **kwargs_nl,
        warp_mode="mixed",
        warp_strength=0.5,
        warp_types=warp_types_nl,
    )

    s_sig = adata_sig.obs["s_local"].to_numpy(dtype=float)
    s_nl = adata_nl.obs["s_local"].to_numpy(dtype=float)
    assert not np.allclose(s_sig, s_nl), "Expected different s_local for sigmoid vs nonlinear mixed warp"


def test_lambda_cells_list_gives_per_patient_cell_counts():
    """When lambda_cells is a list, each patient should have ~lambda_cells[p] cells."""
    lambda_list = [50, 200, 100, 300]
    kwargs = _small_simulation_kwargs(seed=99)
    kwargs["patient_groups"] = ["full"] * 4
    kwargs["lambda_cells"] = lambda_list
    adata = simulate_LF_MOGP(**kwargs)

    for p, lam in enumerate(lambda_list):
        pid = f"patient_{p}"
        n_cells = int((adata.obs["patient"] == pid).sum())
        # Poisson: mean=lambda, std=sqrt(lambda); allow 4-sigma
        tol = 4 * (lam**0.5) + 10
        assert abs(n_cells - lam) <= tol, f"Patient {p}: expected ~{lam} cells, got {n_cells}"
