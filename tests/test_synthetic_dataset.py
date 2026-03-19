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
