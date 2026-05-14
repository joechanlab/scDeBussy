"""
Tests for the soft-EM alignment (use_soft_em=True).

Correctness tests:
  - forward-backward rows sum to 1 (every patient bin visited exactly once)
  - forward-backward log_Z is finite for valid inputs
  - soft-EM costs are finite and show net improvement over all iterations
    (strict per-iteration monotonicity is not guaranteed because the M-step
    uses softdtw_barycenter, which optimises a different objective than -sum log Z)
  - distance matrix helpers produce correct values for known inputs

Performance test:
  - compare alignment RMSE to ground truth on simulate_LF_MOGP synthetic data
    between hard-EM (use_soft_em=False) and soft-EM (use_soft_em=True)
"""

import numpy as np
import pytest

from scdebussy.tl import initialize_structured_loadings, scDeBussy, simulate_LF_MOGP
from scdebussy.tl._alignment import (
    _compute_distance_matrix,
    _forward_backward_open,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic(seed=0, P=6, n_bins=40, warp_mode="random_monotone", warp_strength=0.3):
    rng = np.random.default_rng(seed)
    T = 50
    time_grid = np.linspace(0, 1, T)
    W, gene_categories = initialize_structured_loadings(K=12, M=4, rng=rng)
    patient_groups = ["full"] * P
    factor_kernels = [{"lengthscale": 0.3, "variance": 1.0}] * 4
    deviation_kernel = {"lengthscale": 0.2, "variance": 0.1}
    adata = simulate_LF_MOGP(
        K=12,
        P=P,
        M=4,
        time_grid=time_grid,
        factor_kernels=factor_kernels,
        deviation_kernel=deviation_kernel,
        W=W,
        gene_categories=gene_categories,
        patient_groups=patient_groups,
        lambda_cells=80,
        sigma_noise=0.1,
        rng=rng,
        warp_mode=warp_mode,
        warp_strength=warp_strength,
    )
    return adata


def _rmse(adata, key):
    truth = adata.obs["tau_global"].to_numpy(dtype=float)
    pred = adata.obs[key].to_numpy(dtype=float)
    mask = np.isfinite(truth) & np.isfinite(pred)
    return float(np.sqrt(np.mean((pred[mask] - truth[mask]) ** 2)))


# ---------------------------------------------------------------------------
# Unit tests: _compute_distance_matrix
# ---------------------------------------------------------------------------


def test_distance_matrix_sqeuclidean_identical_rows():
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    D = _compute_distance_matrix(A, A, "sqeuclidean")
    assert D.shape == (2, 2)
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-10)
    np.testing.assert_allclose(D[0, 1], 2.0, atol=1e-10)


def test_distance_matrix_cosine_orthogonal():
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    D = _compute_distance_matrix(A, A, "cosine")
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-6)
    np.testing.assert_allclose(D[0, 1], 1.0, atol=1e-6)


def test_distance_matrix_non_negative():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((10, 5))
    B = rng.standard_normal((10, 5))
    for method in ["sqeuclidean", "euclidean", "cosine", "cityblock"]:
        D = _compute_distance_matrix(A, B, method)
        assert (D >= 0).all(), f"{method} produced negative distances"


# ---------------------------------------------------------------------------
# Unit tests: _forward_backward_open
# ---------------------------------------------------------------------------


def test_forward_backward_rows_sum_to_one():
    """For asymmetric transitions, every patient bin is on exactly one path cell -> rows sum to 1."""
    rng = np.random.default_rng(2)
    T = 15
    delta = np.abs(rng.standard_normal((T, T))) + 0.1
    A_bar, log_Z = _forward_backward_open(delta, gamma=0.5, window_size=T)
    row_sums = A_bar.sum(axis=1)
    np.testing.assert_allclose(
        row_sums, 1.0, atol=1e-5, err_msg="Each patient bin must map to exactly one expected barycenter position"
    )


def test_forward_backward_log_z_finite():
    rng = np.random.default_rng(3)
    delta = np.abs(rng.standard_normal((20, 20)))
    _, log_Z = _forward_backward_open(delta, gamma=1.0, window_size=20)
    assert np.isfinite(log_Z), "log_Z should be finite for valid inputs"


def test_forward_backward_non_negative_A_bar():
    rng = np.random.default_rng(4)
    delta = np.abs(rng.standard_normal((12, 12)))
    A_bar, _ = _forward_backward_open(delta, gamma=1.0, window_size=12)
    assert (A_bar >= 0).all()


def test_forward_backward_sakoe_chiba_band():
    """Cells outside the Sakoe-Chiba band must have zero expected alignment weight."""
    T, W = 20, 3
    rng = np.random.default_rng(5)
    delta = np.abs(rng.standard_normal((T, T)))
    A_bar, _ = _forward_backward_open(delta, gamma=1.0, window_size=W)
    for t in range(T):
        for tp in range(T):
            if abs(t - tp) > W:
                assert A_bar[t, tp] == 0.0, f"A_bar[{t},{tp}] should be 0 (outside band)"


def test_forward_backward_low_gamma_approaches_viterbi():
    """With very small gamma, A_bar should be concentrated on a single path per row."""
    _rng = np.random.default_rng(6)
    T = 10
    # Identity-like delta: cheapest path is t'=t
    delta = np.abs(np.arange(T)[:, None] - np.arange(T)[None, :]).astype(float) * 2.0
    A_bar, _ = _forward_backward_open(delta, gamma=1e-4, window_size=T)
    # Each row should be nearly a one-hot vector
    row_entropy = -(A_bar * np.log(A_bar + 1e-30)).sum(axis=1)
    assert row_entropy.mean() < 0.1, "Low gamma should produce near-deterministic alignment"


# ---------------------------------------------------------------------------
# Soft-EM convergence: costs must be finite and show net improvement
# ---------------------------------------------------------------------------


def test_soft_em_costs_finite_and_improving():
    """Soft-EM costs must all be finite and the last cost must not exceed the first.

    Strict per-iteration monotonicity is not guaranteed because the M-step uses
    softdtw_barycenter (which optimises a different objective than -sum_p log Z_p),
    but the algorithm must make net progress over all iterations.
    """
    adata = _make_synthetic(seed=7, P=4, warp_strength=0.25)
    scDeBussy(
        adata,
        n_bins=30,
        bandwidth=0.12,
        gamma=0.5,
        max_iter=8,
        tol=0.0,  # disable early stopping so we see all iterations
        use_soft_em=True,
        verbose=False,
    )
    costs = adata.uns["barycenter"]["em_convergence"]["iteration_costs"]
    assert len(costs) >= 2, "Need at least 2 iterations to check convergence"
    assert all(np.isfinite(c) for c in costs), f"Some iteration costs are not finite: {costs}"
    # Net improvement: we are minimising, so the final cost must not exceed the initial
    # (additive tolerance handles costs near zero or of either sign)
    assert costs[-1] <= costs[0] + 1e-3 * abs(costs[0]), (
        f"Soft-EM failed to make net progress: first={costs[0]:.4f} last={costs[-1]:.4f}"
    )


# ---------------------------------------------------------------------------
# Pipeline smoke tests
# ---------------------------------------------------------------------------


def test_soft_em_runs_and_writes_aligned_pseudotime():
    adata = _make_synthetic(seed=8, P=4)
    result = scDeBussy(adata, n_bins=25, bandwidth=0.12, max_iter=5, use_soft_em=True)
    assert "aligned_pseudotime" in result.obs
    assert result.obs["aligned_pseudotime"].notna().all()


def test_soft_em_stores_use_soft_em_flag():
    adata = _make_synthetic(seed=9, P=4)
    scDeBussy(adata, n_bins=25, bandwidth=0.12, max_iter=3, use_soft_em=True)
    assert adata.uns["barycenter"]["params"]["use_soft_em"] is True


def test_hard_em_still_works_after_changes():
    adata = _make_synthetic(seed=10, P=4)
    scDeBussy(adata, n_bins=25, bandwidth=0.12, max_iter=3, use_soft_em=False)
    assert adata.obs["aligned_pseudotime"].notna().all()
    assert adata.uns["barycenter"]["params"]["use_soft_em"] is False


# ---------------------------------------------------------------------------
# Performance comparison on synthetic data with known ground truth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("warp_strength", [0.2, 0.4])
def test_soft_em_rmse_not_worse_than_hard_em(warp_strength):
    """Soft-EM should match or beat hard-EM RMSE against ground-truth pseudotime."""
    adata_hard = _make_synthetic(seed=42, P=8, warp_strength=warp_strength)
    adata_soft = _make_synthetic(seed=42, P=8, warp_strength=warp_strength)

    scDeBussy(adata_hard, n_bins=40, bandwidth=0.1, gamma=0.5, max_iter=15, tol=1e-4, use_soft_em=False)
    scDeBussy(adata_soft, n_bins=40, bandwidth=0.1, gamma=0.5, max_iter=15, tol=1e-4, use_soft_em=True)

    rmse_hard = _rmse(adata_hard, "aligned_pseudotime")
    rmse_soft = _rmse(adata_soft, "aligned_pseudotime")

    # Soft-EM should not be worse by more than 10% of hard-EM RMSE
    assert rmse_soft <= rmse_hard * 1.10, (
        f"warp_strength={warp_strength}: soft RMSE {rmse_soft:.4f} > 110% of hard RMSE {rmse_hard:.4f}"
    )

    print(
        f"\nwarp_strength={warp_strength}  hard RMSE={rmse_hard:.4f}  soft RMSE={rmse_soft:.4f}  "
        f"improvement={100 * (rmse_hard - rmse_soft) / rmse_hard:+.1f}%"
    )
