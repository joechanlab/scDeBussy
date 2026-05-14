"""
Compare NW vs LLR interpolation on toy datasets.

Tests cover:
- Return shape and non-negativity of variance
- Boundary bias: LLR should be more accurate at tau=0 and tau=1 than NW
- Adaptive bandwidth changes the smoothed output
- Variance is larger in sparse regions than dense ones (adaptive bandwidth case)
- Full scDeBussy pipeline runs with both methods and stores patient_variances
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from scdebussy.tl import scDeBussy, smooth_patient_trajectory

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _linear_patient(n_cells=400, noise=0.05, seed=0):
    """Single-patient AnnData where gene0 = s and gene1 = 1-s (known boundary values)."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(0, 1, n_cells)
    X = np.column_stack([s, 1.0 - s]) + rng.normal(0, noise, (n_cells, 2))
    obs = pd.DataFrame({"patient": "p0", "s_local": s}, index=[f"c{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=["g0", "g1"])
    return ad.AnnData(X=X, obs=obs, var=var)


def _two_patient_adata(seed=1):
    """Two patients with linear trajectories suitable for scDeBussy."""
    rng = np.random.default_rng(seed)
    results = []
    for pid in range(2):
        n = rng.integers(40, 80)
        s = rng.uniform(0, 1, n)
        X = np.column_stack([s, 1.0 - s]) + rng.normal(0, 0.05, (n, 2))
        obs = pd.DataFrame(
            {"patient": f"patient_{pid}", "s_local": np.clip(s, 0, 1)},
            index=[f"p{pid}_c{i}" for i in range(n)],
        )
        results.append(ad.AnnData(X=X, obs=obs))
    adata = ad.concat(results)
    adata.var_names = ["g0", "g1"]
    return adata


# ---------------------------------------------------------------------------
# Shape and type tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["nw", "llr"])
def test_smooth_returns_three_tensors(method):
    adata = _linear_patient()
    out = smooth_patient_trajectory(adata, "p0", method=method, n_bins=20, bandwidth=0.1)
    assert len(out) == 3, "expected (x_smoothed, density, x_variance)"
    x_smoothed, density, x_variance = out
    assert isinstance(x_smoothed, torch.Tensor)
    assert isinstance(density, torch.Tensor)
    assert isinstance(x_variance, torch.Tensor)


@pytest.mark.parametrize("method", ["nw", "llr"])
def test_smooth_output_shapes(method):
    adata = _linear_patient()
    n_bins, n_genes = 30, 2
    x_smoothed, density, x_variance = smooth_patient_trajectory(
        adata, "p0", method=method, n_bins=n_bins, bandwidth=0.1
    )
    assert x_smoothed.shape == (n_bins, n_genes)
    assert density.shape == (n_bins,)
    assert x_variance.shape == (n_bins, n_genes)


@pytest.mark.parametrize("method", ["nw", "llr"])
def test_variance_non_negative(method):
    adata = _linear_patient()
    _, _, x_variance = smooth_patient_trajectory(adata, "p0", method=method, n_bins=50, bandwidth=0.1)
    assert (x_variance >= 0).all(), "variance must be non-negative"


# ---------------------------------------------------------------------------
# Boundary bias test — core motivation for LLR
# ---------------------------------------------------------------------------


def test_llr_has_smaller_boundary_error_than_nw():
    """LLR should be more accurate at tau=0 and tau=1 than NW.

    With f(s)=s, the true value is 0 at tau=0 and 1 at tau=1.
    The NW estimator is biased inward at the boundaries because the kernel
    has no support for s<0 or s>1.  LLR corrects this via the local slope.
    """
    adata = _linear_patient(n_cells=600, noise=0.02, seed=42)
    n_bins = 100

    nw_smoothed, _, _ = smooth_patient_trajectory(adata, "p0", method="nw", n_bins=n_bins, bandwidth=0.08)
    llr_smoothed, _, _ = smooth_patient_trajectory(adata, "p0", method="llr", n_bins=n_bins, bandwidth=0.08)

    nw = nw_smoothed.numpy()
    llr = llr_smoothed.numpy()

    # gene0: true value at tau=0 is 0.0, at tau=1 is 1.0
    boundary_bins = 5  # first and last 5 bins
    true_left, true_right = 0.0, 1.0

    nw_left_err = abs(nw[:boundary_bins, 0].mean() - true_left)
    llr_left_err = abs(llr[:boundary_bins, 0].mean() - true_left)
    nw_right_err = abs(nw[-boundary_bins:, 0].mean() - true_right)
    llr_right_err = abs(llr[-boundary_bins:, 0].mean() - true_right)

    assert llr_left_err < nw_left_err, (
        f"LLR left-boundary error ({llr_left_err:.4f}) should be smaller than NW ({nw_left_err:.4f})"
    )
    assert llr_right_err < nw_right_err, (
        f"LLR right-boundary error ({llr_right_err:.4f}) should be smaller than NW ({nw_right_err:.4f})"
    )


# ---------------------------------------------------------------------------
# Adaptive bandwidth test
# ---------------------------------------------------------------------------


def test_adaptive_bandwidth_changes_output():
    """Adaptive and fixed bandwidth should produce different smoothed trajectories.

    With n=50 cells and knn_k=15, the 15th-nearest-neighbour distance is ~0.3 on
    average, well above the floor bandwidth=0.05.  The balloon estimator therefore
    uses a wider kernel than the fixed-bandwidth version, producing different output.
    """
    adata = _linear_patient(n_cells=50, noise=0.02, seed=3)
    fixed, _, _ = smooth_patient_trajectory(
        adata, "p0", method="llr", n_bins=50, bandwidth=0.05, adaptive_bandwidth=False
    )
    adaptive, _, _ = smooth_patient_trajectory(
        adata, "p0", method="llr", n_bins=50, bandwidth=0.05, adaptive_bandwidth=True, knn_k=15
    )
    assert not torch.allclose(fixed, adaptive), "adaptive bandwidth should alter the smoothed output"


def test_adaptive_bandwidth_variance_larger_in_sparse_region():
    """In a patient with cells only in [0, 0.5], variance should be larger in [0.5, 1]."""
    rng = np.random.default_rng(7)
    n = 200
    s = rng.uniform(0, 0.5, n)  # cells only in early pseudotime
    X = np.column_stack([s, 1.0 - s]) + rng.normal(0, 0.05, (n, 2))
    obs = pd.DataFrame({"patient": "p0", "s_local": s}, index=[f"c{i}" for i in range(n)])
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=["g0", "g1"]))

    _, _, var_adaptive = smooth_patient_trajectory(
        adata, "p0", method="llr", n_bins=50, bandwidth=0.05, adaptive_bandwidth=True, knn_k=10
    )
    var = var_adaptive.numpy()
    dense_var = var[:25, :].mean()  # bins in [0, 0.5] — covered by data
    sparse_var = var[25:, :].mean()  # bins in [0.5, 1] — no cells
    assert sparse_var > dense_var, (
        f"sparse region variance ({sparse_var:.4f}) should exceed dense region ({dense_var:.4f})"
    )


# ---------------------------------------------------------------------------
# Invalid input tests
# ---------------------------------------------------------------------------


def test_smooth_invalid_method_raises():
    adata = _linear_patient()
    with pytest.raises(ValueError, match="method must be"):
        smooth_patient_trajectory(adata, "p0", method="cubic")


def test_smooth_missing_patient_raises():
    adata = _linear_patient()
    with pytest.raises(ValueError, match="No cells found"):
        smooth_patient_trajectory(adata, "nonexistent", method="llr")


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["nw", "llr"])
def test_scdebussy_runs_with_both_methods(method):
    adata = _two_patient_adata()
    result = scDeBussy(adata, n_bins=20, bandwidth=0.15, max_iter=2, method=method)
    assert "aligned_pseudotime" in result.obs
    assert result.obs["aligned_pseudotime"].notna().all()


def test_scdebussy_stores_patient_variances():
    adata = _two_patient_adata()
    scDeBussy(adata, n_bins=20, bandwidth=0.15, max_iter=2, method="llr")
    bary = adata.uns["barycenter"]
    assert "patient_variances" in bary
    for pid in ["patient_0", "patient_1"]:
        assert pid in bary["patient_variances"]
        arr = bary["patient_variances"][pid]
        assert arr.shape == (20, 2)
        assert (arr >= 0).all()


def test_scdebussy_stores_method_in_params():
    adata = _two_patient_adata()
    scDeBussy(adata, n_bins=20, bandwidth=0.15, max_iter=2, method="nw")
    assert adata.uns["barycenter"]["params"]["method"] == "nw"


def test_scdebussy_invalid_method_raises():
    adata = _two_patient_adata()
    with pytest.raises(ValueError, match="method must be"):
        scDeBussy(adata, n_bins=20, bandwidth=0.15, max_iter=2, method="gp")


def test_scdebussy_adaptive_bandwidth_runs():
    adata = _two_patient_adata()
    result = scDeBussy(adata, n_bins=20, bandwidth=0.1, max_iter=2, method="llr", adaptive_bandwidth=True, knn_k=8)
    assert result.obs["aligned_pseudotime"].notna().all()
    assert adata.uns["barycenter"]["params"]["adaptive_bandwidth"] is True
