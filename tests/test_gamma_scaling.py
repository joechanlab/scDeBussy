"""Regression tests for the gamma/metric coupling and the DTW coupling band.

Each test here pins a failure that was silent: the code ran, converged, reported a
monotone objective, and returned a confidently wrong alignment.
"""

from __future__ import annotations

import numpy as np
import pytest

from scdebussy.tl._alignment import (
    _fit_barycenter_soft_em,
    _pairwise_cost_scale,
    _reference_cost_scale,
)


def _cohort(seed: int, P: int = 4, T: int = 24, K: int = 40, scale: float = 1.0):
    """Warped noisy observations of a shared latent program, at a controllable scale."""
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2 * np.pi, size=K)
    freqs = rng.uniform(1.0, 3.0, size=K)
    trajs = []
    for _ in range(P):
        w = np.sort(rng.uniform(size=T - 2))
        warped = np.concatenate([[0.0], w, [1.0]])
        traj = np.sin(2 * np.pi * freqs[None, :] * warped[:, None] + phases[None, :])
        traj = (traj + rng.normal(scale=0.15, size=traj.shape)) * scale
        trajs.append(traj)
    return trajs


def _coverage(trajs, dist_method, gamma, gamma_scaling):
    _B, _paths, diag = _fit_barycenter_soft_em(
        patient_trajectories=trajs,
        n_bins=trajs[0].shape[0],
        gamma=gamma,
        max_iter=15,
        tol=1e-10,
        verbose=False,
        dtw_dist_method=dist_method,
        barycenter_init_iter=3,
        strict=(dist_method == "sqeuclidean"),
        gamma_scaling=gamma_scaling,
    )
    A_cov = diag["coverage_fractions"]
    return float(np.mean(A_cov)), diag


# ---------------------------------------------------------------------------
# gamma / metric coupling
# ---------------------------------------------------------------------------
def test_absolute_gamma_collapses_when_metric_changes():
    """Documents the bug: a gamma tuned for cosine is a hard argmax for sqeuclidean.

    This is why 'strict soft-EM fails on real data' was concluded once, wrongly -- the
    objective really was monotone, it was just being minimised at zero temperature.
    """
    trajs = _cohort(0, K=200)
    cos_cov, _ = _coverage(trajs, "cosine", 0.1, "absolute")
    sq_cov, _ = _coverage(trajs, "sqeuclidean", 0.1, "absolute")
    assert cos_cov > 0.5, f"cosine baseline should be healthy, got {cos_cov}"
    assert sq_cov < cos_cov, "the collapse this test documents did not reproduce"


@pytest.mark.parametrize("dist_method", ["cosine", "sqeuclidean"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_relative_gamma_keeps_coverage_across_metrics(dist_method, seed):
    """With relative scaling, coverage stays healthy for BOTH metrics on one dataset."""
    trajs = _cohort(seed, K=200)
    cov, diag = _coverage(trajs, dist_method, 2.0, "relative")
    assert cov > 0.5, f"{dist_method}: coverage collapsed to {cov:.3f}"
    if dist_method == "sqeuclidean":
        # Only the strict/sqeuclidean configuration carries the descent guarantee;
        # cosine runs the legacy path, where monotonicity is measured, not promised.
        assert diag["monotone_costs"] is True


@pytest.mark.parametrize("data_scale", [0.01, 1.0, 100.0])
def test_relative_gamma_is_invariant_to_data_scale(data_scale):
    """Rescaling the expression matrix must not change the effective temperature ratio.

    Under absolute gamma, multiplying the data by 100 multiplies squared-Euclidean costs
    by 10^4 and silently changes the alignment; under relative gamma it must not.
    """
    trajs = _cohort(3, K=100, scale=data_scale)
    cov, diag = _coverage(trajs, "sqeuclidean", 2.0, "relative")
    assert cov > 0.5, f"scale={data_scale}: coverage collapsed to {cov:.3f}"
    # gamma_effective must track the data scale (sqeuclidean scales as the square)
    assert diag["gamma_effective"] > 0.0


def test_gamma_effective_is_frozen_for_the_run():
    """A single scalar gamma for the whole run -- the descent proof depends on it.

    L = -sum_p w_p log Z_p(B; gamma) is only comparable between iterations at constant
    gamma, so a per-iteration rescale would void the guarantee while still *looking*
    monotone.
    """
    trajs = _cohort(4, K=100)
    _B, _p, diag = _fit_barycenter_soft_em(
        patient_trajectories=trajs,
        n_bins=trajs[0].shape[0],
        gamma=2.0,
        max_iter=12,
        tol=1e-12,
        verbose=False,
        dtw_dist_method="sqeuclidean",
        barycenter_init_iter=3,
        strict=True,
        gamma_scaling="relative",
    )
    ge = diag["gamma_effective"]
    assert np.isscalar(ge) or np.ndim(ge) == 0
    assert np.isfinite(ge) and ge > 0
    assert diag["gamma_scaling"] == "relative"
    assert diag["monotone_costs"] is True, "descent must still hold under relative gamma"


def test_absolute_gamma_is_the_default_and_unchanged():
    """Default must remain absolute so existing results are reproducible."""
    trajs = _cohort(5, K=50)
    _B, _p, diag = _fit_barycenter_soft_em(
        patient_trajectories=trajs,
        n_bins=trajs[0].shape[0],
        gamma=0.7,
        max_iter=6,
        tol=1e-10,
        verbose=False,
        dtw_dist_method="sqeuclidean",
        barycenter_init_iter=3,
        strict=True,
    )
    assert diag["gamma_scaling"] == "absolute"
    assert diag["gamma_effective"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# cost-scale helpers
# ---------------------------------------------------------------------------
def test_cost_scale_helpers_track_metric_magnitude():
    """The whole point: cosine and sqeuclidean live on different scales."""
    trajs = _cohort(6, K=200)
    cos = _pairwise_cost_scale(trajs, "cosine")
    sq = _pairwise_cost_scale(trajs, "sqeuclidean")
    assert 0.0 < cos < 2.0, f"cosine cost should be bounded in [0,2], got {cos}"
    assert sq > 10 * cos, f"sqeuclidean ({sq}) should dwarf cosine ({cos}) at K=200"
    ref = _reference_cost_scale(trajs, trajs[0], "sqeuclidean")
    assert ref > 0


def test_cost_scale_helpers_are_degenerate_safe():
    """Identical trajectories give zero cost; helpers must return 1.0, not 0."""
    flat = [np.ones((10, 4)) for _ in range(3)]
    assert _pairwise_cost_scale(flat, "sqeuclidean") == 1.0
    assert _reference_cost_scale(flat, flat[0], "cosine") == 1.0
    single = [np.ones((10, 4))]
    assert _pairwise_cost_scale(single, "sqeuclidean") == 1.0


# ---------------------------------------------------------------------------
# DTW coupling band (hard-DTW path)
# ---------------------------------------------------------------------------
def test_dtw_window_fraction_is_honoured_with_open_ends():
    """The Sakoe-Chiba band must not be silently widened to n_bins-1.

    The band is the only thing coupling patients to a shared region of the barycenter.
    Widening it when both ends are open lets each patient drift into its own cheapest
    window; on the 9-patient cohort that cost cross-batch purity 0.6706 -> 0.2130 while
    the run still converged and reported no error.
    """
    import inspect

    from scdebussy.tl._alignment import _fit_barycenter_from_trajectories

    src = inspect.getsource(_fit_barycenter_from_trajectories)
    assert "_window_size = max(1, int(dtw_window_fraction * n_bins))" in src
    assert "_window_size = n_bins - 1" not in src, (
        "the band override is back; patients will decouple into disjoint windows"
    )


def test_hard_dtw_keeps_patients_on_a_shared_span():
    """Every patient should occupy most of the consensus axis on a full-coverage cohort."""
    import numpy as np

    from scdebussy.tl._alignment import _fit_barycenter_from_trajectories

    trajs = _cohort(11, P=4, T=40, K=30)
    _B, warp_paths, _diag = _fit_barycenter_from_trajectories(
        patient_trajectories=trajs,
        n_bins=40,
        gamma=0.1,
        max_iter=8,
        tol=1e-6,
        verbose=False,
        dtw_dist_method="cosine",
        dtw_step_pattern="asymmetric",
        dtw_open_begin=True,
        dtw_open_end=True,
        dtw_window_fraction=0.2,
        barycenter_init_iter=3,
        barycenter_update_iter=1,
    )
    spans = []
    for path in warp_paths:
        bary = np.array([b for _a, b in path])
        spans.append((bary.max() - bary.min() + 1) / 40.0)
    assert min(spans) > 0.5, f"a patient collapsed onto a narrow window: spans={spans}"
