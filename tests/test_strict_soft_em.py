"""Decisive mathematical tests for the strict soft-EM descent guarantee.

These four tests target the claim in ``docs/mathematical_specification.md`` §2.2.b that
``L(B) = -sum_p w_p log Z_p(B)`` is non-increasing across EM iterations:

a. ``test_forward_backward_matches_brute_force_enumeration`` -- the E-step's log Z and
   posterior occupancy must equal an exhaustive enumeration of admissible lattice paths.
b. ``test_analytic_m_step_equals_numerical_minimum`` -- the closed-form M-step must equal
   a numerical minimiser of the expected-cost objective (dagger).
c. ``test_strict_objective_non_increasing_every_iteration`` -- descent must hold at every
   iteration across many seeds, not merely end-to-end.
d. ``test_posterior_mean_mapping_bounded_and_monotone`` -- posterior-mean warp mappings
   must be bounded and non-decreasing for every patient.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import logsumexp

from scdebussy.tl._alignment import (
    _compute_distance_matrix,
    _fit_barycenter_soft_em,
    _forward_backward_open,
)

# The lattice implemented by _forward_backward_open: every step advances t by 1,
# and the barycenter index advances by 0, 1, 2 or 3.
_STEPS = (0, 1, 2, 3)


def _enumerate_paths(T: int, window: int, advance_mask=None):
    """All admissible open-begin/open-end paths as tuples of barycenter indices.

    A path is a length-T sequence tp[0..T-1] with tp[0] free (open begin), tp[t+1] -
    tp[t] in {0,1,2,3}, all indices within [0, T-1], and |t - tp[t]| <= window.
    When advance_mask[t] is False only the 0-step into t is allowed, mirroring the
    forward recursion's gating on advance_mask[t].
    """
    paths = []

    def rec(t, prefix):
        if t == T:
            paths.append(tuple(prefix))
            return
        allowed = _STEPS if (advance_mask is None or advance_mask[t]) else (0,)
        for s in allowed:
            nxt = prefix[-1] + s
            if nxt > T - 1:
                continue
            if abs(t - nxt) > window:
                continue
            prefix.append(nxt)
            rec(t + 1, prefix)
            prefix.pop()

    for start in range(T):
        if abs(0 - start) > window:
            continue
        rec(1, [start])
    return paths


def _brute_force(delta, gamma, window, advance_mask=None):
    """Exact log Z and occupancy A[t, t'] by explicit enumeration."""
    T = delta.shape[0]
    paths = _enumerate_paths(T, window, advance_mask)
    assert paths, "no admissible paths -- test lattice is degenerate"
    log_w = np.array([-sum(delta[t, tp[t]] for t in range(T)) / gamma for tp in paths])
    log_Z = logsumexp(log_w)
    post = np.exp(log_w - log_Z)
    A = np.zeros((T, T))
    for prob, tp in zip(post, paths, strict=False):
        for t in range(T):
            A[t, tp[t]] += prob
    return log_Z, A


# ---------------------------------------------------------------------------
# (a) E-step exactness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("T", [3, 4, 5])
@pytest.mark.parametrize("gamma", [0.1, 1.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_forward_backward_matches_brute_force_enumeration(T, gamma, seed):
    """log Z and posterior occupancy match exhaustive path enumeration."""
    rng = np.random.default_rng(seed)
    delta = rng.uniform(0.1, 2.0, size=(T, T))
    window = T - 1

    A_fb, log_Z_fb = _forward_backward_open(delta, gamma, window)
    log_Z_bf, A_bf = _brute_force(delta, gamma, window)

    assert log_Z_fb == pytest.approx(log_Z_bf, rel=1e-10, abs=1e-10)
    np.testing.assert_allclose(A_fb, A_bf, rtol=1e-8, atol=1e-10)
    # Every step advances t by exactly one, so each row is a probability distribution.
    np.testing.assert_allclose(A_fb.sum(axis=1), np.ones(T), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_forward_backward_matches_brute_force_with_advance_mask(seed):
    """Same exactness with LLR advance-blocking active (restricted path set)."""
    rng = np.random.default_rng(seed)
    T = 5
    delta = rng.uniform(0.1, 2.0, size=(T, T))
    advance_mask = np.array([True, True, False, True, True])
    gamma = 0.5
    window = T - 1

    A_fb, log_Z_fb = _forward_backward_open(delta, gamma, window, advance_mask=advance_mask)
    log_Z_bf, A_bf = _brute_force(delta, gamma, window, advance_mask=advance_mask)

    assert log_Z_fb == pytest.approx(log_Z_bf, rel=1e-10, abs=1e-10)
    np.testing.assert_allclose(A_fb, A_bf, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(A_fb.sum(axis=1), np.ones(T), rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# (b) M-step optimality
# ---------------------------------------------------------------------------
def _expected_cost(B_flat, trajs, mats, weights, T, K, row_weights=None):
    """Q(B) = sum_p w_p sum_{t,t'} A_p[t,t'] rho_p[t] ||X_p(t) - B(t')||^2."""
    B = B_flat.reshape(T, K)
    total = 0.0
    for p, (traj, A) in enumerate(zip(trajs, mats, strict=False)):
        W = A if row_weights is None else A * row_weights[p][:, None]
        # ||X(t) - B(t')||^2 for all (t, t')
        D = _compute_distance_matrix(np.asarray(traj), B, "sqeuclidean")
        total += float(weights[p]) * float((W * D).sum())
    return total


@pytest.mark.parametrize("use_row_weights", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_analytic_m_step_equals_numerical_minimum(use_row_weights, seed):
    """The closed-form weighted mean is the numerical argmin of (dagger)."""
    rng = np.random.default_rng(seed)
    T, K, P = 6, 3, 3
    trajs = [rng.normal(size=(T, K)) for _ in range(P)]
    # Valid posterior occupancies: non-negative with unit row sums.
    mats = []
    for _ in range(P):
        A = rng.uniform(0.05, 1.0, size=(T, T))
        mats.append(A / A.sum(axis=1, keepdims=True))
    weights = rng.uniform(0.5, 1.5, size=P)
    weights /= weights.sum()
    row_weights = [rng.uniform(1.0, 5.0, size=T) for _ in range(P)] if use_row_weights else None

    # Analytic minimiser
    num = np.zeros((T, K))
    den = np.zeros(T)
    for p, (traj, A) in enumerate(zip(trajs, mats, strict=False)):
        W = A if row_weights is None else A * row_weights[p][:, None]
        num += weights[p] * (W.T @ np.asarray(traj))
        den += weights[p] * W.sum(axis=0)
    assert np.all(den > 0), "test fixture must cover every barycenter bin"
    B_analytic = num / den[:, None]

    # Numerical minimiser from a deliberately poor start
    x0 = rng.normal(size=T * K) * 3.0
    res = minimize(
        _expected_cost,
        x0,
        args=(trajs, mats, weights, T, K, row_weights),
        method="L-BFGS-B",
        options={"maxiter": 20000, "ftol": 1e-16, "gtol": 1e-14},
    )
    B_numeric = res.x.reshape(T, K)

    np.testing.assert_allclose(B_analytic, B_numeric, rtol=1e-4, atol=1e-5)
    q_analytic = _expected_cost(B_analytic.ravel(), trajs, mats, weights, T, K, row_weights)
    q_numeric = _expected_cost(B_numeric.ravel(), trajs, mats, weights, T, K, row_weights)
    assert q_analytic <= q_numeric + 1e-8


def test_mean_m_step_is_not_optimal_for_cosine_cost():
    """Guard rail: the mean M-step is genuinely wrong for cosine cost.

    This is why strict mode forces sqeuclidean.  If this test ever fails, the
    justification for that restriction needs revisiting.
    """
    rng = np.random.default_rng(0)
    T, K, P = 5, 3, 3
    trajs = [rng.normal(size=(T, K)) + 2.0 for _ in range(P)]
    mats = []
    for _ in range(P):
        A = rng.uniform(0.05, 1.0, size=(T, T))
        mats.append(A / A.sum(axis=1, keepdims=True))
    weights = np.ones(P) / P

    def q_cos(B_flat):
        B = B_flat.reshape(T, K)
        total = 0.0
        for p, (traj, A) in enumerate(zip(trajs, mats, strict=False)):
            D = _compute_distance_matrix(np.asarray(traj), B, "cosine")
            total += weights[p] * float((A * D).sum())
        return total

    num = np.zeros((T, K))
    den = np.zeros(T)
    for p, (traj, A) in enumerate(zip(trajs, mats, strict=False)):
        num += weights[p] * (A.T @ np.asarray(traj))
        den += weights[p] * A.sum(axis=0)
    B_mean = num / den[:, None]

    res = minimize(q_cos, B_mean.ravel(), method="L-BFGS-B", options={"maxiter": 20000})
    # A strictly better B exists, so the mean is not the argmin under cosine cost.
    assert res.fun < q_cos(B_mean.ravel()) - 1e-6


# ---------------------------------------------------------------------------
# (c) Per-iteration descent
# ---------------------------------------------------------------------------
def _make_patients(rng, P=4, T=24, K=5, with_density=False):
    """Warped, noisy observations of a shared smooth latent program."""
    phases = rng.uniform(0.0, 2 * np.pi, size=K)
    freqs = rng.uniform(1.0, 3.0, size=K)
    trajs, densities = [], []
    for _ in range(P):
        # random monotone warp
        w = np.sort(rng.uniform(size=T - 2))
        warped = np.concatenate([[0.0], w, [1.0]])
        traj = np.sin(2 * np.pi * freqs[None, :] * warped[:, None] + phases[None, :])
        traj += rng.normal(scale=0.15, size=traj.shape)
        trajs.append(traj)
        if with_density:
            d = rng.uniform(0.2, 1.0, size=T)
            d[: rng.integers(1, 4)] = 0.01  # a few LLR-only bins
            densities.append(d)
    return trajs, (densities if with_density else None)


@pytest.mark.parametrize("seed", list(range(25)))
@pytest.mark.parametrize("with_density", [False, True])
def test_strict_objective_non_increasing_every_iteration(seed, with_density):
    """L must not increase at ANY iteration, for every seed."""
    rng = np.random.default_rng(seed)
    trajs, densities = _make_patients(rng, with_density=with_density)
    T = trajs[0].shape[0]

    _B, _paths, diag = _fit_barycenter_soft_em(
        patient_trajectories=trajs,
        n_bins=T,
        gamma=1.0,
        max_iter=30,
        tol=1e-12,
        verbose=False,
        dtw_dist_method="sqeuclidean",
        barycenter_init_iter=3,
        patient_densities=densities,
        strict=True,
    )

    costs = diag["iteration_costs"]
    assert len(costs) >= 3, f"too few iterations to be a meaningful test: {costs}"
    diffs = np.diff(costs)
    # Allow only floating-point slack, scaled to the magnitude of the objective.
    slack = 1e-9 * max(1.0, max(abs(c) for c in costs))
    bad = [(i, costs[i], costs[i + 1]) for i, d in enumerate(diffs) if d > slack]
    assert not bad, f"objective increased at iteration(s) {bad}"
    assert diag["monotone_costs"] is True
    assert diag["strict"] is True


@pytest.mark.parametrize("seed", list(range(10)))
def test_legacy_mode_is_not_claimed_monotone(seed):
    """Documents the failure the strict mode exists to fix.

    We do not assert that legacy *always* violates descent -- only that the code no
    longer advertises a guarantee it cannot keep.  ``strict`` is reported as False and
    ``monotone_costs`` is measured rather than assumed.
    """
    rng = np.random.default_rng(100 + seed)
    trajs, densities = _make_patients(rng, with_density=True)
    T = trajs[0].shape[0]
    _B, _paths, diag = _fit_barycenter_soft_em(
        patient_trajectories=trajs,
        n_bins=T,
        gamma=1.0,
        max_iter=30,
        tol=1e-12,
        verbose=False,
        dtw_dist_method="cosine",
        barycenter_init_iter=3,
        patient_densities=densities,
        strict=False,
    )
    assert diag["strict"] is False
    assert isinstance(diag["monotone_costs"], bool)


def test_strict_rejects_non_squared_euclidean_cost():
    rng = np.random.default_rng(0)
    trajs, _ = _make_patients(rng, P=3, T=12, K=3)
    with pytest.raises(ValueError, match="sqeuclidean"):
        _fit_barycenter_soft_em(
            patient_trajectories=trajs,
            n_bins=12,
            gamma=1.0,
            max_iter=3,
            tol=1e-9,
            verbose=False,
            dtw_dist_method="cosine",
            barycenter_init_iter=2,
            strict=True,
        )


def test_strict_returns_barycenter_consistent_with_final_cost():
    """The returned B, its recorded cost and the returned warp paths must agree."""
    rng = np.random.default_rng(7)
    trajs, densities = _make_patients(rng, with_density=True)
    T = trajs[0].shape[0]
    B, paths, diag = _fit_barycenter_soft_em(
        patient_trajectories=trajs,
        n_bins=T,
        gamma=1.0,
        max_iter=6,  # force a max_iter exit so the extra E-step path is exercised
        tol=0.0,
        verbose=False,
        dtw_dist_method="sqeuclidean",
        barycenter_init_iter=3,
        patient_densities=densities,
        strict=True,
    )
    # Recompute L at the returned barycenter; it must equal the recorded final cost.
    total = 0.0
    w = 1.0 / len(trajs)
    for p, traj in enumerate(trajs):
        delta = _compute_distance_matrix(np.asarray(traj), B, "sqeuclidean")
        d = np.asarray(densities[p], dtype=float)
        pos = d[d > 1e-6]
        ref = float(np.median(pos))
        rw = np.clip(ref / (d + 1e-8), 1.0, 5.0)
        delta = delta * rw[:, None]
        _A, log_Z = _forward_backward_open(delta, 1.0, T - 1, advance_mask=d >= d.max() * 0.05)
        total -= w * log_Z
    assert diag["final_cost"] == pytest.approx(total, rel=1e-9, abs=1e-9)
    assert len(paths) == len(trajs)


# ---------------------------------------------------------------------------
# (d) Posterior-mean mapping: bounded and monotone
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", list(range(15)))
@pytest.mark.parametrize("with_density", [False, True])
def test_posterior_mean_mapping_bounded_and_monotone(seed, with_density):
    """Every patient's warp path is within range and non-decreasing."""
    rng = np.random.default_rng(200 + seed)
    trajs, densities = _make_patients(rng, with_density=with_density)
    T = trajs[0].shape[0]
    _B, warp_paths, _diag = _fit_barycenter_soft_em(
        patient_trajectories=trajs,
        n_bins=T,
        gamma=1.0,
        max_iter=15,
        tol=1e-10,
        verbose=False,
        dtw_dist_method="sqeuclidean",
        barycenter_init_iter=3,
        patient_densities=densities,
        strict=True,
    )
    assert len(warp_paths) == len(trajs)
    for p, path in enumerate(warp_paths):
        pat_idx = [a for a, _ in path]
        bary_idx = np.array([b for _, b in path])
        assert pat_idx == list(range(T)), f"patient {p}: every bin must appear once"
        assert bary_idx.min() >= 0 and bary_idx.max() <= T - 1, f"patient {p}: out of range"
        diffs = np.diff(bary_idx)
        assert np.all(diffs >= 0), f"patient {p}: non-monotone warp path at {np.where(diffs < 0)[0]}"


@pytest.mark.parametrize("seed", list(range(15)))
def test_posterior_mean_is_monotone_before_rounding(seed):
    """The un-rounded posterior mean over t' is itself non-decreasing in t.

    Each admissible path is non-decreasing in t', A[t,:] is exactly the marginal law
    of t'(t), and a convex combination of non-decreasing functions is non-decreasing.
    """
    rng = np.random.default_rng(300 + seed)
    T = 12
    delta = rng.uniform(0.1, 3.0, size=(T, T))
    A, _log_Z = _forward_backward_open(delta, 0.7, T - 1)
    row = A.sum(axis=1)
    assert np.all(row > 0)
    mean_bary = (A * np.arange(T)[None, :]).sum(axis=1) / row
    diffs = np.diff(mean_bary)
    assert np.all(diffs >= -1e-9), f"posterior mean not monotone: min diff {diffs.min()}"
