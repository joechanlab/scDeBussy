"""Synthetic gates for the opt-in joint posterior-warp graph (warp='on').

Mirrors the fixture/oracle style of ``test_v2_lattice.py``. See
``validation/v21/experiment_round/JOINT_WARP_CONTRACT.md`` for the design this
tests against.
"""

import numpy as np
import pytest
from scipy.special import logsumexp

from scdebussy.align._barycenter import sufficient_statistics
from scdebussy.align._dp import Posterior, infer, infer_joint
from scdebussy.align._model import LatticeModel, enumerate_joint_paths, enumerate_paths
from scdebussy.align._types import Landmarks


def joint_model(
    delta,
    mask=None,
    beta=1.0,
    eta_i=1.0,
    eta_d=1.0,
    lambda_i=2.0,
    lambda_d=2.0,
    omega_warp=0.1,
    lambda_terminal_gap=1.0,
):
    return LatticeModel(
        np.asarray(delta),
        beta,
        eta_i,
        eta_d,
        lambda_i,
        lambda_d,
        mask,
        "open",
        warp="on",
        omega_warp=omega_warp,
        lambda_terminal_gap=lambda_terminal_gap,
    )


def old_model(delta, mode="open", mask=None, beta=1.0, scale=1.0):
    return LatticeModel(np.asarray(delta) * scale, beta, scale, 2 * scale, 3 * scale, 4 * scale, mask, mode)


def assert_conservation(model, r, atol=1e-9):
    n, t = model.delta.shape
    np.testing.assert_allclose(r.a.sum(1) + r.patient_unmatched, 1.0, atol=atol)
    np.testing.assert_allclose(r.consensus_occupancy + r.consensus_unmatched, 1.0, atol=atol)
    assert r.patient_occupancy.sum() + r.patient_unmatched.sum() == pytest.approx(n, abs=atol)
    assert r.consensus_occupancy.sum() + r.consensus_unmatched.sum() == pytest.approx(t, abs=atol)
    assert r.patient_occupancy.sum() == pytest.approx(
        r.expected_match_11 + r.expected_match_12 + 2 * r.expected_match_21, abs=1e-6
    )
    assert r.consensus_occupancy.sum() == pytest.approx(
        r.expected_match_11 + 2 * r.expected_match_12 + r.expected_match_21, abs=1e-6
    )
    assert np.isfinite(r.log_z)
    # Every path leaves the source via exactly one edge, and flow is conserved at
    # every internal node (outgoing mass == incoming mass); this is the discrete
    # analogue of "posterior normalizes" for an edge marginal, since summing ep
    # over ALL edges is not itself meaningful (a path uses several edges).
    g = model.graph
    from_source = g.source == 0
    np.testing.assert_allclose(r.edge_probability[from_source].sum(), 1.0, atol=atol)
    into_sink = g.target == g.n_nodes - 1
    np.testing.assert_allclose(r.edge_probability[into_sink].sum(), 1.0, atol=atol)
    for node in range(1, g.n_nodes - 1):
        outgoing = r.edge_probability[g.source == node].sum()
        incoming = r.edge_probability[g.target == node].sum()
        assert outgoing == pytest.approx(incoming, abs=1e-8)


# ---------------------------------------------------------------------------
# Old-mode preservation
# ---------------------------------------------------------------------------


def test_warp_off_matches_old_graph_exactly():
    """warp='off' must produce the exact same Graph/Posterior as before this change."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        n, t = rng.integers(1, 6, size=2)
        delta = rng.uniform(0, 9, (n, t))
        mask = rng.random(n) > 0.3
        m_default = LatticeModel(delta, 0.7, 1.0, 2.0, 3.0, 4.0, mask, "open")
        m_explicit_off = LatticeModel(delta, 0.7, 1.0, 2.0, 3.0, 4.0, mask, "open", warp="off")
        oracle = enumerate_paths(m_default)
        for m in (m_default, m_explicit_off):
            assert type(m.graph).__name__ == "Graph"
            r = infer(m)
            assert r.log_z == pytest.approx(oracle.log_z, abs=1e-10)
            np.testing.assert_allclose(r.a, np.exp(oracle.log_a), atol=1e-10)
            np.testing.assert_allclose(
                r.insertion,
                np.clip(-np.expm1(np.minimum(0.0, logsumexp(oracle.log_a, axis=1))), 0, 1),
                atol=1e-8,
            )


def test_warp_on_requires_open_mode_and_positive_terminal_cost():
    with pytest.raises(ValueError):
        LatticeModel(np.zeros((2, 2)), 1.0, 1.0, 1.0, 1.0, 1.0, mode="global", warp="on", lambda_terminal_gap=1.0)
    with pytest.raises(ValueError):
        LatticeModel(np.zeros((2, 2)), 1.0, 1.0, 1.0, 1.0, 1.0, mode="open", warp="on", lambda_terminal_gap=0.0)
    with pytest.raises(ValueError):
        LatticeModel(np.zeros((2, 2)), 1.0, 1.0, 1.0, 1.0, 1.0, mode="open", warp="bogus")


# ---------------------------------------------------------------------------
# Oracle vs numpy vs numba equivalence (joint graph)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_joint_generated_equivalence(seed):
    rng = np.random.default_rng(seed)
    n, t = rng.integers(1, 5, size=2)
    delta = rng.uniform(0, 9, (n, t))
    mask = rng.random(n) > 0.2
    m = joint_model(delta, mask, beta=0.8, omega_warp=rng.uniform(0.0, 0.5), lambda_terminal_gap=rng.uniform(0.2, 2.0))
    oracle = enumerate_joint_paths(m)
    for backend in ("numpy", "numba"):
        r = infer_joint(m, backend=backend)
        assert r.log_z == pytest.approx(oracle.log_z, abs=1e-9)
        np.testing.assert_allclose(r.a, np.exp(oracle.log_a), atol=1e-9)
        np.testing.assert_allclose(r.patient_occupancy, oracle.patient_occupancy, atol=1e-9)
        np.testing.assert_allclose(r.consensus_occupancy, oracle.consensus_occupancy, atol=1e-9)
        np.testing.assert_allclose(r.patient_unmatched_internal, oracle.patient_unmatched_internal, atol=1e-9)
        np.testing.assert_allclose(r.patient_unmatched_terminal, oracle.patient_unmatched_terminal, atol=1e-9)
        np.testing.assert_allclose(r.consensus_unmatched_internal, oracle.consensus_unmatched_internal, atol=1e-9)
        np.testing.assert_allclose(r.consensus_unmatched_terminal, oracle.consensus_unmatched_terminal, atol=1e-9)
        assert r.expected_match_11 == pytest.approx(oracle.expected_match_11, abs=1e-9)
        assert r.expected_match_12 == pytest.approx(oracle.expected_match_12, abs=1e-9)
        assert r.expected_match_21 == pytest.approx(oracle.expected_match_21, abs=1e-9)
        np.testing.assert_allclose(r.edge_probability, oracle.edge_probability, atol=1e-9)
        assert r.entropy == pytest.approx(oracle.entropy, abs=1e-9)
        assert r.map_energy == pytest.approx(oracle.map_energy, abs=1e-8)
        assert_conservation(m, r)
        # Deterministic repeated calls.
        r_again = infer_joint(m, backend=backend)
        np.testing.assert_array_equal(r.a, r_again.a)
        np.testing.assert_array_equal(r.patient_occupancy, r_again.patient_occupancy)


@pytest.mark.parametrize("n,t", [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1)])
def test_joint_boundary_shapes(n, t):
    rng = np.random.default_rng(n * 100 + t)
    delta = rng.uniform(0, 5, (n, t))
    m = joint_model(delta, beta=1.0)
    oracle = enumerate_joint_paths(m)
    for backend in ("numpy", "numba"):
        r = infer_joint(m, backend=backend)
        assert r.log_z == pytest.approx(oracle.log_z, abs=1e-9)
        assert_conservation(m, r)


# ---------------------------------------------------------------------------
# Case A: identity (no stretch, warp should not be needed)
# ---------------------------------------------------------------------------


def test_case_a_identity():
    # Two identical tiny trajectories: cheapest path is plain (1,1) matches throughout.
    delta = np.array([[0.0, 5.0, 9.0], [5.0, 0.0, 5.0], [9.0, 5.0, 0.0]])
    m = joint_model(
        delta, beta=5.0, eta_i=3.0, eta_d=3.0, lambda_i=3.0, lambda_d=3.0, omega_warp=1.0, lambda_terminal_gap=3.0
    )
    r = infer_joint(m, backend="numpy")
    assert_conservation(m, r)
    np.testing.assert_allclose(np.diag(r.a), np.array([1.0, 1.0, 1.0]), atol=1e-3)
    assert r.expected_match_12 + r.expected_match_21 < 1e-3
    assert r.patient_unmatched.sum() < 1e-3
    assert r.consensus_unmatched.sum() < 1e-3


# ---------------------------------------------------------------------------
# Case B: pure stretch/compression (both directions)
# ---------------------------------------------------------------------------


def test_case_b_warp12_favored():
    # 2 patient landmarks vs 3 consensus states: patient trajectory runs "slower"
    # than consensus, so the cheapest structure is one (1,2) step.
    delta = np.array(
        [
            [0.0, 0.0, 9.0],
            [9.0, 0.0, 0.0],
        ]
    )
    m = joint_model(
        delta, beta=8.0, eta_i=5.0, eta_d=5.0, lambda_i=5.0, lambda_d=5.0, omega_warp=0.05, lambda_terminal_gap=5.0
    )
    r = infer_joint(m, backend="numpy")
    assert_conservation(m, r)
    # n=2 patient landmarks covering t=3 consensus states forces, by the count
    # identity alone (expected_12 - expected_21 = t - n = 1), exactly one warp12
    # edge plus one plain match -- not an all-warp path.
    assert r.expected_match_12 > 0.9
    assert r.expected_match_11 > 0.9
    assert r.expected_match_21 < 0.05
    # Warp mass dominates gap mass (patient_unmatched/consensus_unmatched near 0).
    assert r.expected_match_12 > r.patient_unmatched.sum() + r.consensus_unmatched.sum() + 0.5
    np.testing.assert_allclose(r.patient_occupancy, 1.0, atol=1e-2)
    np.testing.assert_allclose(r.consensus_occupancy, 1.0, atol=1e-2)
    # MAP correspondence must be monotone in landmark index.
    map_j = [g.corr0_j[e] for g, e in [(m.graph, e) for e in r.map_edges] if g.corr0_i[e] >= 0]
    assert map_j == sorted(map_j)


def test_case_b_warp21_favored():
    # 3 patient landmarks vs 2 consensus states: patient trajectory runs "faster",
    # so the cheapest structure is one (2,1) step.
    delta = np.array(
        [
            [0.0, 9.0],
            [0.0, 0.0],
            [9.0, 0.0],
        ]
    )
    m = joint_model(
        delta, beta=8.0, eta_i=5.0, eta_d=5.0, lambda_i=5.0, lambda_d=5.0, omega_warp=0.05, lambda_terminal_gap=5.0
    )
    r = infer_joint(m, backend="numpy")
    assert_conservation(m, r)
    # n=3 patient landmarks covering t=2 consensus states forces, by the count
    # identity (expected_21 - expected_12 = n - t = 1), exactly one warp21 edge
    # plus one plain match -- not an all-warp path.
    assert r.expected_match_21 > 0.9
    assert r.expected_match_11 > 0.9
    assert r.expected_match_12 < 0.05
    assert r.expected_match_21 > r.patient_unmatched.sum() + r.consensus_unmatched.sum() + 0.5
    np.testing.assert_allclose(r.patient_occupancy, 1.0, atol=1e-2)
    np.testing.assert_allclose(r.consensus_occupancy, 1.0, atol=1e-2)


# ---------------------------------------------------------------------------
# Case C: missing internal segment
# ---------------------------------------------------------------------------


def test_case_c_missing_internal_segment():
    # Consensus state 1 (interior) has no good match in either patient landmark;
    # warp is deliberately made expensive so a short internal gap wins instead.
    delta = np.array(
        [
            [0.0, 9.0, 9.0],
            [9.0, 9.0, 0.0],
        ]
    )
    m = joint_model(
        delta, beta=6.0, eta_i=1.0, eta_d=1.0, lambda_i=1.0, lambda_d=1.0, omega_warp=5.0, lambda_terminal_gap=10.0
    )
    r = infer_joint(m, backend="numpy")
    assert_conservation(m, r)
    # Interior consensus state 1 should be unmatched (internally), not warp-absorbed.
    assert r.consensus_unmatched_internal[1] > 0.8
    assert r.consensus_unmatched_terminal[1] < 1e-3
    assert r.expected_match_12 < 0.05
    assert r.expected_match_21 < 0.05
    # Endpoints should still be well matched.
    assert r.a[0, 0] > 0.8
    assert r.a[1, 2] > 0.8


# ---------------------------------------------------------------------------
# Case D: missing terminal segment
# ---------------------------------------------------------------------------


def test_case_d_missing_terminal_segment_and_cost_sweep():
    # Consensus state 2 (tail) only matches patient landmark 1 poorly; a cheap
    # terminal cost should let the tail go unmatched, while raising the terminal
    # cost should recover matching instead.
    delta = np.array(
        [
            [0.0, 9.0, 9.0],
            [9.0, 0.0, 9.0],
        ]
    )
    cheap = joint_model(
        delta, beta=6.0, eta_i=1.0, eta_d=1.0, lambda_i=1.0, lambda_d=1.0, omega_warp=5.0, lambda_terminal_gap=0.2
    )
    r_cheap = infer_joint(cheap, backend="numpy")
    assert_conservation(cheap, r_cheap)
    assert r_cheap.consensus_unmatched_terminal[2] > 0.7
    assert r_cheap.consensus_unmatched_internal.sum() < 0.1
    # Shared prefix still aligns.
    assert r_cheap.a[0, 0] > 0.8
    assert r_cheap.a[1, 1] > 0.8

    expensive = joint_model(
        delta, beta=6.0, eta_i=1.0, eta_d=1.0, lambda_i=1.0, lambda_d=1.0, omega_warp=5.0, lambda_terminal_gap=30.0
    )
    r_expensive = infer_joint(expensive, backend="numpy")
    assert_conservation(expensive, r_expensive)
    # Raising terminal cost should reduce (not necessarily eliminate) terminal
    # unmatched mass at the tail relative to the cheap-terminal case.
    assert r_expensive.consensus_unmatched_terminal[2] < r_cheap.consensus_unmatched_terminal[2]
    # Sane settings should not force an all-unmatched collapse.
    assert r_expensive.patient_unmatched.sum() < r_expensive.patient_occupancy.sum() + 1.0
    assert r_expensive.expected_match_11 + r_expensive.expected_match_12 + r_expensive.expected_match_21 > 0.5


def test_no_all_terminal_collapse_across_warm_cold_beta():
    delta = np.array([[0.0, 0.0], [0.0, 0.0]])
    for beta in (0.05, 1.0, 20.0):
        m = joint_model(
            delta, beta=beta, eta_i=2.0, eta_d=2.0, lambda_i=2.0, lambda_d=2.0, omega_warp=0.2, lambda_terminal_gap=5.0
        )
        r = infer_joint(m, backend="numpy")
        assert_conservation(m, r)
        # With cheap matches (delta=0) and an expensive terminal-gap penalty, the
        # all-terminal path should never dominate the posterior.
        all_terminal_edge = np.flatnonzero((m.graph.source == 0) & (m.graph.target == m.graph.n_nodes - 1))
        assert r.edge_probability[all_terminal_edge].sum() < 0.5


# ---------------------------------------------------------------------------
# Barycenter M-step consumes A unchanged
# ---------------------------------------------------------------------------


def test_barycenter_consumes_joint_A_unchanged():
    rng = np.random.default_rng(7)
    genes = 4
    barycenter = rng.normal(size=(3, genes))
    expr = rng.normal(size=(2, genes))
    lm = Landmarks(
        patient="p1",
        coordinates=np.array([0.2, 0.7]),
        expression=expr,
        variance=np.ones((2, genes)),
        precision=rng.uniform(0.5, 2.0, (2, genes)),
        slope=np.zeros(2),
        n_eff=np.ones(2),
        local_count=np.ones(2),
        matchable=np.array([True, True]),
    )
    delta = np.array([[0.0, 0.0, 5.0], [5.0, 5.0, 5.0]])
    m = joint_model(
        delta,
        mask=lm.matchable,
        beta=20.0,
        eta_i=5.0,
        eta_d=5.0,
        lambda_i=5.0,
        lambda_d=5.0,
        omega_warp=0.01,
        lambda_terminal_gap=5.0,
    )
    p = infer_joint(m, backend="numpy")
    assert p.expected_match_12 > 0.9  # forced (1,2) dominance, so A has genuine half-weight cells

    old = Posterior(
        p.log_z,
        p.log_a,
        np.zeros(2),
        np.zeros(3),
        p.expected_energy,
        p.entropy,
        0.0,
        0.0,
        0.0,
        0.0,
        np.zeros(3),
        p.map_edges,
        p.map_energy,
        p.edge_probability,
    )
    stats_joint = sufficient_statistics((lm,), (p,), barycenter)
    stats_old = sufficient_statistics((lm,), (old,), barycenter)
    np.testing.assert_allclose(stats_joint.mean, stats_old.mean)
    np.testing.assert_allclose(stats_joint.s0, stats_old.s0)


def test_occupancy_support_helper_dispatches_by_type():
    from scdebussy.align._project import occupancy_support

    delta = np.array([[0.0, 5.0], [5.0, 0.0]])
    old = infer(old_model(delta))
    joint = infer_joint(joint_model(delta, beta=2.0))
    m_old, u_old = occupancy_support(old)
    m_joint, u_joint = occupancy_support(joint)
    np.testing.assert_allclose(m_old, old.a.sum(1))
    np.testing.assert_allclose(u_old, old.insertion)
    np.testing.assert_allclose(m_joint, joint.patient_occupancy)
    np.testing.assert_allclose(u_joint, joint.patient_unmatched)
