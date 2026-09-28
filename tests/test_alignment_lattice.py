"""Independent fixtures followed by oracle / NumPy / Numba equivalence."""

import numpy as np
import pytest

from scdebussy.alignment.lattice import LatticeModel, enumerate_paths
from scdebussy.alignment.posterior import infer


def model(delta, mode="open", mask=None, beta=1.0, scale=1.0):
    return LatticeModel(np.asarray(delta) * scale, beta, scale, 2 * scale, 3 * scale, 4 * scale, mask, mode)


def test_hand_one_by_one():
    m = model([[2.0]])
    oracle = enumerate_paths(m)
    np.testing.assert_array_equal(np.sort(oracle.energies), [2.0, 3.0])
    result = infer(m)
    assert result.a[0, 0] / result.no_match_probability == pytest.approx(np.exp(1.0))
    global_oracle = enumerate_paths(model([[2.0]], "global"))
    # M, ID, DI; each gap ordering pays both opening charges.
    np.testing.assert_array_equal(np.sort(global_oracle.energies), [2.0, 10.0, 10.0])


@pytest.mark.parametrize("mode", ["open", "global"])
@pytest.mark.parametrize("seed", range(100))
def test_generated_equivalence(mode, seed):
    rng = np.random.default_rng(seed)
    n, t = rng.integers(2, 6, size=2)
    m = model(rng.uniform(0, 9, (n, t)), mode, rng.random(n) > 0.3, beta=0.7)
    oracle = enumerate_paths(m)
    for backend in ("numpy", "numba"):
        r = infer(m, backend=backend)
        assert r.log_z == pytest.approx(oracle.log_z, abs=1e-10)
        np.testing.assert_allclose(r.a, np.exp(oracle.log_a), atol=1e-10, rtol=1e-8)
        np.testing.assert_allclose(r.count_probability, oracle.count_probability, atol=1e-10, rtol=1e-8)
        np.testing.assert_allclose(r.edge_probability, oracle.edge_probability, atol=1e-10, rtol=1e-8)
        assert r.entropy == pytest.approx(oracle.entropy, abs=1e-10)
        assert r.map_energy == pytest.approx(oracle.map_energy)
        np.testing.assert_allclose(r.a.sum(1) + r.insertion, 1.0, atol=1e-10)
        np.testing.assert_allclose(r.a.sum(0) + r.deletion, 1.0, atol=1e-10)
        assert r.expected_matches + r.expected_insertions == pytest.approx(n)
        assert r.expected_matches + r.expected_deletions == pytest.approx(t)
        assert r.count_probability.sum() == pytest.approx(1.0)


def test_terminal_offsets_and_all_gap():
    m = model(np.zeros((4, 6)))
    g = m.graph
    # Last M ends at (2,3): 2 patient and 3 consensus landmarks remain.
    source = 1 + 3 * (2 * 7 + 3)
    e = np.flatnonzero((g.source == source) & (g.target == g.n_nodes - 1)).item()
    assert g.energy[e] == 2 * m.eta_i + 3 * m.eta_d
    assert g.opens_i[e] == g.opens_d[e] == 0
    r = infer(model(np.zeros((4, 6)), mask=np.zeros(4, bool)))
    assert r.no_match_probability == pytest.approx(1.0)
    assert r.expected_matches == 0
    assert r.expected_open_i == r.expected_open_d == 0
    assert r.entropy == pytest.approx(0.0)


@pytest.mark.parametrize("mask", [[False, True, True], [True, False, True], [True, True, False], [False, False, True]])
def test_unmatchable(mask):
    for mode in ("global", "open"):
        r = infer(model(np.zeros((3, 3)), mode, mask))
        assert np.isneginf(r.log_a[~np.array(mask)]).all()
        np.testing.assert_array_equal(r.insertion[~np.array(mask)], 1.0)


def test_limits_and_scale():
    delta = np.array([[0.2, 8.0], [9.0, 0.3]])
    a = infer(model(delta))
    b = infer(model(delta, scale=5.0, beta=0.2))
    np.testing.assert_allclose(a.a, b.a)
    hot = infer(model(delta, beta=1e-12))
    assert hot.entropy == pytest.approx(np.log(len(enumerate_paths(model(delta)).paths)), abs=1e-10)
    cold = infer(model(delta, beta=1e4))
    np.testing.assert_allclose(cold.a, np.eye(2))


def test_conditional_means_can_reverse():
    m = LatticeModel(np.array([[100.0, 0.0], [0.0, 100.0]]), 1.0, 1.0, 1.0, 3.0, 3.0)
    a = infer(m).a
    means = (a @ np.array([0.0, 0.5])) / a.sum(1)
    assert means[0] > means[1]


def test_log_marginals_survive_underflow_and_model_is_immutable():
    m = model([[10000.0]])
    r = infer(m)
    assert r.a[0, 0] == 0.0
    assert np.isfinite(r.log_a[0, 0])
    with pytest.raises(ValueError):
        m.delta.setflags(write=True)
    with pytest.raises(ValueError):
        m.graph.energy[0] = 9.0
