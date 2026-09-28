import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scdebussy.alignment import TerminalCosts, fit, project, summarize
from scdebussy.alignment.local_warp import LocalWarpLattice
from scdebussy.alignment.local_warp_posterior import infer_local_warp


def local(delta, mask=None, **kw):
    return LocalWarpLattice(
        np.asarray(delta, float),
        2.0,
        2.0,
        2.0,
        3.0,
        3.0,
        mask,
        "open",
        warp="on",
        omega_warp=kw.get("warp_penalty", 0.1),
        terminal_costs=kw.get("terminal_costs", TerminalCosts.symmetric(1.0)),
    )


def check(model, result):
    np.testing.assert_allclose(result.a.sum(1) + result.patient_unmatched, 1, atol=1e-9)
    np.testing.assert_allclose(result.consensus_occupancy + result.consensus_unmatched, 1, atol=1e-9)
    assert result.patient_occupancy.sum() == pytest.approx(
        result.expected_match_11 + result.expected_match_12 + 2 * result.expected_match_21
    )
    assert result.consensus_occupancy.sum() == pytest.approx(
        result.expected_match_11 + 2 * result.expected_match_12 + result.expected_match_21
    )


def test_oracle_and_conservation():
    from scdebussy.alignment.local_warp import enumerate_local_warp_paths

    m = local([[0, 1, 4], [2, 0, 1]])
    oracle = enumerate_local_warp_paths(m)
    result = infer_local_warp(m, backend="numpy")
    assert result.log_z == pytest.approx(oracle.log_z)
    np.testing.assert_allclose(result.a, np.exp(oracle.log_a))
    check(m, result)


def test_warp_and_matchability():
    result = infer_local_warp(local([[0, 0, 9], [9, 9, 0]], warp_penalty=0.001), backend="numpy")
    assert result.expected_match_12 + result.expected_match_21 > 0
    result = infer_local_warp(local(np.zeros((2, 3)), mask=[True, False]), backend="numpy")
    assert np.isneginf(result.log_a[1]).all()


def test_asymmetric_terminal_prices():
    cheap_end = infer_local_warp(
        local(np.full((2, 3), 10.0), terminal_costs=TerminalCosts(4, 0.01, 4, 0.01)), backend="numpy"
    )
    expensive_end = infer_local_warp(
        local(np.full((2, 3), 10.0), terminal_costs=TerminalCosts(4, 8, 4, 8)), backend="numpy"
    )
    assert cheap_end.patient_unmatched_terminal.sum() >= expensive_end.patient_unmatched_terminal.sum()


def data():
    t = np.tile(np.linspace(0, 1, 30), 4)
    p = np.repeat(["a", "b", "c", "d"], 30)
    x = np.c_[t, np.sin(3 * t)]
    return ad.AnnData(x, obs=pd.DataFrame({"p": p, "t": t}))


def test_fit_baseline_and_local_warp():
    a = data()
    kw = {
        "patient_key": "p",
        "pseudotime_key": "t",
        "n_landmarks": 6,
        "locality_radius": 0.3,
        "beta": 1.0,
        "eta_i": 2.0,
        "eta_d": 2.0,
        "max_iter": 5,
    }
    base = fit(a, **kw)
    explicit = fit(a, alignment_model="affine_gap", **kw)
    np.testing.assert_allclose(base.barycenter, explicit.barycenter)
    warp = fit(a, alignment_model="local_warp", warp_penalty=0.1, terminal_costs=TerminalCosts.symmetric(1), **kw)
    for row in warp.history:
        assert row["free_energy"] <= row["free_energy_before"] + 1e-8
    q = project(warp)
    s = summarize(warp, q)
    assert np.isfinite(q.position).any()
    assert np.isfinite(s.coverage).all()


def test_directional_stretch_and_compression():
    stretch = infer_local_warp(local([[0, 0]], warp_penalty=0.001), backend="numpy")
    assert stretch.expected_match_12 > 0.8
    assert stretch.expected_match_21 == pytest.approx(0)
    compression = infer_local_warp(local([[0], [0]], warp_penalty=0.001), backend="numpy")
    assert compression.expected_match_21 > 0.8
    assert compression.expected_match_12 == pytest.approx(0)
