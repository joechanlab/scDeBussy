"""The simulator's causal interventions and deliberate nonidentifiability."""

import numpy as np
import pytest

from scdebussy.alignment import SimulationConfig, nonidentifiable_pair, simulate


def test_delay_changes_time_not_expression_or_density():
    base = simulate(seed=4)
    delayed = simulate(seed=4, warp=0.5)
    np.testing.assert_array_equal(base.X, delayed.X)
    np.testing.assert_array_equal(base.obs.z_true, delayed.obs.z_true)
    assert not np.array_equal(base.obs.s_local, delayed.obs.s_local)


def test_expression_changes_expression_only():
    base = simulate(seed=4)
    changed = simulate(seed=4, expression=1.0)
    np.testing.assert_array_equal(base.obs.s_local, changed.obs.s_local)
    np.testing.assert_array_equal(base.obs.z_true, changed.obs.z_true)
    affected = changed.var.expression_affected.to_numpy()
    np.testing.assert_array_equal(base.X[:, ~affected], changed.X[:, ~affected])
    assert np.max(np.abs(base.X - changed.X)) > 0.9


def test_gap_removes_interior_observations():
    data = simulate(gap_depth=1.0, cells_per_patient=500)
    z = data.obs.loc[data.obs.condition == 1, "z_true"]
    assert not z.between(0.43, 0.57).any()


@pytest.mark.parametrize(
    "panel", ["all", "random", "high_variance", "low_variance", "duplicated", "nuisance", "informative"]
)
@pytest.mark.parametrize("noise", ["gaussian", "heavy_tailed", "correlated", "heteroskedastic"])
def test_robustness_panels(panel, noise):
    cfg = SimulationConfig(gene_panel=panel, noise_model=noise, cells_per_patient=20)
    a = simulate(cfg)
    assert np.isfinite(a.X).all()
    assert a.var_names.is_unique
    np.testing.assert_array_equal(a.X, simulate(cfg).X)


def test_reversal_and_nonidentifiability():
    a = simulate(reversal=True)
    assert (np.diff(a.obs.loc[a.obs.patient == "p1", "s_local"]) < 0).any()
    a, b = nonidentifiable_pair()
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.obs.s_local, b.obs.s_local)
    assert not np.array_equal(a.obs.z_true, b.obs.z_true)
