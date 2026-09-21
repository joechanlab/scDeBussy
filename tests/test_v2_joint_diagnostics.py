"""Units and normalization of joint-warp diagnostic quantities."""

from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys

import numpy as np
import pytest

from scdebussy.align._dp import JointPosterior, Posterior
from scdebussy.align._project import summarize


def _summary(log_a, expression, barycenter, *, joint, occupancy=None):
    log_a = np.asarray(log_a, float)
    n, t = log_a.shape
    if joint:
        posterior = JointPosterior(
            log_z=0.0, log_a=log_a,
            patient_occupancy=np.exp(log_a).sum(axis=1),
            consensus_occupancy=np.asarray(occupancy, float),
            patient_unmatched_internal=np.zeros(n), patient_unmatched_terminal=np.zeros(n),
            consensus_unmatched_internal=np.zeros(t), consensus_unmatched_terminal=np.zeros(t),
            expected_energy=0.0, entropy=0.0, expected_match_11=0.0,
            expected_match_12=0.0, expected_match_21=0.0,
            expected_open_i=0.0, expected_open_d=0.0,
            map_edges=np.empty(0), map_energy=0.0, edge_probability=np.empty(0),
        )
    else:
        posterior = Posterior(
            log_z=0.0, log_a=log_a, insertion=np.zeros(n), deletion=np.zeros(t),
            expected_energy=0.0, entropy=0.0, expected_insertions=0.0,
            expected_deletions=0.0, expected_open_i=0.0, expected_open_d=0.0,
            count_probability=np.empty(0), map_edges=np.empty(0),
            map_energy=0.0, edge_probability=np.empty(0),
        )
    lm = SimpleNamespace(patient="p", coordinates=np.linspace(0, 1, n),
                         expression=np.asarray(expression, float), support_intervals=None)
    result = SimpleNamespace(patients=("p",), landmarks=(lm,), posteriors=(posterior,),
                             barycenter=np.asarray(barycenter, float),
                             consensus_coordinates=np.linspace(0, 1, t))
    projection = SimpleNamespace(patients=np.array([], dtype=str))
    return summarize(result, projection)


def test_two_to_one_residual_is_weighted_mean_not_sum():
    q = _summary([[0.0], [0.0]], [[2.0], [4.0]], [[1.0]], joint=True, occupancy=[1.0])
    assert q.residual[0, 0, 0] == pytest.approx(2.0)  # (2 + 4) / 2 - 1
    assert q.residual_support[0, 0] == pytest.approx(1.0)  # occupancy, not A column sum 2


def test_one_to_two_residual_uses_correspondence_mass():
    q = _summary([[np.log(.5), np.log(.5)]], [[3.0]], [[1.0], [2.0]],
                 joint=True, occupancy=[1.0, 1.0])
    np.testing.assert_allclose(q.residual[0, :, 0], [2.0, 1.0])
    np.testing.assert_allclose(q.residual_support[0], [1.0, 1.0])


def test_zero_and_near_zero_correspondence_mass_leave_residual_undefined():
    q = _summary([[0.0, -np.inf, np.log(1e-14)]], [[3.0]],
                 [[0.0], [0.0], [0.0]], joint=True, occupancy=[1.0, 0.0, 1e-14])
    assert q.residual[0, 0, 0] == pytest.approx(3.0)
    assert np.isnan(q.residual[0, 1:, 0]).all()


def test_old_mode_residual_and_support_keep_column_sum_semantics():
    q = _summary([[np.log(.25)], [np.log(.75)]], [[2.0], [6.0]], [[1.0]], joint=False)
    assert q.residual[0, 0, 0] == pytest.approx(4.0)
    assert q.residual_support[0, 0] == pytest.approx(1.0)


def _aggregation_module():
    root = Path(__file__).resolve().parents[1] / "validation/v21/experiment_round/eight_patient/joint_warp"
    sys.path.insert(0, str(root.parents[1]))  # experiment_round, for summarize_and_plot
    spec = importlib.util.spec_from_file_location("aggregate_joint_warp", root / "aggregate_joint_warp.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_warp_fractions_have_consistent_units_and_partition():
    fractions = _aggregation_module().posterior_fractions(1, 2, 3, 4, 5)
    assert fractions["warp_edge_fraction_among_match_edges"] == pytest.approx(5 / 6)
    assert fractions["warp_patient_state_fraction"] == pytest.approx(8 / 18)
    assert sum(v for k, v in fractions.items() if k.endswith("patient_state_fraction")) == pytest.approx(1)
    assert "warp_frac" not in fractions
    all_gap = _aggregation_module().posterior_fractions(0, 0, 0, 4, 5)
    assert np.isnan(all_gap["warp_edge_fraction_among_match_edges"])
    assert sum(v for k, v in all_gap.items() if k.endswith("patient_state_fraction")) == pytest.approx(1)


def test_standard_phase4_reporting_includes_paired_unit_baseline(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1] / "validation/v21/experiment_round/eight_patient/joint_warp"
    spec = importlib.util.spec_from_file_location("regenerate_reporting", root / "regenerate_reporting.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda args, check: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["report", "--adata", "input.h5ad", "--cache-root", "cache",
                                          "--output-dir", str(tmp_path)])
    module.main()
    assert len(calls) == 3
    assert "--paired-unit-cache" in calls[0]
    assert any(str(value).endswith("trajectory_250_unit.npz") for value in calls[0])
    assert any(str(value).endswith("trajectory_250_unit.npz") for value in calls[1])
