"""Structural checks for the isolated DTW prototype."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation/v21/experiment_round"))
from warp_prototype import align_cost


def test_diagonal_and_zero_warp_limit():
    x = np.linspace(0, 1, 8)
    cost = (x[:, None] - x[None, :]) ** 2
    mapped, detail = align_cost(cost, np.ones(8, bool), x, omega=1e6, internal_cost=10, terminal_cost=10)
    np.testing.assert_allclose(mapped, x)
    assert detail["warp_12"] == detail["warp_21"] == 0
    assert detail["internal_patient_gap"] == detail["internal_consensus_gap"] == 0


def test_stretch_uses_warp_without_internal_gap():
    local = np.array([0, .25, .5, .75, 1.])
    consensus = np.linspace(0, 1, 7)
    cost = (local[:, None] - consensus[None, :]) ** 2
    mapped, detail = align_cost(cost, np.ones(5, bool), consensus, omega=.001, internal_cost=10, terminal_cost=10)
    assert detail["warp_12"] > 0
    assert detail["internal_patient_gap"] == detail["internal_consensus_gap"] == 0
    assert np.all(np.diff(mapped) >= 0)
    assert mapped[0] == 0 and mapped[-1] <= 1


def test_missing_internal_row_is_gap_not_speed_change():
    x = np.linspace(0, 1, 7)
    cost = (x[:, None] - x[None, :]) ** 2
    cost[3] += 100
    mapped, detail = align_cost(cost, np.ones(7, bool), x, omega=10, internal_cost=1, terminal_cost=10)
    assert detail["internal_patient_gap"] >= 1
    assert detail["warp_12"] == detail["warp_21"] == 0
    assert np.all(np.diff(mapped) >= 0)


def test_terminal_consensus_gap_is_separate():
    x = np.linspace(0, 1, 6)
    cost = np.abs(x[:, None] - x[None, :])
    cost[:, 4:] += 5
    mapped, detail = align_cost(cost, np.ones(6, bool), x, omega=.01, internal_cost=10, terminal_cost=.1)
    assert detail["terminal_consensus_gap"] > 0
    assert detail["internal_consensus_gap"] == 0
    assert np.all(np.diff(mapped) >= 0)
