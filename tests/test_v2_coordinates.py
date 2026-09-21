"""Coordinate truth tests independent of the fitted alignment algorithm."""
from types import SimpleNamespace

import numpy as np
import pytest

from scdebussy.align._project import project


def identity_result(time, coordinates=None, log_mass=None):
    coordinates = np.linspace(0, 1, 100) if coordinates is None else np.asarray(coordinates)
    t = len(coordinates)
    log_a = np.full((t, t), -np.inf)
    np.fill_diagonal(log_a, 0 if log_mass is None else log_mass)
    time = np.asarray(time)
    return SimpleNamespace(
        barycenter=np.zeros((t, 1)), consensus_coordinates=coordinates,
        landmarks=(SimpleNamespace(patient='p', coordinates=coordinates),),
        posteriors=(SimpleNamespace(log_a=log_a),),
        cell_ids=np.arange(len(time)).astype(str), cell_patients=np.repeat('p', len(time)),
        cell_pseudotime=time, patients=('p',),
        config={'chunk_size': 100, 'locality_radius': 0.1},
    )


def test_identity_endpoints_and_interior():
    time = np.r_[0, 0.003, np.linspace(0.01, 0.99, 197), 0.997, 1]
    p = project(identity_result(time))
    np.testing.assert_allclose(p.position, time, atol=1e-14)
    np.testing.assert_allclose(p.unprojected, 0, atol=1e-14)


def test_partial_domain_is_not_stretched_or_extrapolated():
    grid = np.linspace(0.2, 0.8, 61)
    p = project(identity_result([0, 0.2, 0.345, 0.8, 1], grid))
    np.testing.assert_allclose(p.position[1:4], [0.2, 0.345, 0.8])
    assert np.isnan(p.position[[0, 4]]).all()
    np.testing.assert_array_equal(p.unprojected[[0, 4]], 1)


def test_interpolation_does_not_bridge_wide_knot_gap():
    p = project(identity_result([0.2, 0.5, 0.8], [0.2, 0.8]))
    assert np.isnan(p.position[1])
    assert p.unprojected[1] == 1


def test_unmatched_neighbor_weight_is_not_redistributed():
    grid = np.linspace(0, 1, 101)
    masses = np.zeros(101)
    masses[51] = -np.inf
    p = project(identity_result([0.505], grid, masses))
    assert p.unprojected[0] == pytest.approx(0.5)
    assert p.position[0] == pytest.approx(0.5)


def test_observed_hole_and_partial_support_are_not_bridged():
    from scdebussy.align._project import interpolation_weights

    grid = np.linspace(0, 1, 101)
    intervals = np.array([[0.1, 0.4], [0.6, 0.9]])
    weights = interpolation_weights(np.array([0, 0.2, 0.5, 0.8, 1]), grid, 0.1, intervals)
    np.testing.assert_allclose(weights.sum(1), [0, 1, 0, 1, 0])
    np.testing.assert_allclose((weights @ grid)[[1, 3]], [0.2, 0.8])


def test_nonuniform_affine_reproduction():
    from scdebussy.align._project import interpolation_weights

    grid = np.array([0.2, 0.23, 0.3, 0.38, 0.4])
    time = np.linspace(0.2, 0.4, 501)
    weights = interpolation_weights(time, grid, 0.1)
    assert (weights >= 0).all()
    np.testing.assert_allclose(weights @ (2 + 3 * grid), 2 + 3 * time, atol=1e-14)


def test_wasserstein_uses_stored_nonuniform_axis():
    from scdebussy.align._compare import _statistics
    from scdebussy.align._project import PatientQuantities

    occupancy = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]])
    summary = PatientQuantities(np.zeros((4, 3)), np.ones((4, 3)), np.zeros(4),
                               occupancy, occupancy, np.ones(4), np.zeros((4, 3, 1)),
                               np.ones((4, 3)), np.array([0.2, 0.3, 0.8]))
    assert _statistics(summary, np.array([0, 0, 1, 1]))[1] == pytest.approx(0.6)


def test_historical_kernel_endpoint_reproducer():
    from scdebussy.align._landmarks import kernel

    grid = np.arange(100) / 100
    weights = kernel([0, 1], grid, 0.1)
    np.testing.assert_allclose((weights / weights.sum(1, keepdims=True)) @ grid,
                               [0.03388431, 0.96000447], atol=5e-9)
