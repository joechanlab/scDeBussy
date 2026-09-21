"""fit()-level gates for the opt-in joint posterior-warp mode (Phase 3).

Complements tests/test_v2_joint_warp.py (which gates the graph/inference layer
in isolation, not fit()). See validation/v21/experiment_round/JOINT_WARP_CONTRACT.md
for the design these tests hold fit() to.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scdebussy import align
from scdebussy.align._dp import JointPosterior, Posterior, check_joint_invariants


def straight_dataset(seed=3, patients=6, cells=60):
    rng = np.random.default_rng(seed)
    time = np.tile(np.linspace(0, 1, cells), patients)
    patient = np.repeat([f"p{i}" for i in range(patients)], cells)
    x = np.column_stack([time, np.sin(2 * time), (time - 0.5) ** 2]) + rng.normal(0, 0.05, (len(time), 3))
    return ad.AnnData(
        x,
        obs=pd.DataFrame({"patient": patient, "time": time}, index=[f"c{i}" for i in range(len(time))]),
        var=pd.DataFrame(index=["a", "b", "c"]),
    )


def warped_dataset(seed=5, patients=4, cells=50, warp_strength=0.22):
    """Like ``straight_dataset``, but one patient's expression follows a locally
    stretched/compressed reparametrization of pseudotime, so a 1:1-only model
    cannot align it as well as a model with (1,2)/(2,1) warp edges."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, cells)

    def curve(s):
        return np.column_stack([s, np.sin(2 * s), (s - 0.5) ** 2])

    xs, times, pats = [], [], []
    for p in range(patients):
        if p == 0:
            # Monotone local stretch/compression: same underlying biology, but
            # visited at a warped rate relative to observed pseudotime "grid".
            s = grid + warp_strength * np.sin(2 * np.pi * grid) / (2 * np.pi)
            s = np.clip(s, 0.0, 1.0)
        else:
            s = grid
        xs.append(curve(s) + rng.normal(0, 0.04, (cells, 3)))
        times.append(grid)
        pats.append(np.full(cells, f"p{p}"))
    x = np.concatenate(xs)
    time = np.concatenate(times)
    patient = np.concatenate(pats)
    return ad.AnnData(
        x,
        obs=pd.DataFrame({"patient": patient, "time": time}, index=[f"c{i}" for i in range(len(time))]),
        var=pd.DataFrame(index=["a", "b", "c"]),
    )


def base_options(**kwargs):
    options = {
        "patient_key": "patient",
        "pseudotime_key": "time",
        "n_landmarks": 8,
        "locality_radius": 0.3,
        "beta": 0.5,
        "eta_i": 2.0,
        "eta_d": 2.0,
        "epsilon": 0.01,
        "max_iter": 25,
    }
    options.update(kwargs)
    return options


# ---------------------------------------------------------------------------
# (a) warp="off" reproduces exact current fit() numerics
# ---------------------------------------------------------------------------


def test_warp_off_matches_default_exactly():
    data = straight_dataset()
    implicit = align.fit(data, **base_options())
    explicit = align.fit(data, **base_options(warp="off"))
    np.testing.assert_array_equal(implicit.barycenter, explicit.barycenter)
    for p_i, p_e in zip(implicit.posteriors, explicit.posteriors, strict=True):
        assert isinstance(p_i, Posterior) and isinstance(p_e, Posterior)
        np.testing.assert_array_equal(p_i.log_a, p_e.log_a)
        assert p_i.log_z == p_e.log_z
    assert implicit.history == explicit.history
    assert implicit.converged == explicit.converged
    assert set(implicit.diagnostics["p0"]) == {
        "no_match_probability",
        "low_match_probability",
        "unaligned",
        "path_entropy",
        "numerical_zero_matches",
    }


def test_warp_on_rejects_global_mode_and_nonpositive_terminal_gap():
    data = straight_dataset()
    with pytest.raises(ValueError, match="mode='open'"):
        align.fit(data, **base_options(mode="global", warp="on", lambda_terminal_gap=1.0))
    with pytest.raises(ValueError, match="lambda_terminal_gap > 0"):
        align.fit(data, **base_options(warp="on", lambda_terminal_gap=0.0))


# ---------------------------------------------------------------------------
# (b) warp="on" uses warp edges and improves fit on a stretch/compression case
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def warp_fits():
    data = warped_dataset()
    off = align.fit(data, **base_options(n_landmarks=10, eta_i=1.5, eta_d=1.5, max_iter=40))
    on = align.fit(
        data,
        **base_options(
            n_landmarks=10,
            eta_i=1.5,
            eta_d=1.5,
            max_iter=40,
            warp="on",
            omega_warp=0.05,
            lambda_terminal_gap=1.5,
        ),
    )
    # Warp-disabled-in-price control, holding the joint graph's own terminal-cost
    # structure fixed: same joint model, but omega_warp is prohibitively large so
    # (1,2)/(2,1) edges are never worth taking. Isolates the warp contribution
    # from the unrelated fact that "off" and "on" mode also differ in whether
    # terminal edges pay lambda_terminal_gap at all (JOINT_WARP_CONTRACT.md Q9.1).
    on_no_warp = align.fit(
        data,
        **base_options(
            n_landmarks=10,
            eta_i=1.5,
            eta_d=1.5,
            max_iter=40,
            warp="on",
            omega_warp=1e6,
            lambda_terminal_gap=1.5,
        ),
    )
    return data, off, on, on_no_warp


def test_joint_mode_uses_warp_edges_and_improves_fit(warp_fits):
    _, off, on, on_no_warp = warp_fits
    for p in on.posteriors:
        assert isinstance(p, JointPosterior)
    warp_mass = sum(p.expected_match_12 + p.expected_match_21 for p in on.posteriors)
    assert warp_mass > 0, "joint fit never used a (1,2)/(2,1) warp edge on a warped synthetic case"
    no_warp_mass = sum(p.expected_match_12 + p.expected_match_21 for p in on_no_warp.posteriors)
    assert no_warp_mass == pytest.approx(0.0, abs=1e-6), "prohibitive omega_warp control still used warp edges"
    # Holding the joint graph structure fixed, actually being able to use warp
    # edges must not make the fit worse (it is a strict superset of paths).
    assert on.history[-1]["free_energy"] <= on_no_warp.history[-1]["free_energy"] + 1e-6


def test_joint_mode_diagnostics_and_history_fields(warp_fits):
    _, _, on, _ = warp_fits
    for patient in on.patients:
        d = on.diagnostics[patient]
        for key in (
            "expected_match_11",
            "expected_match_12",
            "expected_match_21",
            "patient_unmatched_internal_mass",
            "patient_unmatched_terminal_mass",
            "consensus_unmatched_internal_mass",
            "consensus_unmatched_terminal_mass",
            "endpoint_occupancy",
            "consensus_share",
            "degenerate_all_gap_risk",
            "mean_displacement",
            "max_displacement",
            "p95_displacement",
        ):
            assert key in d
        assert "endpoint_pileup" not in d
        posterior = on.posteriors[on.patients.index(patient)]
        assert d["endpoint_occupancy"] == pytest.approx(
            posterior.consensus_occupancy[0] + posterior.consensus_occupancy[-1]
        )
        assert "no_match_probability" not in d  # old-mode-only, deliberately not extended
    for row in on.history[:-1]:
        for key in ("barycenter_change", "e11_mass", "warp_mass", "internal_unmatched_mass", "terminal_unmatched_mass"):
            assert key in row
            assert np.isfinite(row[key])


# ---------------------------------------------------------------------------
# (c) EM descent / invariant checks hold across iterations
# ---------------------------------------------------------------------------


def test_joint_mode_descent_and_invariants_hold(warp_fits):
    _, _, on, _ = warp_fits
    for row in on.history:
        assert row["free_energy_conditional"] <= row["free_energy_before"] + 1e-6
        assert row["free_energy"] <= row["free_energy_conditional"] + 1e-6
    for lm, p in zip(on.landmarks, on.posteriors, strict=True):
        check_joint_invariants(p, len(lm.coordinates), len(on.barycenter))  # must not raise


def test_joint_mode_project_and_summarize_use_occupancy(warp_fits):
    data, _, on, _ = warp_fits
    projection = align.project(on, data)
    assert np.isfinite(projection.position).any()
    quantities = align.summarize(on, projection)
    assert quantities.matched_occupancy.shape == (len(on.patients), len(on.barycenter))
    assert np.all(np.isfinite(quantities.matched_occupancy) | np.isnan(quantities.matched_occupancy))


# ---------------------------------------------------------------------------
# (d) deterministic
# ---------------------------------------------------------------------------


def test_joint_mode_is_deterministic():
    data = warped_dataset()
    opts = base_options(
        n_landmarks=10,
        eta_i=1.5,
        eta_d=1.5,
        max_iter=15,
        warp="on",
        omega_warp=0.05,
        lambda_terminal_gap=1.5,
    )
    first = align.fit(data, **opts)
    second = align.fit(data, **opts)
    np.testing.assert_array_equal(first.barycenter, second.barycenter)
    for p1, p2 in zip(first.posteriors, second.posteriors, strict=True):
        np.testing.assert_array_equal(p1.log_a, p2.log_a)
    assert first.history == second.history
