"""Integration, numerical stability, projection, and patient-level inference."""

import os
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from scdebussy import align
from scdebussy.align._barycenter import costs, sufficient_statistics
from scdebussy.align._landmarks import estimate


def dataset(seed=3, patients=6, cells=60):
    rng = np.random.default_rng(seed)
    time = np.tile(np.linspace(0, 1, cells), patients)
    patient = np.repeat([f"p{i}" for i in range(patients)], cells)
    x = np.column_stack([time, np.sin(2 * time), (time - 0.5) ** 2]) + rng.normal(0, 0.05, (len(time), 3))
    return ad.AnnData(
        x,
        obs=pd.DataFrame({"patient": patient, "time": time}, index=[f"c{i}" for i in range(len(time))]),
        var=pd.DataFrame(index=["a", "b", "c"]),
    )


def fit(data, **kwargs):
    options = {
        "patient_key": "patient",
        "pseudotime_key": "time",
        "n_landmarks": 6,
        "locality_radius": 0.3,
        "beta": 0.5,
        "eta_i": 2.0,
        "eta_d": 2.0,
        "epsilon": 0.01,
        "max_iter": 30,
    }
    options.update(kwargs)
    return align.fit(data, **options)


@pytest.fixture(scope="module")
def fitted():
    return fit(dataset())


def test_fit_no_mutation_descent_and_final_consistency(fitted):
    data = dataset()
    before = data.copy()
    result = fit(data)
    np.testing.assert_array_equal(data.X, before.X)
    pd.testing.assert_frame_equal(data.obs, before.obs)
    assert not data.uns and not data.obsm
    for row in result.history:
        assert row["free_energy_conditional"] <= row["free_energy_before"] + 1e-8
        assert row["free_energy"] <= row["free_energy_conditional"] + 1e-8
    assert result.history[-1]["free_energy"] == pytest.approx(-sum(p.log_z for p in result.posteriors))
    for lm, p in zip(result.landmarks, result.posteriors, strict=True):
        check = align.infer(align.LatticeModel(costs(lm, result.barycenter), 0.5, 2.0, 2.0, 6.0, 6.0, lm.matchable))
        np.testing.assert_allclose(p.log_a, check.log_a)
    np.testing.assert_array_equal(result.barycenter, fitted.barycenter)


def test_m_step_is_numerical_minimum(fitted):
    result = fitted
    stats = sufficient_statistics(result.landmarks, result.posteriors, result.barycenter)

    def objective(flat):
        b = flat.reshape(result.barycenter.shape)
        return sum(np.sum(p.a * costs(lm, b)) for lm, p in zip(result.landmarks, result.posteriors, strict=True))

    numerical = minimize(objective, result.barycenter.ravel(), method="BFGS", options={"gtol": 1e-7})
    assert objective(stats.mean.ravel()) <= numerical.fun + 1e-7
    np.testing.assert_allclose(stats.s1 / stats.s0, stats.mean)


def test_log_m_step_does_not_treat_underflow_as_structural_zero(fitted):
    from dataclasses import replace

    posteriors = tuple(replace(p, log_a=p.log_a - 1000.0) for p in fitted.posteriors)
    stats = sufficient_statistics(fitted.landmarks, posteriors, fitted.barycenter + 50.0)
    normal = sufficient_statistics(fitted.landmarks, fitted.posteriors, fitted.barycenter)
    assert np.all(stats.s0 == 0.0)
    assert np.isfinite(stats.log_s0).all()
    np.testing.assert_allclose(stats.mean, normal.mean)
    unavailable = tuple(replace(p, log_a=np.full_like(p.log_a, -np.inf)) for p in fitted.posteriors)
    held = sufficient_statistics(fitted.landmarks, unavailable, fitted.barycenter)
    np.testing.assert_array_equal(held.mean, fitted.barycenter)


def test_projection_mass_occupancy_and_write(fitted, tmp_path):
    p = align.project(fitted)
    np.testing.assert_allclose(p.probability.sum(1) + p.unprojected, 1.0, atol=1e-10)
    np.testing.assert_allclose(p.position, p.conditional_probability @ (np.arange(6) / 6))
    assert (p.entropy >= 0).all()
    summary = align.summarize(fitted)
    np.testing.assert_allclose(summary.cell_occupancy.sum(1), 1.0)
    np.testing.assert_allclose(summary.matched_occupancy.sum(1), 1.0)
    assert ((summary.coverage >= 0) & (summary.coverage <= 1)).all()
    data = dataset()
    data.obs["aligned_pseudotime"] = 0.123
    fitted.write(data)
    assert (data.obs.aligned_pseudotime == 0.123).all()
    path = tmp_path / "result.h5ad"
    data.write_h5ad(path)
    reread = ad.read_h5ad(path)
    np.testing.assert_allclose(reread.uns["scdebussy_v2"]["barycenter"], fitted.barycenter)


def test_projection_validation(fitted):
    data = dataset()
    with pytest.raises(ValueError, match="gene"):
        align.project(fitted, data[:, ::-1])
    data.obs["patient"] = "unknown"
    with pytest.raises(ValueError, match="unknown"):
        align.project(fitted, data)


def test_remote_cells_do_not_establish_coverage():
    data = dataset(patients=1, cells=100)
    data.obs["time"] = 0.95
    landmarks, _, _ = estimate(data, patient_key="patient", pseudotime_key="time", n_landmarks=10, locality_radius=0.1)
    assert landmarks[0].local_count[0] == 0
    assert landmarks[0].n_eff[0] == 0
    assert not landmarks[0].matchable[0]
    result = fit(data, n_landmarks=10, locality_radius=0.1)
    query = data[:1].copy()
    query.obs["time"] = 0.5
    projection = align.project(result, query)
    assert projection.unprojected[0] == 1.0
    assert np.isnan(projection.position[0])


def test_no_local_landmark_and_nonconvergence():
    data = dataset(patients=1)
    data.obs["time"] = 1.0
    with pytest.raises(align.InsufficientSupportError):
        fit(data, locality_radius=0.001)
    result = fit(dataset(), max_iter=1)
    assert not result.converged
    assert np.isfinite(result.posteriors[0].count_probability).all()


def test_calibration_failure_and_success():
    data = dataset()
    result = fit(data, eta_i=None, eta_d=None)
    assert result.config["eta_i"] > 0
    assert result.config["calibration"] == "band_minimum_quantile"
    from scdebussy.align._barycenter import initialize

    initial = initialize(result.landmarks)
    candidates = []
    for lm in result.landmarks:
        delta = costs(lm, initial)
        for i in np.flatnonzero(lm.matchable):
            allowed = np.abs(lm.coordinates - lm.coordinates[i]) <= 0.25
            candidates.append(delta[i, allowed].min())
    assert result.config["eta_i"] == pytest.approx(np.quantile(candidates, 0.9, method="linear"))
    data.X[:] = 0.0
    with pytest.raises(align.DegenerateCalibrationError):
        fit(data, eta_i=None, eta_d=None)


@pytest.mark.parametrize("scheme", ["exchangeable", "blocked", "paired"])
def test_comparison_is_deterministic_and_handles_designs(fitted, scheme):
    labels = dict(zip(fitted.patients, ["a", "b"] * 3, strict=True))
    metadata = dict(zip(fitted.patients, [0, 0, 1, 1, 2, 2], strict=True))
    kw = {"blocks": metadata} if scheme == "blocked" else {"pairs": metadata} if scheme == "paired" else {}
    first = align.compare(fitted, labels, scheme=scheme, seed=12, n_perm=39, **kw)
    second = align.compare(fitted, labels, scheme=scheme, seed=12, n_perm=39, **kw)
    pd.testing.assert_frame_equal(first.expression, second.expression)
    for table in (first.kinetic, first.occupancy, first.expression):
        assert (table.status == "tested").all()
        assert table.pvalue.between(1 / 40, 1).all()
        assert (table.qvalue >= table.pvalue - 1e-12).all()


def test_invalid_pairing_and_insufficient_groups(fitted):
    with pytest.raises(ValueError, match="pair"):
        align.compare(fitted, [0, 0, 0, 1, 1, 1], scheme="paired", seed=0, pairs=[0, 0, 1, 1, 2, 2], n_perm=9)
    result = align.compare(fitted, [0, 0, 0, 0, 0, 1], scheme="exchangeable", seed=0, n_perm=9)
    assert result.kinetic.status.iloc[0] == "not_testable"
    assert result.expression.pvalue.isna().all()


def test_no_torch_on_v2_import():
    command = "import sys; import scdebussy.align; assert 'torch' not in sys.modules; from scdebussy import align; assert callable(align.fit)"
    subprocess.run([sys.executable, "-c", command], env=os.environ.copy(), check=True)


def test_sparse_and_chunked_landmarks_match_dense():
    from scipy.sparse import csr_matrix

    dense = dataset(patients=2)
    sparse = dense.copy()
    sparse.X = csr_matrix(sparse.X)
    options = {"patient_key": "patient", "pseudotime_key": "time", "n_landmarks": 8, "locality_radius": 0.25}
    one, _, _ = estimate(dense, chunk_size=10000, **options)
    chunks, _, _ = estimate(sparse, chunk_size=7, **options)
    for left, right in zip(one, chunks, strict=True):
        for name in ("expression", "variance", "slope", "n_eff", "local_count", "matchable"):
            np.testing.assert_allclose(getattr(left, name), getattr(right, name), atol=1e-10)


def test_permutation_analytic_null(monkeypatch):
    from types import SimpleNamespace

    from scdebussy.align._compare import compare
    from scdebussy.align._project import PatientQuantities

    summary = PatientQuantities(
        np.zeros((4, 2)),
        np.ones((4, 2)),
        np.arange(4.0),
        np.full((4, 2), 0.5),
        np.full((4, 2), 0.5),
        np.ones(4),
        np.zeros((4, 2, 1)),
        np.ones((4, 2)),
    )
    monkeypatch.setattr("scdebussy.align._compare.summarize", lambda result: summary)
    result = SimpleNamespace(patients=("a", "b", "c", "d"), genes=("g",))
    unpaired = compare(result, [0, 0, 1, 1], scheme="exchangeable", seed=4, n_perm=4999)
    paired = compare(result, [0, 0, 1, 1], scheme="paired", pairs=[0, 1, 0, 1], seed=4, n_perm=4999)
    # Two of six unrestricted assignments attain |mean difference| = 2;
    # two of four within-pair assignments attain it.
    assert unpaired.kinetic.pvalue.iloc[0] == pytest.approx(1 / 3, abs=0.025)
    assert paired.kinetic.pvalue.iloc[0] == pytest.approx(0.5, abs=0.025)
    assert unpaired.expression.pvalue.iloc[0] == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"beta": 0},
        {"locality_radius": 0},
        {"n_landmarks": 1},
        {"mode": "invalid"},
        {"backend": "invalid"},
        {"eta_d": None},
    ],
)
def test_invalid_fit_parameters(kwargs):
    with pytest.raises(ValueError):
        fit(dataset(), **kwargs)


def test_smoother_linear_truth_and_singular_fallback():
    data = dataset(patients=1, cells=100)
    data.X[:, 0] = 2 + 3 * data.obs.time.to_numpy()
    options = {"patient_key": "patient", "pseudotime_key": "time", "n_landmarks": 10, "locality_radius": 0.2}
    (lm,), _, _ = estimate(data, **options)
    np.testing.assert_allclose(lm.expression[:, 0], 2 + 3 * lm.coordinates, atol=1e-10)
    np.testing.assert_allclose(lm.slope[:, 0], 3.0, atol=1e-10)
    np.testing.assert_allclose(lm.variance[:, 0], 0.0, atol=1e-10)
    data.obs["time"] = 0.5
    data.X[:, 0] = np.arange(100.0)
    (singular,), _, _ = estimate(data, **options)
    assert singular.slope[5, 0] == 0.0
    assert singular.expression[5, 0] == pytest.approx(49.5)
    assert singular.variance[5, 0] == pytest.approx(np.var(np.arange(100.0)) / 100)


def test_projection_retains_unmatched_and_underflow_probability(fitted):
    from dataclasses import replace

    original = align.project(fitted)
    reduced = replace(fitted, posteriors=tuple(replace(p, log_a=p.log_a + np.log(0.05)) for p in fitted.posteriors))
    projected = align.project(reduced)
    np.testing.assert_allclose(projected.probability, 0.05 * original.probability)
    np.testing.assert_allclose(projected.position, original.position)
    assert (projected.unprojected >= 0.95 - 1e-10).all()
    tiny = replace(fitted, posteriors=tuple(replace(p, log_a=p.log_a - 1000.0) for p in fitted.posteriors))
    projected = align.project(tiny)
    assert (projected.probability == 0.0).all()
    assert np.isfinite(projected.log_retained).all()
    np.testing.assert_allclose(projected.position, original.position)


def test_unaligned_patient_remains_undefined():
    data = dataset()
    data.obs.loc[data.obs.patient == "p0", "time"] = 1.0
    with pytest.warns(RuntimeWarning, match="p0"):
        result = fit(data, locality_radius=0.1)
    assert result.diagnostics["p0"]["unaligned"]
    assert result.posteriors[0].no_match_probability == pytest.approx(1.0)
    summary = align.summarize(result)
    assert np.isnan(summary.kinetic_mean[0])
    assert np.isnan(summary.cell_occupancy[0]).all()
    assert np.isnan(summary.residual[0]).all()
