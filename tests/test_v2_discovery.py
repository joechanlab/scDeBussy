"""Expression events, density robustness, and rectangular public API contracts."""
import anndata as ad
import numpy as np
import pandas as pd

from scdebussy import align
from scdebussy.align._landmarks import estimate


def events(shifts=(0.35, 0.65), density=False, flat=False, hole=False):
    frames, values = [], []
    for p, shift in enumerate(shifts):
        time = np.linspace(0, 1, 801)
        if density:
            time = np.r_[time, np.linspace(0.35, 0.5, 1000)]
        if hole:
            time = time[(time <= 0.4) | (time >= 0.6)]
        x = np.tanh((time - shift) / 0.035)
        values.append(np.column_stack([x, 2*x]) if not flat else np.zeros((len(time), 2)))
        frames.append(pd.DataFrame({'patient': f'p{p}', 'time': time}))
    obs = pd.concat(frames, ignore_index=True)
    obs.index = obs.index.astype(str)
    return ad.AnnData(np.vstack(values), obs=obs)


def discover(data, **kw):
    return estimate(data, patient_key='patient', pseudotime_key='time', n_landmarks=60,
                    locality_radius=0.1, landmark_method='expression', **kw)[0]


def test_shifted_expression_events_have_patient_specific_anchors():
    a, b = discover(events())
    for lm, event in zip((a, b), (0.35, 0.65), strict=True):
        peak = np.argmax(np.linalg.norm(lm.slope, axis=1))
        assert abs(lm.coordinates[peak] - event) < 0.015
    assert not np.array_equal(a.coordinates, b.coordinates)
    assert len(a.coordinates) <= 60 and len(b.coordinates) <= 60


def test_density_alone_does_not_shift_event():
    (a,) = discover(events((0.5,)))
    (b,) = discover(events((0.5,), density=True))
    peaks = [lm.coordinates[np.argmax(np.linalg.norm(lm.slope, axis=1))] for lm in (a, b)]
    np.testing.assert_allclose(peaks, [0.5, 0.5], atol=1/241)


def test_holes_flat_and_noise_are_deterministic():
    for flat in (False, True):
        data = events((0.3,), flat=flat, hole=True)
        if not flat:
            data.X += np.random.default_rng(13).normal(0, 0.03, data.shape)
        (a,), (b,) = discover(data), discover(data)
        np.testing.assert_array_equal(a.coordinates, b.coordinates)
        assert not ((a.coordinates > 0.4) & (a.coordinates < 0.6)).any()
        assert a.coordinates[0] == 0 and a.coordinates[-1] == 1


def test_partial_coverage():
    data = events((0.5,))
    data = data[(data.obs.time >= 0.2) & (data.obs.time <= 0.8)].copy()
    (lm,) = discover(data)
    assert lm.coordinates[0] >= 0.2 and lm.coordinates[-1] <= 0.8


def test_ragged_fit_project_compare_and_write(tmp_path):
    data = events((0.3, 0.6, 0.4, 0.5))
    data = data[~((data.obs.patient == 'p0') & (data.obs.time > 0.5))].copy()
    result = align.fit(data, patient_key='patient', pseudotime_key='time', n_landmarks=60,
                       consensus_resolution=21, landmark_method='expression', locality_radius=0.1,
                       beta=1, eta_i=2, eta_d=2, epsilon=0.01, max_iter=5)
    lengths = [len(lm.coordinates) for lm in result.landmarks]
    assert len(set(lengths)) > 1
    assert result.barycenter.shape == (21, 2)
    assert all(p.log_a.shape == (n, 21) for p, n in zip(result.posteriors, lengths, strict=True))
    summary = align.summarize(result)
    assert [len(x) for x in summary.kinetic_curve] == lengths
    assert [len(x) for x in summary.matchability] == lengths
    projection = align.project(result)
    np.testing.assert_allclose(projection.probability.sum(1) + projection.unprojected, 1)
    align.compare(result, [0, 0, 1, 1], scheme='exchangeable', seed=2, n_perm=5)
    result.write(data)
    data.write_h5ad(tmp_path / 'ragged.h5ad')
    reread = ad.read_h5ad(tmp_path / 'ragged.h5ad')
    assert [len(v['coordinates']) for v in reread.uns['scdebussy_v2']['landmarks'].values()] == lengths
    for step in result.history:
        assert step['free_energy'] <= step['free_energy_before'] + 1e-7
