"""Opt-in expression-cost weighting leaves the default contract intact."""

import numpy as np
import pandas as pd
import pytest
import anndata as ad

from scdebussy import align


def dataset(cells=30):
    time = np.tile(np.linspace(0, 1, cells), 3)
    patient = np.repeat(["p0", "p1", "p2"], cells)
    x = np.column_stack([time, np.sin(2 * time), (time - .5) ** 2])
    return ad.AnnData(x, obs=pd.DataFrame({"patient": patient, "time": time}), var=pd.DataFrame(index=["a", "b", "c"]))


def fit(data, **kwargs):
    return align.fit(data, patient_key="patient", pseudotime_key="time", n_landmarks=6,
                     locality_radius=.3, beta=.5, eta_i=2., eta_d=2., epsilon=.01,
                     max_iter=30, **kwargs)


def test_precision_modes_are_finite_and_explicit():
    data = dataset(cells=30)
    original = fit(data)
    explicit = fit(data, precision_mode="estimated")
    np.testing.assert_allclose(original.barycenter, explicit.barycenter)
    unit = fit(data, precision_mode="unit")
    assert all(np.all(lm.precision == 1) for lm in unit.landmarks)
    capped = fit(data, precision_mode="capped", contribution_cap_quantile=0.5)
    assert all(np.all(lm.precision <= base.precision) for lm, base in zip(capped.landmarks, original.landmarks))
    assert all(np.isfinite(lm.precision).all() for lm in capped.landmarks)
    assert capped.config["precision_mode"] == "capped"


@pytest.mark.parametrize("mode", ["bad", "", None])
def test_precision_mode_rejects_unknown_values(mode):
    with pytest.raises(ValueError, match="precision_mode"):
        fit(dataset(cells=20), precision_mode=mode)
