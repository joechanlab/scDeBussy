import numpy as np
import pytest

from scdebussy.tl import scDeBussy, smooth_patient_trajectory


def test_scDeBussy_writes_results_to_anndata(sample_adata):
    result = scDeBussy(
        sample_adata,
        patient_key="patient",
        pseudotime_key="s_local",
        n_bins=12,
        bandwidth=0.15,
        max_iter=3,
    )

    assert result is sample_adata
    assert "aligned_pseudotime" in sample_adata.obs
    assert "barycenter" in sample_adata.uns

    barycenter = sample_adata.uns["barycenter"]
    assert barycenter["expression"].shape == (12, sample_adata.n_vars)
    assert len(barycenter["aligned_pseudotime"]) == 12
    assert barycenter["patient_ids"] == ["patient1", "patient2"]
    assert len(barycenter["warp_paths"]) == 2
    assert sample_adata.obs["aligned_pseudotime"].notna().all()


def test_scDeBussy_preserves_patientwise_monotonicity(sample_adata):
    scDeBussy(
        sample_adata,
        patient_key="patient",
        pseudotime_key="s_local",
        n_bins=10,
        bandwidth=0.2,
        max_iter=2,
    )

    for _, patient_obs in sample_adata.obs.groupby("patient"):
        sorted_obs = patient_obs.sort_values("s_local")
        aligned = sorted_obs["aligned_pseudotime"].to_numpy()
        assert np.all(np.diff(aligned) >= -1e-8)


def test_scDeBussy_validates_required_columns(sample_adata):
    broken = sample_adata.copy()
    del broken.obs["patient"]

    with pytest.raises(ValueError, match="patient"):
        scDeBussy(broken, patient_key="patient", pseudotime_key="s_local")


def test_smooth_patient_trajectory_returns_expected_shapes(sample_adata):
    smoothed, density = smooth_patient_trajectory(
        sample_adata,
        patient_id="patient1",
        patient_key="patient",
        pseudotime_key="s_local",
        n_bins=9,
        bandwidth=0.1,
    )

    assert tuple(smoothed.shape) == (9, sample_adata.n_vars)
    assert tuple(density.shape) == (9,)
    assert np.all(density.numpy() > 0)
