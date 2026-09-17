"""Locality-aware, chunked local-linear landmark estimation."""

import numpy as np
import pandas as pd

from ._types import Landmarks


def validate_adata(adata, patient_key, pseudotime_key):
    """Validate identifiers and pseudotime without normalizing inputs."""
    if patient_key not in adata.obs or pseudotime_key not in adata.obs:
        raise ValueError("patient_key and pseudotime_key must be present in adata.obs")
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("cell and gene identifiers must be unique")
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("at least one cell and one gene are required")
    if adata.obs[patient_key].isna().any():
        raise ValueError("patient identifiers cannot be missing")
    raw = adata.obs[patient_key]
    patients = raw.astype(str).to_numpy()
    if len(pd.unique(raw)) != len(pd.unique(patients)):
        raise ValueError("patient identifiers collide after conversion to strings")
    time = np.asarray(adata.obs[pseudotime_key], dtype=float)
    if not np.isfinite(time).all() or ((time < 0) | (time > 1)).any():
        raise ValueError("pseudotime must be finite and in [0, 1]")
    return patients, time


def kernel(time, coordinates, radius):
    """Raw compact Gaussian weights; no adaptive bridging across holes."""
    distance = np.asarray(time)[:, None] - np.asarray(coordinates)[None, :]
    return np.exp(-0.5 * (distance / (radius / 2)) ** 2) * (np.abs(distance) <= radius)


def estimate(
    adata, *, patient_key, pseudotime_key, n_landmarks, locality_radius, n_min=5, epsilon=1e-6, chunk_size=1024
):
    """Estimate patient landmarks using O(chunk_size*(T+G) + T*G) workspace."""
    patients, time = validate_adata(adata, patient_key, pseudotime_key)
    grid = np.arange(n_landmarks, dtype=float) / n_landmarks
    results = []
    for patient in pd.unique(patients):
        indices = np.flatnonzero(patients == patient)
        s0, s1, s2, sq, count = (np.zeros(n_landmarks) for _ in range(5))
        y0, y1, y2 = (np.zeros((n_landmarks, adata.n_vars)) for _ in range(3))
        for start in range(0, len(indices), chunk_size):
            ids = indices[start : start + chunk_size]
            values = adata.X[ids]
            values = np.asarray(values.toarray() if hasattr(values, "toarray") else values, dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("expression must be finite; normalize/impute explicitly before fitting")
            distances = time[ids, None] - grid[None, :]
            weights = kernel(time[ids], grid, locality_radius)
            s0 += weights.sum(0)
            s1 += (weights * distances).sum(0)
            s2 += (weights * distances**2).sum(0)
            sq += (weights**2).sum(0)
            count += (weights > 0).sum(0)
            y0 += weights.T @ values
            y1 += (weights * distances).T @ values
            y2 += weights.T @ (values**2)
        neff = np.divide(s0**2, sq, out=np.zeros_like(s0), where=sq > 0)
        mean = np.divide(y0, s0[:, None], out=np.zeros_like(y0), where=s0[:, None] > 0)
        slope = np.zeros_like(mean)
        determinant = s0 * s2 - s1**2
        regular = determinant > np.finfo(float).eps * np.maximum(s0 * s2, 1.0)
        mean[regular] = (s2[regular, None] * y0[regular] - s1[regular, None] * y1[regular]) / determinant[regular, None]
        slope[regular] = (s0[regular, None] * y1[regular] - s1[regular, None] * y0[regular]) / determinant[
            regular, None
        ]
        residual_sum = (
            y2
            - 2 * mean * y0
            - 2 * slope * y1
            + mean**2 * s0[:, None]
            + 2 * mean * slope * s1[:, None]
            + slope**2 * s2[:, None]
        )
        dispersion = np.divide(
            np.maximum(residual_sum, 0.0), s0[:, None], out=np.zeros_like(mean), where=s0[:, None] > 0
        )
        variance = np.divide(dispersion, neff[:, None], out=np.zeros_like(mean), where=neff[:, None] > 0)
        mask = (count >= n_min) & (neff >= n_min)
        results.append(
            Landmarks(
                str(patient),
                grid.copy(),
                mean,
                variance,
                1 / (variance + epsilon),
                slope,
                neff,
                count.astype(int),
                mask,
            )
        )
    return tuple(results), patients, time
