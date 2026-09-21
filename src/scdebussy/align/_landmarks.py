"""Locality-aware, chunked local-linear landmark estimation."""

from dataclasses import replace

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
    adata, *, patient_key, pseudotime_key, n_landmarks, locality_radius, n_min=5, epsilon=1e-6, chunk_size=1024,
    landmark_method="fixed", discovery_resolution=None
):
    """Estimate patient landmarks using O(chunk_size*(T+G) + T*G) workspace."""
    patients, time = validate_adata(adata, patient_key, pseudotime_key)
    budget = n_landmarks
    resolution = discovery_resolution or max(201, 4 * budget + 1)
    fixed_grid = np.linspace(0.0, 1.0, budget)
    results = []
    for patient in pd.unique(patients):
        indices = np.flatnonzero(patients == patient)
        observed = np.unique(time[indices])
        blocks = np.split(observed, np.flatnonzero(np.diff(observed) > locality_radius) + 1)
        intervals = np.asarray([(block[0], block[-1]) for block in blocks])
        grid = fixed_grid if landmark_method == "fixed" else np.unique(
            np.r_[np.linspace(0.0, 1.0, resolution), intervals.ravel()]
        )
        n_landmarks = len(grid)
        density_weights = np.ones(len(time))
        if landmark_method == "expression":
            # Fixed coordinate bins, never cell quantiles. Each occupied pilot bin
            # contributes equal total weight, reducing sampling-density event shifts.
            bins = np.minimum((time[indices] * (resolution - 1)).astype(int), resolution - 2)
            density_weights[indices] = 1 / np.bincount(bins, minlength=resolution - 1)[bins]
        s0, s1, s2, sq, count = (np.zeros(n_landmarks) for _ in range(5))
        y0, y1, y2 = (np.zeros((n_landmarks, adata.n_vars)) for _ in range(3))
        for start in range(0, len(indices), chunk_size):
            ids = indices[start : start + chunk_size]
            values = adata.X[ids]
            values = np.asarray(values.toarray() if hasattr(values, "toarray") else values, dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("expression must be finite; normalize/impute explicitly before fitting")
            distances = time[ids, None] - grid[None, :]
            weights = kernel(time[ids], grid, locality_radius) * density_weights[ids, None]
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
                intervals,
            )
        )
    if landmark_method == "expression":
        results = [discover(lm, budget, locality_radius) for lm in results]
    return tuple(results), patients, time


def discover(lm, budget, radius):
    """Freeze expression-change peaks plus support knots from a uniform pilot.

    Score is mean squared local slope after scaling each gene by its supported
    pilot expression range. Genes whose range is at most twice their median
    approximate mean standard error are excluded. Peaks must exceed 10% of the
    patient's maximum score. Greedy suppression separates events by radius/2.
    Component boundaries are mandatory; remaining slots select events and then cover intervals
    wider than radius/2. The budget is an upper bound.
    """
    grid = lm.coordinates
    eligible = np.zeros(len(grid), bool)
    for lo, hi in lm.support_intervals:
        eligible |= (grid >= lo) & (grid <= hi)
    eligible &= lm.matchable
    indices = np.flatnonzero(eligible)
    if not len(indices):
        # Keep one explicitly unmatchable row so the lattice can represent this patient.
        indices = np.array([int(np.argmin(np.abs(grid - lm.support_intervals[0, 0])))])
        return replace(lm, **{name: getattr(lm, name)[indices].copy() for name in (
            "coordinates", "expression", "variance", "precision", "slope", "n_eff", "local_count", "matchable"
        )})
    # Unmatchable pilot rows also break support; do not select or interpolate across them.
    components = []
    for lo, hi in lm.support_intervals:
        part = indices[(grid[indices] >= lo) & (grid[indices] <= hi)]
        if len(part):
            components.extend(np.split(part, np.flatnonzero(np.diff(part) > 1) + 1))
    selected = set(int(i) for part in components for i in (part[0], part[-1]))
    if len(selected) > budget:
        raise ValueError("landmark budget cannot retain all supported component boundaries")
    span = np.ptp(lm.expression[indices], axis=0)
    noise = 2 * np.median(np.sqrt(lm.variance[indices]), axis=0)
    genes = span > np.maximum(noise, 1e-12)
    score = np.zeros(len(grid))
    if genes.any():
        score = np.mean((lm.slope[:, genes] / span[genes]) ** 2, axis=1)
    peaks = []
    for part in components:
        for a, b, c in zip(part[:-2], part[1:-1], part[2:], strict=True):
            if score[b] > score[a] and score[b] >= score[c] and score[b] >= 0.1 * score[indices].max():
                peaks.append(int(b))
    events = []
    for index in sorted(peaks, key=lambda i: (-score[i], grid[i])):
        if len(selected) >= budget:
            break
        if all(abs(grid[index] - grid[other]) >= radius / 2 for other in events):
            selected.add(index)
            events.append(index)
    # Fill long supported intervals with deterministic support knots.
    while len(selected) < budget:
        candidate, distance = None, radius / 4
        for part in components:
            chosen = np.array(sorted(selected.intersection(part.tolist())))
            distances = np.min(np.abs(grid[part, None] - grid[chosen][None, :]), axis=1)
            k = int(np.argmax(distances))
            if distances[k] > distance + 1e-14:
                candidate, distance = int(part[k]), distances[k]
        if candidate is None:
            break
        selected.add(candidate)
    chosen = np.array(sorted(selected))
    intervals = np.array([(grid[part[0]], grid[part[-1]]) for part in components])
    return replace(lm, support_intervals=intervals, **{
        name: getattr(lm, name)[chosen].copy() for name in (
            "coordinates", "expression", "variance", "precision", "slope", "n_eff", "local_count", "matchable"
        )
    })
