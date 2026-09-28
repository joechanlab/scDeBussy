"""Unconditional posterior projection and patient-level descriptive quantities."""

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp, xlogy

from .landmarks import validate_adata
from .local_warp_posterior import LocalWarpPosterior
from .types import ProjectionResult


def interpolation_weights(time, coordinates, radius, support_intervals=None):
    """Nonnegative adjacent-knot basis, separate from expression smoothing.

    No extrapolation. Intervals wider than radius are unsupported. When observed
    support intervals are provided, the cell and both knots must lie in one
    interval (training gaps larger than radius split those intervals). Matchability
    never renormalizes this basis: unmatched neighbors lose their original mass.
    """
    time, coordinates = np.asarray(time), np.asarray(coordinates)
    weights = np.zeros((len(time), len(coordinates)))
    right = np.searchsorted(coordinates, time)
    exact = (right < len(coordinates)) & (coordinates[np.minimum(right, len(coordinates) - 1)] == time)
    weights[np.flatnonzero(exact), right[exact]] = 1.0
    inside = (~exact) & (right > 0) & (right < len(coordinates))
    rows = np.flatnonzero(inside)
    left, hi = right[rows] - 1, right[rows]
    widths = coordinates[hi] - coordinates[left]
    valid = widths <= radius + 1e-14
    rows, left, hi, widths = rows[valid], left[valid], hi[valid], widths[valid]
    fraction = (time[rows] - coordinates[left]) / widths
    weights[rows, left], weights[rows, hi] = 1 - fraction, fraction
    if support_intervals is not None:
        allowed = np.zeros(len(time), bool)
        for lo, high in support_intervals:
            knots_inside = (coordinates >= lo) & (coordinates <= high)
            allowed |= (time >= lo) & (time <= high) & ~np.any((weights > 0) & ~knots_inside, axis=1)
        weights[~allowed] = 0
    return weights


def project(result, adata=None) -> ProjectionResult:
    """Project original or new cells for known patients without re-alignment.

    Conditional positions need not be monotone. Finite log support is retained
    even when ordinary probabilities underflow; missing means structural zero.
    """
    if adata is None:
        ids, patients, time = result.cell_ids.copy(), result.cell_patients.copy(), result.cell_pseudotime.copy()
    else:
        if tuple(map(str, adata.var_names)) != result.genes:
            raise ValueError("gene identifiers and order must match the fitted result")
        patients, time = validate_adata(adata, result.config["patient_key"], result.config["pseudotime_key"])
        if not set(patients).issubset(result.patients):
            raise ValueError("cannot project unknown patients without fitting their alignment")
        ids = np.asarray(adata.obs_names, str)
    t = len(result.barycenter)
    conditional = np.zeros((len(ids), t))
    log_retained = np.full(len(ids), -np.inf)
    reversals = {}
    for lm, posterior in zip(result.landmarks, result.posteriors, strict=True):
        rows = np.flatnonzero(patients == lm.patient)
        log_m = (
            np.where(posterior.patient_occupancy > 0, np.log(posterior.patient_occupancy), -np.inf)
            if isinstance(posterior, LocalWarpPosterior)
            else logsumexp(posterior.log_a, axis=1)
        )
        valid_landmarks = np.isfinite(log_m)
        position_given_landmark = np.zeros_like(posterior.log_a)
        position_given_landmark[valid_landmarks] = np.exp(
            posterior.log_a[valid_landmarks] - log_m[valid_landmarks, None]
        )
        for start in range(0, len(rows), result.config["chunk_size"]):
            ix = rows[start : start + result.config["chunk_size"]]
            normalized = interpolation_weights(
                time[ix],
                lm.coordinates,
                result.config["locality_radius"],
                getattr(lm, "support_intervals", None),
            )
            log_r = np.full_like(normalized, -np.inf)
            np.log(normalized, out=log_r, where=normalized > 0)
            mixed = log_r + log_m
            log_mass = logsumexp(mixed, axis=1)
            supported = np.isfinite(log_mass)
            log_retained[ix] = np.minimum(0.0, log_mass)
            mixing = np.zeros_like(mixed)
            mixing[supported] = np.exp(mixed[supported] - log_mass[supported, None])
            conditional[ix] = mixing @ position_given_landmark
    probability = conditional * np.exp(log_retained[:, None])
    unprojected = np.clip(-np.expm1(log_retained), 0.0, 1.0)
    positions = conditional @ result.consensus_coordinates
    positions[~np.isfinite(log_retained)] = np.nan
    entropy = -xlogy(probability, probability).sum(1) - xlogy(unprojected, unprojected)
    status = np.where(unprojected > 0.5, "unsupported", "supported")
    for patient in result.patients:
        ix = np.flatnonzero((patients == patient) & np.isfinite(positions))
        ix = ix[np.argsort(time[ix], kind="stable")]
        reversals[patient] = int(np.sum(np.diff(positions[ix]) < -1e-10))
    return ProjectionResult(
        ids, patients, time, probability, conditional, log_retained, unprojected, positions, entropy, status, reversals
    )


@dataclass(frozen=True)
class PatientQuantities:
    """Posterior-derived descriptive statistics, in result patient order."""

    kinetic_curve: tuple[np.ndarray, ...]
    matchability: tuple[np.ndarray, ...]
    kinetic_mean: np.ndarray
    matched_occupancy: np.ndarray
    cell_occupancy: np.ndarray
    coverage: np.ndarray
    residual: np.ndarray
    residual_support: np.ndarray
    consensus_coordinates: np.ndarray


def summarize(result, projection=None) -> PatientQuantities:
    """Compute local-time displacements, occupancy mass/shape, and residuals."""
    projection = project(result) if projection is None else projection
    n, t, genes = len(result.patients), *result.barycenter.shape
    kinetic = [np.full(len(lm.coordinates), np.nan) for lm in result.landmarks]
    mass = [np.zeros(len(lm.coordinates)) for lm in result.landmarks]
    matched, cell = np.full((n, t), np.nan), np.full((n, t), np.nan)
    residual = np.full((n, t, genes), np.nan)
    support = np.zeros((n, t))
    average, coverage = np.full(n, np.nan), np.zeros(n)
    v = result.consensus_coordinates
    for p, (lm, post) in enumerate(zip(result.landmarks, result.posteriors, strict=True)):
        log_m = (
            np.where(post.patient_occupancy > 0, np.log(post.patient_occupancy), -np.inf)
            if isinstance(post, LocalWarpPosterior)
            else logsumexp(post.log_a, axis=1)
        )
        log_n = (
            np.where(post.consensus_occupancy > 0, np.log(post.consensus_occupancy), -np.inf)
            if isinstance(post, LocalWarpPosterior)
            else logsumexp(post.log_a, axis=0)
        )
        mass[p], support[p] = np.exp(log_m), np.exp(log_n)
        rows = np.isfinite(log_m)
        kinetic[p][rows] = np.exp(post.log_a[rows] - log_m[rows, None]) @ v - lm.coordinates[rows]
        widths = np.zeros(len(lm.coordinates))
        intervals = lm.support_intervals
        if intervals is None:
            intervals = [(lm.coordinates[0], lm.coordinates[-1])]
        for lo, hi in intervals:
            within = np.flatnonzero((lm.coordinates >= lo) & (lm.coordinates <= hi))
            if len(within) > 1:
                u = lm.coordinates[within]
                widths[within] = (np.r_[u[1:], u[-1]] - np.r_[u[0], u[:-1]]) / 2
        if rows.any():
            widths = np.where(widths > 0, widths, 0.0)
            log_widths = np.full_like(widths, -np.inf)
            np.log(widths, out=log_widths, where=widths > 0)
            log_weights = log_m[rows] + log_widths[rows]
            if np.isfinite(logsumexp(log_weights)):
                average[p] = np.exp(log_weights - logsumexp(log_weights)) @ kinetic[p][rows]
            matched[p] = np.exp(log_n - logsumexp(log_n))
        cols = (
            np.isfinite(logsumexp(post.log_a, axis=0)) if isinstance(post, LocalWarpPosterior) else np.isfinite(log_n)
        )
        residual[p, cols] = np.exp(post.log_a[:, cols] - log_n[None, cols]).T @ lm.expression - result.barycenter[cols]
        ix = np.flatnonzero(projection.patients == lm.patient)
        if len(ix):
            logs = projection.log_retained[ix]
            normalizer = logsumexp(logs)
            coverage[p] = np.exp(normalizer) / len(ix)
            if np.isfinite(normalizer):
                cell[p] = np.exp(logs - normalizer) @ projection.conditional_probability[ix]
    return PatientQuantities(tuple(kinetic), tuple(mass), average, matched, cell, coverage, residual, support, v)
