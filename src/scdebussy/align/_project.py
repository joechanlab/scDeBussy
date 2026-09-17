"""Unconditional posterior projection and patient-level descriptive quantities."""

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp, xlogy

from ._landmarks import kernel, validate_adata
from ._types import ProjectionResult


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
        log_m = logsumexp(posterior.log_a, axis=1)
        valid_landmarks = np.isfinite(log_m)
        position_given_landmark = np.zeros_like(posterior.log_a)
        position_given_landmark[valid_landmarks] = np.exp(
            posterior.log_a[valid_landmarks] - log_m[valid_landmarks, None]
        )
        for start in range(0, len(rows), result.config["chunk_size"]):
            ix = rows[start : start + result.config["chunk_size"]]
            raw = kernel(time[ix], lm.coordinates, result.config["locality_radius"])
            totals = raw.sum(1, keepdims=True)
            normalized = np.divide(raw, totals, out=np.zeros_like(raw), where=totals > 0)
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
    positions = conditional @ (np.arange(t) / t)
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

    kinetic_curve: np.ndarray
    matchability: np.ndarray
    kinetic_mean: np.ndarray
    matched_occupancy: np.ndarray
    cell_occupancy: np.ndarray
    coverage: np.ndarray
    residual: np.ndarray
    residual_support: np.ndarray


def summarize(result, projection=None) -> PatientQuantities:
    """Compute rank displacements, occupancy mass/shape, and residuals."""
    projection = project(result) if projection is None else projection
    n, t, genes = len(result.patients), *result.barycenter.shape
    kinetic = np.full((n, t), np.nan)
    matched, cell = np.full((n, t), np.nan), np.full((n, t), np.nan)
    residual = np.full((n, t, genes), np.nan)
    support, mass = np.zeros((n, t)), np.zeros((n, t))
    average, coverage = np.full(n, np.nan), np.zeros(n)
    v = np.arange(t) / t
    for p, (lm, post) in enumerate(zip(result.landmarks, result.posteriors, strict=True)):
        log_m = logsumexp(post.log_a, axis=1)
        log_n = logsumexp(post.log_a, axis=0)
        mass[p], support[p] = np.exp(log_m), np.exp(log_n)
        rows = np.isfinite(log_m)
        kinetic[p, rows] = np.exp(post.log_a[rows] - log_m[rows, None]) @ v - lm.coordinates[rows]
        widths = (np.r_[lm.coordinates[1:], lm.coordinates[-1]] - np.r_[lm.coordinates[0], lm.coordinates[:-1]]) / 2
        if rows.any():
            log_weights = log_m[rows] + np.log(widths[rows])
            average[p] = np.exp(log_weights - logsumexp(log_weights)) @ kinetic[p, rows]
            matched[p] = np.exp(log_n - logsumexp(log_n))
        cols = np.isfinite(log_n)
        residual[p, cols] = np.exp(post.log_a[:, cols] - log_n[None, cols]).T @ lm.expression - result.barycenter[cols]
        ix = np.flatnonzero(projection.patients == lm.patient)
        if len(ix):
            logs = projection.log_retained[ix]
            normalizer = logsumexp(logs)
            coverage[p] = np.exp(normalizer) / len(ix)
            if np.isfinite(normalizer):
                cell[p] = np.exp(logs - normalizer) @ projection.conditional_probability[ix]
    return PatientQuantities(kinetic, mass, average, matched, cell, coverage, residual, support)
