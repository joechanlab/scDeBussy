"""Blocked costs and stable sufficient-statistic minimization."""

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .types import InsufficientSupportError


def costs(landmarks, barycenter, block_size=64):
    """Compute costs without a patient-landmark x consensus x gene tensor."""
    x, w = landmarks.expression, landmarks.precision
    delta = np.empty((len(x), len(barycenter)))
    for start in range(0, len(barycenter), block_size):
        stop = min(start + block_size, len(barycenter))
        # One patient row at a time avoids cancellation in ||x-b||^2.
        for i in range(len(x)):
            delta[i, start:stop] = ((barycenter[start:stop] - x[i]) ** 2 * w[i]).mean(axis=1)
    return delta


def initialize(landmarks, coordinates=None, locality_radius=None):
    """Equal-patient mean at actual consensus coordinates, then fill missing B.

    Only the initial consensus may fill globally unsupported coordinates; this
    never changes patient support or the fixed EM objective.
    """
    from .projection import interpolation_weights

    coordinates = landmarks[0].coordinates if coordinates is None else coordinates
    size, genes = len(coordinates), landmarks[0].expression.shape[1]
    total, counts = np.zeros((size, genes)), np.zeros(size)
    for lm in landmarks:
        weights = interpolation_weights(
            coordinates, lm.coordinates, np.inf if locality_radius is None else locality_radius, lm.support_intervals
        )
        supported = (weights.sum(1) > 0) & ~np.any((weights > 0) & ~lm.matchable, axis=1)
        total[supported] += weights[supported] @ lm.expression
        counts += supported
    supported = counts > 0
    if not supported.any():
        available = [lm for lm in landmarks if lm.matchable.any()]
        if not available:
            raise InsufficientSupportError("no matchable landmark; inspect locality radius and n_min")
        # Very narrow observed domains may fall entirely between consensus knots.
        # Seed B only; neither patient support nor posterior rows are modified.
        return np.mean(
            [
                np.column_stack(
                    [
                        np.interp(coordinates, lm.coordinates[lm.matchable], lm.expression[lm.matchable, g])
                        for g in range(genes)
                    ]
                )
                for lm in available
            ],
            axis=0,
        )
    total[supported] /= counts[supported, None]
    for g in range(genes):
        total[:, g] = np.interp(coordinates, coordinates[supported], total[supported, g])
    return total


@dataclass(frozen=True)
class Statistics:
    """Ordinary and log support, plus its numerically stable minimizer."""

    s0: np.ndarray
    s1: np.ndarray
    log_s0: np.ndarray
    mean: np.ndarray


def sufficient_statistics(landmarks, posteriors, previous):
    """Normalize log weights before summing; retain only structural zeros."""
    means = previous.copy()
    log_s0 = np.full(previous.shape, -np.inf)
    log_w = np.concatenate([np.log(lm.precision) for lm in landmarks], axis=0)
    x = np.concatenate([lm.expression for lm in landmarks], axis=0)
    log_a = np.concatenate([p.log_a for p in posteriors], axis=0)
    for j in range(len(previous)):
        weights = log_a[:, j, None] + log_w
        denominator = logsumexp(weights, axis=0)
        valid = np.isfinite(denominator)
        log_s0[j] = denominator
        if valid.any():
            means[j, valid] = np.sum(np.exp(weights[:, valid] - denominator[valid]) * x[:, valid], axis=0)
    s0 = np.exp(log_s0)
    return Statistics(s0, s0 * means, log_s0, means)


def expected_match_energy(landmarks, posteriors, barycenter):
    """Return the posterior expected squared-error matching energy."""
    return sum(float(np.sum(p.a * costs(lm, barycenter))) for lm, p in zip(landmarks, posteriors, strict=True))
