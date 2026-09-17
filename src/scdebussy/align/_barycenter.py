"""Blocked costs and stable sufficient-statistic minimization."""

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from ._types import InsufficientSupportError


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


def initialize(landmarks):
    """Equal-patient pointwise mean with interpolation only at initialization."""
    size, genes = landmarks[0].expression.shape
    total, counts = np.zeros((size, genes)), np.zeros(size)
    for lm in landmarks:
        total[lm.matchable] += lm.expression[lm.matchable]
        counts += lm.matchable
    supported = counts > 0
    if not supported.any():
        raise InsufficientSupportError("no matchable landmark; inspect locality radius and n_min")
    total[supported] /= counts[supported, None]
    for g in range(genes):
        total[:, g] = np.interp(np.arange(size), np.flatnonzero(supported), total[supported, g])
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
    return sum(float(np.sum(p.a * costs(lm, barycenter))) for lm, p in zip(landmarks, posteriors, strict=True))
