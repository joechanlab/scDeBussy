"""Patient-label permutation inference; no fitting uses condition labels."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ._project import summarize
from ._types import ComparisonResult


def _metadata(values, patients, name):
    if isinstance(values, Mapping) or isinstance(values, pd.Series):
        if set(values.keys()) != set(patients):
            raise ValueError(f"{name} must contain exactly one entry per fitted patient")
        array = np.asarray([values[p] for p in patients], dtype=object)
    else:
        array = np.asarray(values, dtype=object)
    if array.shape != (len(patients),) or pd.isna(array).any():
        raise ValueError(f"{name} must have one nonmissing value per fitted patient")
    return array


def _bh(pvalues):
    result = np.full_like(pvalues, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(pvalues))
    order = valid[np.argsort(pvalues[valid])]
    if len(order):
        adjusted = pvalues[order] * len(order) / np.arange(1, len(order) + 1)
        result[order] = np.minimum(1.0, np.minimum.accumulate(adjusted[::-1])[::-1])
    return result


def _statistics(summary, labels):
    """All group-dependent support is recomputed for each label assignment."""
    a, b = labels == 0, labels == 1
    genes, t = summary.residual.shape[2], summary.residual.shape[1]
    out = np.full(2 + genes, np.nan)
    if min(a.sum(), b.sum()) < 2:
        return out
    ka = summary.kinetic_mean[a]
    kb = summary.kinetic_mean[b]
    if np.isfinite(ka).sum() >= 2 and np.isfinite(kb).sum() >= 2:
        out[0] = abs(np.nanmean(ka) - np.nanmean(kb))
    qa = summary.cell_occupancy[a]
    qb = summary.cell_occupancy[b]
    qa, qb = qa[np.isfinite(qa).all(1)], qb[np.isfinite(qb).all(1)]
    if min(len(qa), len(qb)) >= 2:
        # W1 integrates the CDF difference over the stored coordinate intervals.
        out[1] = np.abs(np.cumsum(qa.mean(0) - qb.mean(0))[:-1]) @ np.diff(summary.consensus_coordinates)
    for gene in range(genes):
        ra, rb = summary.residual[a, :, gene], summary.residual[b, :, gene]
        va = np.isfinite(ra) & (summary.residual_support[a] > 0)
        vb = np.isfinite(rb) & (summary.residual_support[b] > 0)
        ca, cb = va.sum(0), vb.sum(0)
        ma = np.divide(np.where(va, ra, 0.0).sum(0), ca, out=np.zeros(t), where=ca > 0)
        mb = np.divide(np.where(vb, rb, 0.0).sum(0), cb, out=np.zeros(t), where=cb > 0)
        weights = np.minimum(
            np.where(va, summary.residual_support[a], 0.0).sum(0), np.where(vb, summary.residual_support[b], 0.0).sum(0)
        )
        weights[(ca == 0) | (cb == 0)] = 0.0
        if weights.sum() > 0:
            out[2 + gene] = (weights / weights.sum()) @ (ma - mb) ** 2
    return out


def compare(result, labels, *, scheme, seed, blocks=None, pairs=None, n_perm=9999) -> ComparisonResult:
    """Test two patient groups with explicit exchangeability and RNG seed.

    Missing statistics are never imputed. If any sampled assignment makes a
    statistic undefined, that test is marked not_testable rather than silently
    conditioning its null distribution on convenient assignments.
    """
    if scheme not in {"exchangeable", "blocked", "paired"}:
        raise ValueError("scheme must be exchangeable, blocked, or paired")
    if not isinstance(n_perm, (int, np.integer)) or isinstance(n_perm, bool) or n_perm < 1:
        raise ValueError("n_perm must be a positive integer")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be an explicit nonnegative integer")
    raw = _metadata(labels, result.patients, "labels")
    groups = tuple(pd.unique(raw))
    if len(groups) != 2:
        raise ValueError("exactly two condition groups are required")
    encoded = np.asarray(raw == groups[1], dtype=int)
    units = []
    if scheme == "exchangeable":
        if blocks is not None or pairs is not None:
            raise ValueError("exchangeable permutations do not accept blocks or pairs")
        units = [np.arange(len(raw))]
    else:
        if (scheme == "blocked" and (blocks is None or pairs is not None)) or (
            scheme == "paired" and (pairs is None or blocks is not None)
        ):
            raise ValueError("provide only the metadata corresponding to the chosen scheme")
        meta = _metadata(blocks if scheme == "blocked" else pairs, result.patients, scheme)
        units = [np.flatnonzero(meta == value) for value in pd.unique(meta)]
        if scheme == "paired" and any(len(ix) != 2 or encoded[ix].sum() != 1 for ix in units):
            raise ValueError("each pair must contain exactly two patients, one per condition")
    movable = any(len(np.unique(encoded[ix])) > 1 for ix in units)
    summary = summarize(result)
    observed = _statistics(summary, encoded)
    if not movable:
        observed[:] = np.nan
    exceed = np.zeros(len(observed), int)
    invalid = ~np.isfinite(observed)
    rng = np.random.default_rng(seed)
    for _ in range(n_perm):
        permuted = encoded.copy()
        for ix in units:
            permuted[ix] = rng.permutation(encoded[ix])
        null = _statistics(summary, permuted)
        invalid |= ~np.isfinite(null)
        exceed += null >= observed - 1e-12 * np.maximum(1.0, np.abs(observed))
    pvalues = (1 + exceed).astype(float) / (1 + n_perm)
    pvalues[invalid] = np.nan

    def table(indices, names):
        p = pvalues[indices]
        return pd.DataFrame(
            {
                "statistic": observed[indices],
                "pvalue": p,
                "qvalue": _bh(p),
                "status": np.where(np.isfinite(p), "tested", "not_testable"),
            },
            index=names,
        )

    return ComparisonResult(
        table(slice(0, 1), ["kinetic"]),
        table(slice(1, 2), ["occupancy"]),
        table(slice(2, None), result.genes),
        groups,
        {
            "scheme": scheme,
            "seed": int(seed),
            "n_perm": int(n_perm),
            "correction": "BH within family",
            "min_patients_per_group": 2,
            "invalid_null": "not_testable",
        },
    )
