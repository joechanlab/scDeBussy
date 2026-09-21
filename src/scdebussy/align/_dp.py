"""Log-space DAG inference. Kernels contain no affine-gap or boundary rules."""

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.special import logsumexp

from ._model import JointGraph, NoValidPathError


@njit(cache=True)
def _messages(src, dst, log_weight, n_nodes):
    forward = np.full(n_nodes, -np.inf)
    backward = np.full(n_nodes, -np.inf)
    best = np.full(n_nodes, -np.inf)
    previous = np.full(n_nodes, -1, np.int64)
    forward[0] = best[0] = 0.0
    backward[-1] = 0.0
    for e in range(len(src)):
        a, b = src[e], dst[e]
        forward[b] = np.logaddexp(forward[b], forward[a] + log_weight[e])
        candidate = best[a] + log_weight[e]
        if candidate > best[b]:
            best[b] = candidate
            previous[b] = e
    for e in range(len(src) - 1, -1, -1):
        a, b = src[e], dst[e]
        backward[a] = np.logaddexp(backward[a], log_weight[e] + backward[b])
    return forward, backward, best, previous


@njit(cache=True)
def _count_messages(src, dst, log_weight, is_match, n_nodes, max_count):
    counts = np.full((n_nodes, max_count + 1), -np.inf)
    counts[0, 0] = 0.0
    for e in range(len(src)):
        step = int(is_match[e])
        for k in range(max_count + 1 - step):
            counts[dst[e], k + step] = np.logaddexp(counts[dst[e], k + step], counts[src[e], k] + log_weight[e])
    return counts[-1]


def _numpy_messages(graph, weights):
    forward = np.full(graph.n_nodes, -np.inf)
    backward = np.full(graph.n_nodes, -np.inf)
    best = np.full(graph.n_nodes, -np.inf)
    previous = np.full(graph.n_nodes, -1, np.int64)
    forward[0] = best[0] = backward[-1] = 0.0
    for node in range(graph.n_nodes):
        for e in range(graph.offsets[node], graph.offsets[node + 1]):
            dst = graph.target[e]
            forward[dst] = np.logaddexp(forward[dst], forward[node] + weights[e])
            if best[node] + weights[e] > best[dst]:
                best[dst], previous[dst] = best[node] + weights[e], e
    for node in range(graph.n_nodes - 2, -1, -1):
        edges = slice(graph.offsets[node], graph.offsets[node + 1])
        backward[node] = logsumexp(weights[edges] + backward[graph.target[edges]])
    return forward, backward, best, previous


@dataclass(frozen=True)
class Posterior:
    """Unconditional match marginals and exact lattice diagnostics."""

    log_z: float
    log_a: np.ndarray
    insertion: np.ndarray
    deletion: np.ndarray
    expected_energy: float
    entropy: float
    expected_insertions: float
    expected_deletions: float
    expected_open_i: float
    expected_open_d: float
    count_probability: np.ndarray
    map_edges: np.ndarray
    map_energy: float
    edge_probability: np.ndarray

    @property
    def a(self):
        return np.exp(self.log_a)

    @property
    def expected_matches(self):
        return float(np.exp(logsumexp(self.log_a)))

    @property
    def no_match_probability(self):
        return float(self.count_probability[0])

    def low_match_probability(self, alpha=0.1):
        """Return P(N_M / T < alpha), including the strict inequality."""
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must lie in [0, 1]")
        return float(self.count_probability[np.arange(len(self.count_probability)) / self.log_a.shape[1] < alpha].sum())


def infer(model, *, backend="numba", count_distribution=True):
    """Infer in O(T_p T); count distributions add O(T_p T min(T_p,T))."""
    if backend not in {"numba", "numpy"}:
        raise ValueError("backend must be 'numba' or 'numpy'")
    g = model.graph
    with np.errstate(over="raise"):
        weights = -model.beta * g.energy
    f, b, best, previous = (
        _messages(g.source, g.target, weights, g.n_nodes) if backend == "numba" else _numpy_messages(g, weights)
    )
    z = float(f[-1])
    if not np.isfinite(z):
        raise NoValidPathError("no finite-weight admissible path")
    log_ep = f[g.source] + weights + b[g.target] - z
    ep = np.exp(log_ep)
    log_a = np.full(model.delta.shape, -np.inf)
    matched = g.match_i >= 0
    np.logaddexp.at(log_a, (g.match_i[matched], g.match_j[matched]), log_ep[matched])
    # expm1 preserves tiny gap probabilities near a fully matched row/column.
    insertion = np.clip(-np.expm1(np.minimum(0.0, logsumexp(log_a, axis=1))), 0.0, 1.0)
    deletion = np.clip(-np.expm1(np.minimum(0.0, logsumexp(log_a, axis=0))), 0.0, 1.0)
    if count_distribution:
        function = _count_messages if backend == "numba" else _count_messages.py_func
        counts = np.exp(function(g.source, g.target, weights, matched, g.n_nodes, min(model.delta.shape)) - z)
    else:
        # Only used inside EM. Every returned final posterior includes exact counts.
        counts = np.full(min(model.delta.shape) + 1, np.nan)
    path, node = [], g.n_nodes - 1
    while node:
        e = int(previous[node])
        if e < 0:
            raise NoValidPathError("MAP backtrace has no predecessor")
        path.append(e)
        node = g.source[e]
    energy = float(ep @ g.energy)
    entropy = max(0.0, model.beta * energy + z)
    return Posterior(
        z,
        log_a,
        insertion,
        deletion,
        energy,
        entropy,
        float(ep @ g.insertions),
        float(ep @ g.deletions),
        float(ep @ g.opens_i),
        float(ep @ g.opens_d),
        counts,
        np.asarray(path[::-1], dtype=np.int64),
        float(-best[-1] / model.beta),
        ep,
    )


def _accumulate_ranges(lo, hi, weight, size):
    """Sum ``weight`` over half-open ``[lo, hi)`` index ranges via a diff array.

    Vectorized (no Python loop over edges); handles ranges of any width, which
    matters for open-mode terminal edges that can span many indices at once.
    """
    diff = np.zeros(size + 1)
    valid = hi > lo
    np.add.at(diff, lo[valid], weight[valid])
    np.add.at(diff, hi[valid], -weight[valid])
    return np.cumsum(diff)[:size]


@dataclass(frozen=True)
class JointPosterior:
    """Joint warp-graph posterior.

    Deliberately does not reuse old ``Posterior`` field names/meanings (see
    JOINT_WARP_CONTRACT.md): ``A`` is now a weighted *correspondence* matrix,
    not a match-probability matrix, and occupancy is a separate,
    directly-aggregated quantity from it.
    """

    log_z: float
    log_a: np.ndarray
    patient_occupancy: np.ndarray
    consensus_occupancy: np.ndarray
    patient_unmatched_internal: np.ndarray
    patient_unmatched_terminal: np.ndarray
    consensus_unmatched_internal: np.ndarray
    consensus_unmatched_terminal: np.ndarray
    expected_energy: float
    entropy: float
    expected_match_11: float
    expected_match_12: float
    expected_match_21: float
    expected_open_i: float
    expected_open_d: float
    map_edges: np.ndarray
    map_energy: float
    edge_probability: np.ndarray

    @property
    def a(self):
        return np.exp(self.log_a)

    @property
    def patient_unmatched(self):
        return self.patient_unmatched_internal + self.patient_unmatched_terminal

    @property
    def consensus_unmatched(self):
        return self.consensus_unmatched_internal + self.consensus_unmatched_terminal


def infer_joint(model, *, backend="numba"):
    """Forward/backward inference over the joint warp graph.

    Reuses the exact same ``_messages``/``_numpy_messages`` kernels as old-mode
    ``infer`` (they only need topologically-sorted ``source``/``target``/weight
    arrays); occupancy, correspondence, and transition diagnostics are computed
    as a pure NumPy post-pass over the resulting edge posterior ``ep`` and the
    ``JointGraph``'s edge annotations, per JOINT_WARP_CONTRACT.md Q1-Q3.
    """
    if backend not in {"numba", "numpy"}:
        raise ValueError("backend must be 'numba' or 'numpy'")
    g = model.graph
    if not isinstance(g, JointGraph):
        raise ValueError("infer_joint requires a model with warp='on'")
    n, t = model.delta.shape
    with np.errstate(over="raise"):
        weights = -model.beta * g.energy
    f, b, best, previous = (
        _messages(g.source, g.target, weights, g.n_nodes) if backend == "numba" else _numpy_messages(g, weights)
    )
    z = float(f[-1])
    if not np.isfinite(z):
        raise NoValidPathError("no finite-weight admissible path")
    log_ep = f[g.source] + weights + b[g.target] - z
    ep = np.exp(log_ep)

    log_a = np.full((n, t), -np.inf)
    for ci, cj, cw in ((g.corr0_i, g.corr0_j, g.corr0_w), (g.corr1_i, g.corr1_j, g.corr1_w)):
        valid = ci >= 0
        np.logaddexp.at(log_a, (ci[valid], cj[valid]), log_ep[valid] + np.log(cw[valid]))

    def occupancy(lo1, hi1, k1, lo2, hi2, k2, size, kind):
        lo = np.concatenate([lo1, lo2])
        hi = np.concatenate([hi1, hi2])
        k = np.concatenate([k1, k2])
        w = np.concatenate([ep, ep])
        mask = k == kind
        return _accumulate_ranges(lo[mask], hi[mask], w[mask], size)

    patient_occupancy = occupancy(g.p1_lo, g.p1_hi, g.p1_kind, g.p2_lo, g.p2_hi, g.p2_kind, n, 1)
    patient_unmatched_internal = occupancy(g.p1_lo, g.p1_hi, g.p1_kind, g.p2_lo, g.p2_hi, g.p2_kind, n, 2)
    patient_unmatched_terminal = occupancy(g.p1_lo, g.p1_hi, g.p1_kind, g.p2_lo, g.p2_hi, g.p2_kind, n, 3)
    consensus_occupancy = occupancy(g.c1_lo, g.c1_hi, g.c1_kind, g.c2_lo, g.c2_hi, g.c2_kind, t, 1)
    consensus_unmatched_internal = occupancy(g.c1_lo, g.c1_hi, g.c1_kind, g.c2_lo, g.c2_hi, g.c2_kind, t, 2)
    consensus_unmatched_terminal = occupancy(g.c1_lo, g.c1_hi, g.c1_kind, g.c2_lo, g.c2_hi, g.c2_kind, t, 3)

    expected_match_11 = float(ep[g.kind == 0].sum())
    expected_match_12 = float(ep[g.kind == 3].sum())
    expected_match_21 = float(ep[g.kind == 4].sum())
    expected_open_i = float(ep @ g.opens_i)
    expected_open_d = float(ep @ g.opens_d)

    path, node = [], g.n_nodes - 1
    while node:
        e = int(previous[node])
        if e < 0:
            raise NoValidPathError("MAP backtrace has no predecessor")
        path.append(e)
        node = g.source[e]
    energy = float(ep @ g.energy)
    entropy = max(0.0, model.beta * energy + z)
    return JointPosterior(
        z,
        log_a,
        patient_occupancy,
        consensus_occupancy,
        patient_unmatched_internal,
        patient_unmatched_terminal,
        consensus_unmatched_internal,
        consensus_unmatched_terminal,
        energy,
        entropy,
        expected_match_11,
        expected_match_12,
        expected_match_21,
        expected_open_i,
        expected_open_d,
        np.asarray(path[::-1], dtype=np.int64),
        float(-best[-1] / model.beta),
        ep,
    )
