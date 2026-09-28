"""Immutable affine-gap lattice and deliberately exponential reference oracle.

Coordinates count consumed landmarks. Edge kinds are M=0, I=1, D=2;
terminal offsets are represented by their consumed lengths, not interleavings.
"""

from dataclasses import dataclass, replace
from functools import cached_property, lru_cache

import numpy as np
from scipy.special import logsumexp


class NoValidPathError(ValueError):
    """The lattice has no admissible source-to-sink path."""


def immutable_array(value, dtype=float):
    """Copy into an immutable bytes-backed array, including its write flag."""
    a = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True)
class Graph:
    """Topologically ordered edges compiled from one lattice definition."""

    source: np.ndarray
    target: np.ndarray
    energy: np.ndarray
    match_i: np.ndarray
    match_j: np.ndarray
    insertions: np.ndarray
    deletions: np.ndarray
    opens_i: np.ndarray
    opens_d: np.ndarray
    offsets: np.ndarray
    n_nodes: int


@dataclass(frozen=True)
class LatticeModel:
    """One patient's fixed-cost lattice; 1x1 is permitted for analytic fixtures."""

    delta: np.ndarray
    beta: float
    eta_i: float
    eta_d: float
    lambda_i: float
    lambda_d: float
    matchable: np.ndarray | None = None
    mode: str = "open"

    def __post_init__(self):
        delta = immutable_array(self.delta)
        if delta.ndim != 2 or min(delta.shape) < 1 or not np.isfinite(delta).all() or (delta < 0).any():
            raise ValueError("delta must be a nonempty finite nonnegative matrix")
        if not np.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("beta must be finite and positive")
        for name in ("eta_i", "eta_d", "lambda_i", "lambda_d"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.mode not in {"open", "global"}:
            raise ValueError("mode must be 'open' or 'global'")
        mask = np.ones(delta.shape[0], bool) if self.matchable is None else np.asarray(self.matchable, bool)
        if mask.shape != (delta.shape[0],):
            raise ValueError("matchable must have one entry per patient landmark")
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "matchable", immutable_array(mask, bool))

    @cached_property
    def graph(self):
        """Compile both terminal modes to the same numeric DAG interface."""
        n, t = self.delta.shape
        template = _graph_template(
            n, t, tuple(self.matchable), self.eta_i, self.eta_d, self.lambda_i, self.lambda_d, self.mode
        )
        energy = template.energy.copy()
        matched = template.match_i >= 0
        energy[matched] += self.delta[template.match_i[matched], template.match_j[matched]]
        return replace(template, energy=immutable_array(energy))

    def _build_graph(self):
        """Build a cost-independent topology once for repeated EM iterations."""
        n, t = self.delta.shape
        sink = 1 + 3 * (n + 1) * (t + 1)
        rows = []

        def node(i, j, s):
            return 1 + 3 * (i * (t + 1) + j) + s

        def edge(src, dst, cost, mi=-1, mj=-1, ni=0, nd=0, oi=0, od=0):
            rows.append((src, dst, cost, mi, mj, ni, nd, oi, od))

        if self.mode == "open":
            edge(0, sink, n * self.eta_i + t * self.eta_d, ni=n, nd=t)
            for i in range(n):
                if self.matchable[i]:
                    for j in range(t):
                        edge(0, node(i + 1, j + 1, 0), i * self.eta_i + j * self.eta_d + self.delta[i, j], i, j, i, j)
        else:
            if self.matchable[0]:
                edge(0, node(1, 1, 0), self.delta[0, 0], 0, 0)
            edge(0, node(1, 0, 1), self.lambda_i + self.eta_i, ni=1, oi=1)
            edge(0, node(0, 1, 2), self.lambda_d + self.eta_d, nd=1, od=1)

        for i in range(n + 1):
            for j in range(t + 1):
                for s in range(3):
                    src = node(i, j, s)
                    if i < n and j < t and self.matchable[i]:
                        edge(src, node(i + 1, j + 1, 0), self.delta[i, j], i, j)
                    if i < n:
                        edge(src, node(i + 1, j, 1), self.eta_i + (s != 1) * self.lambda_i, ni=1, oi=int(s != 1))
                    if j < t:
                        edge(src, node(i, j + 1, 2), self.eta_d + (s != 2) * self.lambda_d, nd=1, od=int(s != 2))
                    if self.mode == "global" and i == n and j == t:
                        edge(src, sink, 0.0)
                    elif self.mode == "open" and s == 0 and i > 0 and j > 0:
                        edge(src, sink, (n - i) * self.eta_i + (t - j) * self.eta_d, ni=n - i, nd=t - j)
        rows.sort(key=lambda r: r[0])
        data = np.asarray(rows)
        arrays = [immutable_array(data[:, k], float if k == 2 else np.int64) for k in range(9)]
        offsets = np.concatenate(([0], np.cumsum(np.bincount(arrays[0], minlength=sink + 1))))
        return Graph(*arrays, immutable_array(offsets, np.int64), sink + 1)


@lru_cache(maxsize=8)
def _graph_template(n, t, mask, eta_i, eta_d, lambda_i, lambda_d, mode):
    model = LatticeModel(np.zeros((n, t)), 1.0, eta_i, eta_d, lambda_i, lambda_d, np.asarray(mask), mode)
    return model._build_graph()


@dataclass(frozen=True)
class OracleResult:
    """Exhaustive path posterior, intended only for tiny lattices."""

    log_z: float
    log_a: np.ndarray
    edge_probability: np.ndarray
    count_probability: np.ndarray
    entropy: float
    map_energy: float
    paths: tuple
    energies: np.ndarray


def enumerate_paths(model, *, max_paths=1_000_000):
    """Enumerate complete paths without using any DP recurrence."""
    graph = model.graph
    paths, energies = [], []

    def visit(node, prefix, cost):
        if node == graph.n_nodes - 1:
            if len(paths) >= max_paths:
                raise ValueError("oracle path limit exceeded; use a smaller lattice")
            paths.append(tuple(prefix))
            energies.append(cost)
            return
        for e in range(graph.offsets[node], graph.offsets[node + 1]):
            visit(graph.target[e], [*prefix, e], cost + graph.energy[e])

    visit(0, [], 0.0)
    if not paths:
        raise NoValidPathError("no admissible path")
    energies = np.asarray(energies)
    log_weights = -model.beta * energies
    log_z = logsumexp(log_weights)
    log_q = log_weights - log_z
    q = np.exp(log_q)
    log_a = np.full(model.delta.shape, -np.inf)
    ep = np.zeros(len(graph.source))
    counts = np.zeros(min(model.delta.shape) + 1)
    for path, log_prob, prob in zip(paths, log_q, q, strict=True):
        matches = 0
        for e in path:
            ep[e] += prob
            if graph.match_i[e] >= 0:
                i, j = graph.match_i[e], graph.match_j[e]
                log_a[i, j] = np.logaddexp(log_a[i, j], log_prob)
                matches += 1
        counts[matches] += prob
    return OracleResult(
        float(log_z), log_a, ep, counts, float(-np.dot(q, log_q)), float(energies.min()), tuple(paths), energies
    )
