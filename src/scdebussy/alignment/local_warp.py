"""Immutable affine-gap lattice and deliberately exponential reference oracle.

Coordinates count consumed landmarks. Edge kinds are M=0, I=1, D=2;
terminal offsets are represented by their consumed lengths, not interleavings.
"""

from dataclasses import dataclass, replace
from functools import cached_property, lru_cache

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class TerminalCosts:
    """Per-landmark terminal prices for one patient-to-consensus alignment."""

    patient_begin: float
    patient_end: float
    consensus_begin: float
    consensus_end: float

    def __post_init__(self):
        if any(
            not np.isfinite(value) or value < 0
            for value in (self.patient_begin, self.patient_end, self.consensus_begin, self.consensus_end)
        ):
            raise ValueError("terminal costs must be finite and nonnegative")

    @classmethod
    def symmetric(cls, cost):
        """Create equal terminal prices on both axes and boundaries."""
        return cls(cost, cost, cost, cost)


class NoValidPathError(ValueError):
    """The lattice has no admissible source-to-sink path."""


def immutable_array(value, dtype=float):
    """Copy into an immutable bytes-backed array, including its write flag."""
    a = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True)
class LocalWarpGraph:
    """Topologically ordered edges for the opt-in local warp graph.

    Segment fields describe, per edge, up to two contiguous index ranges it
    touches on the patient axis (``p1_*``/``p2_*``) and consensus axis
    (``c1_*``/``c2_*``); ``*_kind`` is 0=none, 1=matched, 2=unmatched-internal,
    3=unmatched-terminal. Two segments are needed only for the source-bundled
    leading-skip-plus-first-match edges (terminal-unmatched prefix + matched
    cell in one edge). Correspondence fields (``corr0_*``/``corr1_*``) give up to
    two weighted ``(i, j, w)`` cells whose weighted sum over edges is ``A``;
    ``corr*_i == -1`` marks an unused slot. ``kind`` labels the edge's
    transition class: 0=(1,1) match, 1=internal patient gap, 2=internal
    consensus gap, 3=(1,2) warp match, 4=(2,1) warp match, 5=terminal/boundary.
    """

    source: np.ndarray
    target: np.ndarray
    energy: np.ndarray
    kind: np.ndarray
    corr0_i: np.ndarray
    corr0_j: np.ndarray
    corr0_w: np.ndarray
    corr1_i: np.ndarray
    corr1_j: np.ndarray
    corr1_w: np.ndarray
    p1_lo: np.ndarray
    p1_hi: np.ndarray
    p1_kind: np.ndarray
    p2_lo: np.ndarray
    p2_hi: np.ndarray
    p2_kind: np.ndarray
    c1_lo: np.ndarray
    c1_hi: np.ndarray
    c1_kind: np.ndarray
    c2_lo: np.ndarray
    c2_hi: np.ndarray
    c2_kind: np.ndarray
    opens_i: np.ndarray
    opens_d: np.ndarray
    offsets: np.ndarray
    n_nodes: int


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
class LocalWarpLattice:
    """One patient's fixed-cost lattice; 1x1 is permitted for analytic fixtures.

    ``warp="on"`` opts into the local-warp posterior-warp graph mode (see
    ``validation/v21/experiment_round/``): match edges may
    also consume ``(1,2)`` or ``(2,1)`` landmarks at cost ``omega_warp`` above the
    averaged/summed ``delta``, and open-mode terminal edges pay an independent
    ``lambda_terminal_gap`` open charge (old mode terminal edges pay none). This
    is entirely opt-in: with ``warp="off"`` (the default), ``graph``/inference are
    byte-identical to the pre-existing behavior.
    """

    delta: np.ndarray
    beta: float
    eta_i: float
    eta_d: float
    lambda_i: float
    lambda_d: float
    matchable: np.ndarray | None = None
    mode: str = "open"
    warp: str = "off"
    omega_warp: float = 0.0
    terminal_costs: TerminalCosts | None = None

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
        if self.warp not in {"off", "on"}:
            raise ValueError("warp must be 'off' or 'on'")
        for name in ("omega_warp",):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.warp == "on":
            if self.mode != "open":
                raise ValueError("warp='on' is only defined for mode='open'")
            if self.terminal_costs is None:
                # Calibrated prior against degenerate all-terminal paths (see contract Q9.2):
                # this does not forbid the all-terminal path, it guarantees it is never free.
                raise ValueError("warp='on' requires lambda_terminal_gap > 0")
        mask = np.ones(delta.shape[0], bool) if self.matchable is None else np.asarray(self.matchable, bool)
        if mask.shape != (delta.shape[0],):
            raise ValueError("matchable must have one entry per patient landmark")
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "matchable", immutable_array(mask, bool))

    @cached_property
    def graph(self):
        """Compile both terminal modes to the same numeric DAG interface."""
        if self.warp == "on":
            return self._build_local_warp_graph()
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

    def _build_local_warp_graph(self):
        """Build the local warp DAG directly (cost-dependent; not template-cached).

        Node numbering matches ``_build_graph`` exactly (``node(i, j, s)``), so
        the two graphs share no code but stay topologically compatible with the
        same forward/backward kernels in ``_dp.py``.
        """
        n, t = self.delta.shape
        delta, matchable = self.delta, self.matchable
        eta_i, eta_d, lam_i, lam_d = self.eta_i, self.eta_d, self.lambda_i, self.lambda_d
        omega, terminal = self.omega_warp, self.terminal_costs
        sink = 1 + 3 * (n + 1) * (t + 1)
        rows = []

        def node(i, j, s):
            return 1 + 3 * (i * (t + 1) + j) + s

        NONE, MATCHED, INTERNAL, TERMINAL = 0, 1, 2, 3
        M, I, D, W12, W21, TERM = 0, 1, 2, 3, 4, 5

        def edge(
            src,
            dst,
            cost,
            kind,
            corr=(),
            p1=None,
            p2=None,
            c1=None,
            c2=None,
            oi=0,
            od=0,
        ):
            (c0i, c0j, c0w) = corr[0] if len(corr) > 0 else (-1, -1, 0.0)
            (c1i, c1j, c1w) = corr[1] if len(corr) > 1 else (-1, -1, 0.0)
            p1lo, p1hi, p1k = p1 if p1 is not None else (0, 0, NONE)
            p2lo, p2hi, p2k = p2 if p2 is not None else (0, 0, NONE)
            c1lo, c1hi, c1k = c1 if c1 is not None else (0, 0, NONE)
            c2lo, c2hi, c2k = c2 if c2 is not None else (0, 0, NONE)
            rows.append(
                (
                    src,
                    dst,
                    cost,
                    kind,
                    c0i,
                    c0j,
                    c0w,
                    c1i,
                    c1j,
                    c1w,
                    p1lo,
                    p1hi,
                    p1k,
                    p2lo,
                    p2hi,
                    p2k,
                    c1lo,
                    c1hi,
                    c1k,
                    c2lo,
                    c2hi,
                    c2k,
                    oi,
                    od,
                )
            )

        def lead(i, j):
            """Terminal-open charge for a leading skip of i patient / j consensus states."""
            return i * terminal.patient_begin + j * terminal.consensus_begin

        # All-terminal (degenerate) path: nothing matched at all.
        edge(
            0,
            sink,
            n * terminal.patient_begin + t * terminal.consensus_begin,
            TERM,
            p1=(0, n, TERMINAL),
            c1=(0, t, TERMINAL),
        )
        # Source-bundled leading skip + first match (mirrors _build_graph's open-mode bundling).
        for i in range(n):
            if not matchable[i]:
                continue
            for j in range(t):
                edge(
                    0,
                    node(i + 1, j + 1, 0),
                    i * eta_i + j * eta_d + lead(i, j) + delta[i, j],
                    M,
                    corr=((i, j, 1.0),),
                    p1=(0, i, TERMINAL),
                    p2=(i, i + 1, MATCHED),
                    c1=(0, j, TERMINAL),
                    c2=(j, j + 1, MATCHED),
                )
            for j in range(t - 1):
                cost = i * eta_i + j * eta_d + lead(i, j) + 0.5 * (delta[i, j] + delta[i, j + 1]) + omega
                edge(
                    0,
                    node(i + 1, j + 2, 0),
                    cost,
                    W12,
                    corr=((i, j, 0.5), (i, j + 1, 0.5)),
                    p1=(0, i, TERMINAL),
                    p2=(i, i + 1, MATCHED),
                    c1=(0, j, TERMINAL),
                    c2=(j, j + 2, MATCHED),
                )
            if i + 1 < n and matchable[i + 1]:
                for j in range(t):
                    cost = i * eta_i + j * eta_d + lead(i, j) + delta[i, j] + delta[i + 1, j] + omega
                    edge(
                        0,
                        node(i + 2, j + 1, 0),
                        cost,
                        W21,
                        corr=((i, j, 1.0), (i + 1, j, 1.0)),
                        p1=(0, i, TERMINAL),
                        p2=(i, i + 2, MATCHED),
                        c1=(0, j, TERMINAL),
                        c2=(j, j + 1, MATCHED),
                    )

        # Interior transitions, from every reachable (i, j, s).
        for i in range(n + 1):
            for j in range(t + 1):
                for s in range(3):
                    src = node(i, j, s)
                    if i < n and j < t and matchable[i]:
                        edge(
                            src,
                            node(i + 1, j + 1, 0),
                            delta[i, j],
                            M,
                            corr=((i, j, 1.0),),
                            p1=(i, i + 1, MATCHED),
                            c1=(j, j + 1, MATCHED),
                        )
                    if i < n and j + 1 < t and matchable[i]:
                        cost = 0.5 * (delta[i, j] + delta[i, j + 1]) + omega
                        edge(
                            src,
                            node(i + 1, j + 2, 0),
                            cost,
                            W12,
                            corr=((i, j, 0.5), (i, j + 1, 0.5)),
                            p1=(i, i + 1, MATCHED),
                            c1=(j, j + 2, MATCHED),
                        )
                    if i + 1 < n and j < t and matchable[i] and matchable[i + 1]:
                        cost = delta[i, j] + delta[i + 1, j] + omega
                        edge(
                            src,
                            node(i + 2, j + 1, 0),
                            cost,
                            W21,
                            corr=((i, j, 1.0), (i + 1, j, 1.0)),
                            p1=(i, i + 2, MATCHED),
                            c1=(j, j + 1, MATCHED),
                        )
                    if i < n:
                        edge(
                            src,
                            node(i + 1, j, 1),
                            eta_i + (s != 1) * lam_i,
                            I,
                            p1=(i, i + 1, INTERNAL),
                            oi=int(s != 1),
                        )
                    if j < t:
                        edge(
                            src,
                            node(i, j + 1, 2),
                            eta_d + (s != 2) * lam_d,
                            D,
                            c1=(j, j + 1, INTERNAL),
                            od=int(s != 2),
                        )
                    if s == 0 and i > 0 and j > 0:
                        cost = (n - i) * terminal.patient_end + (t - j) * terminal.consensus_end
                        edge(
                            src,
                            sink,
                            cost,
                            TERM,
                            p1=(i, n, TERMINAL) if n > i else None,
                            c1=(j, t, TERMINAL) if t > j else None,
                        )
        rows.sort(key=lambda r: r[0])
        data = np.asarray(rows, dtype=float)
        source = data[:, 0].astype(np.int64)
        target = data[:, 1].astype(np.int64)
        energy = data[:, 2]
        kind = data[:, 3].astype(np.int64)
        corr0_i = data[:, 4].astype(np.int64)
        corr0_j = data[:, 5].astype(np.int64)
        corr0_w = data[:, 6]
        corr1_i = data[:, 7].astype(np.int64)
        corr1_j = data[:, 8].astype(np.int64)
        corr1_w = data[:, 9]
        p1_lo = data[:, 10].astype(np.int64)
        p1_hi = data[:, 11].astype(np.int64)
        p1_kind = data[:, 12].astype(np.int64)
        p2_lo = data[:, 13].astype(np.int64)
        p2_hi = data[:, 14].astype(np.int64)
        p2_kind = data[:, 15].astype(np.int64)
        c1_lo = data[:, 16].astype(np.int64)
        c1_hi = data[:, 17].astype(np.int64)
        c1_kind = data[:, 18].astype(np.int64)
        c2_lo = data[:, 19].astype(np.int64)
        c2_hi = data[:, 20].astype(np.int64)
        c2_kind = data[:, 21].astype(np.int64)
        opens_i = data[:, 22].astype(np.int64)
        opens_d = data[:, 23].astype(np.int64)
        offsets = np.concatenate(([0], np.cumsum(np.bincount(source, minlength=sink + 1))))
        return LocalWarpGraph(
            immutable_array(source, np.int64),
            immutable_array(target, np.int64),
            immutable_array(energy, float),
            immutable_array(kind, np.int64),
            immutable_array(corr0_i, np.int64),
            immutable_array(corr0_j, np.int64),
            immutable_array(corr0_w, float),
            immutable_array(corr1_i, np.int64),
            immutable_array(corr1_j, np.int64),
            immutable_array(corr1_w, float),
            immutable_array(p1_lo, np.int64),
            immutable_array(p1_hi, np.int64),
            immutable_array(p1_kind, np.int64),
            immutable_array(p2_lo, np.int64),
            immutable_array(p2_hi, np.int64),
            immutable_array(p2_kind, np.int64),
            immutable_array(c1_lo, np.int64),
            immutable_array(c1_hi, np.int64),
            immutable_array(c1_kind, np.int64),
            immutable_array(c2_lo, np.int64),
            immutable_array(c2_hi, np.int64),
            immutable_array(c2_kind, np.int64),
            immutable_array(opens_i, np.int64),
            immutable_array(opens_d, np.int64),
            immutable_array(offsets, np.int64),
            sink + 1,
        )


@lru_cache(maxsize=8)
def _graph_template(n, t, mask, eta_i, eta_d, lambda_i, lambda_d, mode):
    model = LocalWarpLattice(np.zeros((n, t)), 1.0, eta_i, eta_d, lambda_i, lambda_d, np.asarray(mask), mode)
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


@dataclass(frozen=True)
class LocalWarpOracleResult:
    """Exhaustive path posterior for the local warp graph; tiny lattices only."""

    log_z: float
    log_a: np.ndarray
    patient_occupancy: np.ndarray
    consensus_occupancy: np.ndarray
    patient_unmatched_internal: np.ndarray
    patient_unmatched_terminal: np.ndarray
    consensus_unmatched_internal: np.ndarray
    consensus_unmatched_terminal: np.ndarray
    expected_match_11: float
    expected_match_12: float
    expected_match_21: float
    edge_probability: np.ndarray
    entropy: float
    map_energy: float
    paths: tuple
    energies: np.ndarray


def enumerate_local_warp_paths(model, *, max_paths=1_000_000):
    """Brute-force oracle for the local warp graph: same recipe as enumerate_paths."""
    graph = model.graph
    if not isinstance(graph, LocalWarpGraph):
        raise ValueError("enumerate_local_warp_paths requires a model with warp='on'")
    n, t = model.delta.shape
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
    log_a = np.full((n, t), -np.inf)
    ep = np.zeros(len(graph.source))
    m_i, o_j = np.zeros(n), np.zeros(t)
    u_int, u_term = np.zeros(n), np.zeros(n)
    d_int, d_term = np.zeros(t), np.zeros(t)
    e11 = e12 = e21 = 0.0
    for path, log_prob, prob in zip(paths, log_q, q, strict=True):
        for e in path:
            ep[e] += prob
            for ci, cj, cw in (
                (graph.corr0_i[e], graph.corr0_j[e], graph.corr0_w[e]),
                (graph.corr1_i[e], graph.corr1_j[e], graph.corr1_w[e]),
            ):
                if ci >= 0:
                    log_a[ci, cj] = np.logaddexp(log_a[ci, cj], log_prob + np.log(cw))
            for lo, hi, k in (
                (graph.p1_lo[e], graph.p1_hi[e], graph.p1_kind[e]),
                (graph.p2_lo[e], graph.p2_hi[e], graph.p2_kind[e]),
            ):
                if hi > lo:
                    if k == 1:
                        m_i[lo:hi] += prob
                    elif k == 2:
                        u_int[lo:hi] += prob
                    elif k == 3:
                        u_term[lo:hi] += prob
            for lo, hi, k in (
                (graph.c1_lo[e], graph.c1_hi[e], graph.c1_kind[e]),
                (graph.c2_lo[e], graph.c2_hi[e], graph.c2_kind[e]),
            ):
                if hi > lo:
                    if k == 1:
                        o_j[lo:hi] += prob
                    elif k == 2:
                        d_int[lo:hi] += prob
                    elif k == 3:
                        d_term[lo:hi] += prob
            kind = graph.kind[e]
            if kind == 0:
                e11 += prob
            elif kind == 3:
                e12 += prob
            elif kind == 4:
                e21 += prob
    return LocalWarpOracleResult(
        float(log_z),
        log_a,
        m_i,
        o_j,
        u_int,
        u_term,
        d_int,
        d_term,
        float(e11),
        float(e12),
        float(e21),
        ep,
        float(-np.dot(q, log_q)),
        float(energies.min()),
        tuple(paths),
        energies,
    )
