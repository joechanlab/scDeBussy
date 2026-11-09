"""Function bodies extracted verbatim from the pinned historical backend."""
import numpy as np
from scdebussy.align._dp import _messages
GRID=np.linspace(0,1,100)
ETA=0.516917977676361
LAMBDA=3*ETA
BAND=60
OMEGA=.05
MAX_RUN=12
GAP_SCALE=3.0
BETA=5.0
RHO=.02
MATCH_COST_SENTINEL=1e4
_SECOND_DIFF=None

def build_soft_graph(cost, tail_eta=ETA):
    """Compile the run-regularized graph with a separate RU581 terminal-tail price."""
    patient_gap_mode = 2 * MAX_RUN + 1
    reference_gap_mode = 2 * MAX_RUN + 2
    states = [None]
    indices = {}
    for total in range(2, 201):
        for i in range(max(1, total - 100), min(100, total - 1) + 1):
            j = total - i
            if 1 <= j <= 100:
                for mode in range(reference_gap_mode + 1):
                    indices[i, j, mode] = len(states)
                    states.append((i, j, mode))
    sink = len(states)
    rows = []

    def edge(src, dst, base, slope, kind, mi=-1, mj=-1):
        rows.append((src, dst, base, slope, kind, mi, mj))

    for j in range(min(99, BAND) + 1):
        edge(0, indices[1, j + 1, 0], j * ETA + cost[0, j], 0, 5, 0, j)

    for node in range(1, sink):
        i, j, mode = states[node]
        if i < 100 and j < 100 and abs(i - j) <= BAND:
            edge(node, indices[i + 1, j + 1, 0], cost[i, j], 0, 0, i, j)
        if i < 100 and abs(i - (j - 1)) <= BAND:
            old = mode if 1 <= mode <= MAX_RUN else 0
            new = old + 1
            if new <= MAX_RUN:
                # A run of ell repeat moves pays rho*(ell-1)^2 in total.
                increment = (new - 1) ** 2 - (max(old - 1, 0)) ** 2
                edge(node, indices[i + 1, j, new], cost[i, j - 1] + OMEGA, increment, 1, i, j - 1)
        if i < 100 and j < 99 and abs(i - (j + 1)) <= BAND:
            old = mode - MAX_RUN if MAX_RUN < mode <= 2 * MAX_RUN else 0
            new = old + 1
            if new <= MAX_RUN:
                increment = (new - 1) ** 2 - (max(old - 1, 0)) ** 2
                edge(node, indices[i + 1, j + 2, MAX_RUN + new], cost[i, j + 1] + OMEGA, increment, 2, i, j + 1)
        if i < 100:
            gap = GAP_SCALE * (ETA + (mode != patient_gap_mode) * LAMBDA)
            edge(node, indices[i + 1, j, patient_gap_mode], gap, 0, 3)
        if j < 100:
            gap = GAP_SCALE * (ETA + (mode != reference_gap_mode) * LAMBDA)
            edge(node, indices[i, j + 1, reference_gap_mode], gap, 0, 4)
        if i == 100 and mode <= 2 * MAX_RUN:
            edge(node, sink, (100 - j) * tail_eta, 0, 6)

    data = np.asarray(rows)
    arrays = tuple(np.ascontiguousarray(data[:, k], dtype=float if k in (2, 3) else np.int64) for k in range(7))
    return arrays, sink + 1

def compile_graph(cost, tail_price, matchable=None):
    """Compile fixed topology and remove only the initial expression emissions.

    `matchable`, if given, is a length-100 boolean mask over the patient's own
    landmarks (axis 0 of `cost`). Landmarks flagged False get a permanent
    MATCH_COST_SENTINEL penalty added to every ordinary match/warp edge (kind
    0/1/2) they could source, baked into `static` so it survives every later
    `infer_graph` call regardless of what barycenter-dependent cost value is
    recomputed for that landmark on a given EM iteration -- an unsupported
    landmark's real expression is never reliable, so it should never become
    cheap again just because the barycenter moved. Gap/patient-skip edges
    (kind 3) have mi=-1 and are untouched by this mask, so unsupported regions
    can still be crossed via a gap exactly as before. Passing matchable=None
    (the default) reproduces prior behavior exactly for every existing caller.
    """
    graph, nodes = build_soft_graph(cost, tail_eta=tail_price)
    src, dst, base, slope, kind, mi, mj = graph
    static = base.copy()
    matched = mi >= 0
    static[matched] -= cost[mi[matched], mj[matched]]
    if matchable is not None:
        matchable = np.asarray(matchable, bool)
        safe_mi = np.where(matched, mi, 0)
        unsupported = matched & ~matchable[safe_mi]
        static[unsupported] += MATCH_COST_SENTINEL
    return (src, dst, static, slope, kind, mi, mj), nodes

def infer_graph(compiled, nodes, cost):
    """Exact graph posterior and MAP under the unchanged speed/run/gap costs."""
    src, dst, static, slope, kind, mi, mj = compiled
    energy = static + RHO * slope
    matched = mi >= 0
    energy[matched] += cost[mi[matched], mj[matched]]
    energy = np.ascontiguousarray(energy)
    forward, backward, _best, previous = _messages(src, dst, -BETA * energy, nodes)
    log_z = float(forward[-1])
    if not np.isfinite(log_z):
        raise RuntimeError("joint pilot graph has no path")
    edge_mass = np.exp(forward[src] - BETA * energy + backward[dst] - log_z)
    matrix = np.zeros((100, 100))
    np.add.at(matrix, (mi[matched], mj[matched]), edge_mass[matched])
    trace = []
    node = nodes - 1
    while node:
        edge = int(previous[node])
        if edge < 0:
            raise RuntimeError("MAP backtrace failed")
        trace.append(edge)
        node = int(src[edge])
    trace = np.asarray(trace[::-1], int)
    trace_matched = mi[trace] >= 0
    map_index = np.full(100, -1, int)
    map_index[mi[trace][trace_matched]] = mj[trace][trace_matched]
    mass = matrix.sum(axis=1)
    curve = np.divide(matrix @ GRID, mass, out=np.full(100, np.nan), where=mass > 1e-12)
    map_curve = np.where(map_index >= 0, GRID[np.maximum(map_index, 0)], np.nan)
    shared = np.isfinite(curve) & np.isfinite(map_curve)
    return {
        "matrix": matrix,
        "curve": curve,
        "map_index": map_index,
        "map_curve": map_curve,
        "log_z": log_z,
        "tail_skipped_fraction": float((99 - map_index[map_index >= 0].max()) / 100),
        "internal_patient_gaps": int(np.count_nonzero(kind[trace] == 3)),
        "internal_reference_gaps": int(np.count_nonzero(kind[trace] == 4)),
        "map_posterior_mean_abs_difference": float(np.mean(np.abs(curve[shared] - map_curve[shared]))),
        "map_posterior_late_mean_abs_difference": float(
            np.mean(np.abs(curve[80:][shared[80:]] - map_curve[80:][shared[80:]]))
        ),
        "posterior_mean_match_mass": float(mass.mean()),
        "posterior_late_mean_match_mass": float(mass[80:].mean()),
    }

def match_cost(expression, barycenter):
    """Compute the frozen unit-precision mean squared gene cost."""
    return np.mean((expression[:, None] - barycenter[None]) ** 2, axis=2)

def interpolate_curve(curve, local):
    """Interpolate a landmark mapping at observed local pseudotimes."""
    good = np.isfinite(curve)
    return np.interp(local, GRID[good], curve[good], left=np.nan, right=np.nan)

def second_diff_operator():
    """Build the fixed (98,100) second-difference operator once."""
    global _SECOND_DIFF
    if _SECOND_DIFF is None:
        d = np.zeros((98, 100))
        idx = np.arange(98)
        d[idx, idx] = 1.0
        d[idx, idx + 1] = -2.0
        d[idx, idx + 2] = 1.0
        _SECOND_DIFF = d
    return _SECOND_DIFF

def expectation(x, barycenter, graphs, patients):
    return {patient: infer_graph(*graphs[patient], match_cost(x[patient], barycenter)) for patient in patients}

def barycenter_update(x, posterior, previous, patients):
    numerator = sum(posterior[patient]["matrix"].T @ x[patient] for patient in patients)
    denominator = sum(posterior[patient]["matrix"].sum(axis=0) for patient in patients)
    updated = previous.copy()
    supported = denominator > 1e-10
    updated[supported] = numerator[supported] / denominator[supported, None]
    return updated, denominator

def barycenter_update_smooth(x, posterior, previous, patients, gamma):
    numerator = sum(posterior[patient]["matrix"].T @ x[patient] for patient in patients)
    denominator = sum(posterior[patient]["matrix"].sum(axis=0) for patient in patients)
    if gamma == 0.0:
        return barycenter_update(x, posterior, previous, patients)
    d = second_diff_operator()
    a = np.diag(denominator) + gamma * (d.T @ d)
    updated = np.linalg.solve(a, numerator)
    return updated, denominator

def cell_positions_and_status(posterior, local, cell_patients, patients):
    """Map cells through each patient's posterior curve; approximate per-cell status/unprojected
    from the same per-patient-landmark retained match mass used to decide curve finiteness."""
    mapped = local.copy()
    status = np.full(len(local), "unsupported", dtype="<U11")
    unprojected = np.ones(len(local), float)
    for patient in patients:
        selected = cell_patients == patient
        mapped[selected] = interpolate_curve(posterior[patient]["curve"], local[selected])
        mass = posterior[patient]["matrix"].sum(axis=1)  # retained probability per own-landmark bin, in [0,1]
        own_unprojected = np.clip(1.0 - mass, 0.0, 1.0)
        bins = np.clip((local[selected] * 99).astype(int), 0, 99)
        u = own_unprojected[bins]
        unprojected[selected] = u
        status[selected] = np.where(u > 0.5, "unsupported", "supported")
    return mapped, status, unprojected
