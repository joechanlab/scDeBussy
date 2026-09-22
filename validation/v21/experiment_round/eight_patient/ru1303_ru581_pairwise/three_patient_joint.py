"""Three-patient posterior barycenter pilot with frozen expression-coverage gates."""

import csv
import json
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
from sklearn.neighbors import NearestNeighbors

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from coverage_gate import CHEAP_PRICE, COVERAGE_MARGIN, TERMINAL_K
from focused_followup import ETA
from matplotlib.colors import LinearSegmentedColormap
from run_pairwise import CACHE, COLORS, GRID, INPUT, MARKERS, ROOT, TYPES, interpolate_curve, marker_landmarks
from slope_regularization import BETA, build_soft_graph
from terminal_pricing import RHO

from scdebussy.align._dp import _messages

OUT = Path(__file__).resolve().parent / "three_patient_joint"
PATIENTS = ("RU1303", "RU581", "RU1083")
FULL_PATIENTS = ("RU581", "RU1083")
MAX_ITER = 20
TOL = 1e-6
PATIENCE = 3


def frozen_coverage(x):
    """Infer patient coverage once from three frozen initial expression trajectories."""
    scores = {}
    detail = {}
    for query in PATIENTS:
        centroid = x[query][-TERMINAL_K:].mean(axis=0)
        positions = {}
        for reference in PATIENTS:
            if reference == query:
                continue
            distance = np.mean((x[reference] - centroid) ** 2, axis=1)
            positions[reference] = float(GRID[np.argmin(distance)])
        scores[query] = float(np.median(list(positions.values())))
        detail[query] = positions
    maximum = max(scores.values())
    cheap = {patient: scores[patient] <= maximum - COVERAGE_MARGIN for patient in PATIENTS}
    return scores, detail, cheap


MATCH_COST_SENTINEL = 1e4
"""Permanent energy penalty added to ordinary match/warp edges sourced from an
unmatchable patient landmark (see `compile_graph`'s `matchable` argument). Large
relative to real trajectory-250 unit-precision costs (observed O(1-150)) so
exp(-BETA * energy) underflows cleanly to 0, but far below float overflow --
this is a fixed structural penalty, not a tuned model parameter."""


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


def expectation(x, barycenter, graphs):
    """Align each patient to one shared barycenter with fixed graph topology."""
    return {patient: infer_graph(*graphs[patient], match_cost(x[patient], barycenter)) for patient in PATIENTS}


def barycenter_update(x, posterior, previous):
    """Use posterior correspondence weights for the unchanged mean M-step."""
    numerator = sum(posterior[patient]["matrix"].T @ x[patient] for patient in PATIENTS)
    denominator = sum(posterior[patient]["matrix"].sum(axis=0) for patient in PATIENTS)
    updated = previous.copy()
    supported = denominator > 1e-10
    updated[supported] = numerator[supported] / denominator[supported, None]
    return updated, denominator


def marker_mse(patient, posterior, markers, marker_scale):
    """Evaluate held-out markers against a leave-one-out full-patient template."""
    peers = [peer for peer in FULL_PATIENTS if peer != patient]
    numerator = sum(posterior[peer]["matrix"].T @ markers[peer] for peer in peers)
    denominator = sum(posterior[peer]["matrix"].sum(axis=0) for peer in peers)
    supported = denominator > 1e-10
    template = np.full_like(numerator, np.nan)
    template[supported] = numerator[supported] / denominator[supported, None]
    for gene in range(template.shape[1]):
        template[:, gene] = np.interp(GRID, GRID[supported], template[supported, gene])
    curve = posterior[patient]["curve"]
    valid = np.isfinite(curve)
    predicted = np.column_stack([np.interp(curve[valid], GRID, template[:, gene]) for gene in range(len(MARKERS))])
    return float(np.mean(((markers[patient][valid] - predicted) / marker_scale) ** 2))


def smoothness(barycenter):
    """RMS second difference of the 250-gene consensus trajectory."""
    return float(np.sqrt(np.mean(np.diff(barycenter, n=2, axis=0) ** 2)))


def cell_positions(posterior, local, patients):
    """Project the three observed patient pseudotimes through posterior maps."""
    mapped = local.copy()
    for patient in PATIENTS:
        selected = patients == patient
        mapped[selected] = interpolate_curve(posterior[patient]["curve"], local[selected])
    return mapped


def quality(position, raw, patients, labels):
    """Compute the established 100-bin coverage, cross-patient purity, and movement."""
    finite = np.isfinite(position)
    indices = np.flatnonzero(finite)
    nearest = NearestNeighbors(n_neighbors=50, algorithm="kd_tree").fit(position[indices, None])
    neighbors = nearest.kneighbors(position[indices, None], return_distance=False)
    scores = np.full(len(position), np.nan)
    for local_index, original in enumerate(indices):
        candidate = indices[neighbors[local_index]]
        cross = patients[candidate] != patients[original]
        scores[original] = float(np.mean(labels[candidate][cross] == labels[original])) if cross.any() else 0.0
    bins = np.zeros(len(position), int)
    bins[finite] = np.clip((position[finite] * 99).astype(int), 0, 99)
    coverage = np.mean([len(np.unique(patients[finite & (bins == b)])) / len(PATIENTS) for b in range(100)])
    result = {}
    for patient in (*PATIENTS, "all"):
        selected = patients == patient if patient != "all" else np.ones(len(position), bool)
        valid = selected & finite
        occupied = np.zeros(100, bool)
        occupied[bins[valid]] = True
        result[patient] = {
            "coverage": float(occupied.mean()) if patient != "all" else float(coverage),
            "purity": float(np.nanmean(scores[selected])),
            "mean_displacement": float(np.mean(np.abs(position[valid] - raw[valid]))),
            "max_displacement": float(np.max(np.abs(position[valid] - raw[valid]))),
            "finite_cell_fraction": float(finite[selected].mean()),
        }
    return result


def heatmap_grid(position, patients, labels):
    """Apply the previous 100-bin, three-cell minimum and gray-bin convention."""
    grid = np.full((len(PATIENTS), 100), np.nan)
    for row, patient in enumerate(PATIENTS):
        selected = patients == patient
        values = position[selected]
        types = labels[selected]
        finite = np.isfinite(values)
        bins = np.clip(np.digitize(values[finite], np.linspace(0, 1, 101)) - 1, 0, 99)
        encoded = np.array([TYPES.index(value) / 2 if value in TYPES else np.nan for value in types[finite]])
        for b in range(100):
            contents = encoded[bins == b]
            contents = contents[np.isfinite(contents)]
            if len(contents) >= 3:
                grid[row, b] = contents.mean()
    return grid


def save_heatmap(raw, initial, final, patients, labels):
    """Plot raw, gated pairwise-informed initialization, and final joint fit."""
    cmap = LinearSegmentedColormap.from_list("types", [COLORS[t] for t in TYPES])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.0), dpi=170, sharey=True)
    for ax, (title, position) in zip(
        axes,
        (("Raw", raw), ("Pairwise-informed initialization", initial), ("Final 3-patient joint fit", final)),
        strict=True,
    ):
        ax.imshow(
            np.ma.masked_invalid(heatmap_grid(position, patients, labels)),
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=1,
            extent=[0, 1, len(PATIENTS) - 0.5, -0.5],
        )
        ax.set(
            yticks=np.arange(len(PATIENTS)),
            yticklabels=PATIENTS,
            xlim=(0, 1),
            title=title,
            xlabel="shared position [0,1]",
        )
        ax.set_facecolor("#e8e8e8")
        ru_a = (patients == "RU1303") & (labels == "SCLC-A")
        ax.axvline(np.nanmedian(position[ru_a]), color="black", ls="--", lw=0.8)
    fig.legend(
        handles=[mpatches.Patch(color=COLORS[t], label=t) for t in TYPES]
        + [mpatches.Patch(color="#e8e8e8", label="empty / <3 finite cells")],
        ncol=4,
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(OUT / "three_patient_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def main():
    """Run one fixed three-patient posterior-EM pilot and write compact diagnostics."""
    OUT.mkdir(exist_ok=True)
    fit = np.load(CACHE, allow_pickle=False)
    panel = json.loads(
        (ROOT / "validation/v21/experiment_round/eight_patient/feature_selection/top_250.json").read_text()
    )
    assert fit["genes"].tolist() == panel
    assert not set(MARKERS).intersection(panel)
    names = fit["patient_order"].tolist()
    all_expression = fit["landmark_expression"].reshape(8, 100, 250)
    expression = {patient: all_expression[names.index(patient)] for patient in PATIENTS}
    scores, peer_positions, cheap = frozen_coverage(expression)
    prices = {patient: CHEAP_PRICE if cheap[patient] else ETA for patient in PATIENTS}
    with (OUT / "frozen_gate.json").open("w") as handle:
        json.dump(
            {"scores": scores, "peer_positions": peer_positions, "gate_on": cheap, "tail_prices": prices},
            handle,
            indent=2,
        )
        handle.write("\n")
    # Seed from the two expression-supported full trajectories, then perform
    # one gated posterior update to obtain the pairwise-informed initialization.
    seed = (expression["RU581"] + expression["RU1083"]) / 2
    graphs = {patient: compile_graph(match_cost(expression[patient], seed), prices[patient]) for patient in PATIENTS}
    seed_post = expectation(expression, seed, graphs)
    initial, _ = barycenter_update(expression, seed_post, seed)
    posterior = expectation(expression, initial, graphs)
    initial_post = posterior
    barycenter = initial
    history = []
    stable = 0
    converged = False
    for iteration in range(1, MAX_ITER + 1):
        before = -sum(posterior[patient]["log_z"] for patient in PATIENTS)
        updated, support = barycenter_update(expression, posterior, barycenter)
        following = expectation(expression, updated, graphs)
        after = -sum(following[patient]["log_z"] for patient in PATIENTS)
        slack = 1e-7 * max(1.0, abs(before), abs(after))
        if after > before + slack:
            raise ArithmeticError(f"posterior EM objective increased at iteration {iteration}: {before} -> {after}")
        relative = (before - after) / max(1.0, abs(before))
        history.append(
            {
                "iteration": iteration,
                "negative_log_partition": after,
                "relative_improvement": relative,
                "barycenter_max_change": float(np.max(np.abs(updated - barycenter))),
                "min_consensus_match_support": float(support.min()),
            }
        )
        barycenter, posterior = updated, following
        stable = stable + 1 if relative <= TOL else 0
        if stable >= PATIENCE:
            converged = True
            break

    adata = ad.read_h5ad(INPUT, backed="r")
    assert np.array_equal(fit["cell_ids"], np.asarray(adata.obs_names, str))
    all_patients = adata.obs["patient"].astype(str).to_numpy()
    selected = np.isin(all_patients, PATIENTS)
    patients = all_patients[selected]
    labels = adata.obs["cell_type_final2"].astype(str).to_numpy()[selected]
    raw = adata.obs["s_local"].to_numpy(float)[selected]
    initial_position = cell_positions(initial_post, raw, patients)
    final_position = cell_positions(posterior, raw, patients)
    markers = {patient: marker_landmarks(adata, patient, MARKERS) for patient in PATIENTS}
    marker_scale = np.std(np.vstack(list(markers.values())), axis=0)
    marker_scale = np.where(marker_scale > 1e-8, marker_scale, 1)
    quality_by_phase = {
        "raw": quality(raw, raw, patients, labels),
        "initial": quality(initial_position, raw, patients, labels),
        "final": quality(final_position, raw, patients, labels),
    }
    rows = []
    for phase, state in (("initial", initial_post), ("final", posterior)):
        for patient in PATIENTS:
            p = state[patient]
            row = {
                "phase": phase,
                "patient": patient,
                "frozen_coverage_score": scores[patient],
                "gate_on": cheap[patient],
                "terminal_price": prices[patient],
                "terminal_skipped_fraction": p["tail_skipped_fraction"],
                "internal_patient_gaps": p["internal_patient_gaps"],
                "internal_reference_gaps": p["internal_reference_gaps"],
                "map_posterior_mean_abs_difference": p["map_posterior_mean_abs_difference"],
                "map_posterior_late_mean_abs_difference": p["map_posterior_late_mean_abs_difference"],
                "posterior_mean_match_mass": p["posterior_mean_match_mass"],
                "heldout_marker_mse": marker_mse(patient, state, markers, marker_scale),
                "late_map_80": float(p["map_curve"][80]),
                "late_map_90": float(p["map_curve"][90]),
                "late_map_99": float(p["map_curve"][99]),
                "late_posterior_80": float(p["curve"][80]),
                "late_posterior_90": float(p["curve"][90]),
                "late_posterior_99": float(p["curve"][99]),
                **quality_by_phase[phase][patient],
            }
            rows.append(row)
    with (OUT / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "convergence.json").open("w") as handle:
        json.dump(
            {
                "converged": converged,
                "iterations": len(history),
                "max_iterations": MAX_ITER,
                "tolerance": TOL,
                "patience": PATIENCE,
                "initial_smoothness_rms_second_difference": smoothness(initial),
                "final_smoothness_rms_second_difference": smoothness(barycenter),
                "overall_quality": quality_by_phase,
                "history": history,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    np.savez_compressed(
        OUT / "pilot_state.npz",
        initial_barycenter=initial,
        final_barycenter=barycenter,
        patient_order=np.asarray(PATIENTS),
        initial_curves=np.stack([initial_post[patient]["curve"] for patient in PATIENTS]),
        final_curves=np.stack([posterior[patient]["curve"] for patient in PATIENTS]),
    )
    save_heatmap(raw, initial_position, final_position, patients, labels)
    print(
        json.dumps(
            {"coverage": scores, "gate": cheap, "converged": converged, "iterations": len(history), "rows": rows},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
