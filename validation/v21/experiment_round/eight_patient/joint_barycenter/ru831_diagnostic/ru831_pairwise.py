"""Focused RU831 pairwise diagnostic: why does RU831 plateau/gap even with terminal gating off?

Uses the exact frozen mechanics of the current joint model (frozen trajectory-250
genes, unit precision `match_cost`, `compile_graph`/`infer_graph` DTW-style
warp/gap graph with its built-in run-length regularization, BETA=5, RHO=0.02)
via direct imports from three_patient_joint.py -- not the older v2
LatticeModel/infer machinery used in the original RU1303-RU581 diagnostic.
RU831 is matched against a single fixed reference (RU581, then RU1083 as a
second check) at the FULL terminal price (ETA), matching the gate-ablation
state where RU831's coverage gate is off and its plateau/gap pathology still
appears. No parameter is retuned.
"""

import csv
import json
import sys
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
from dtw import dtw

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PAIRWISE_DIR = Path(__file__).resolve().parents[2] / "ru1303_ru581_pairwise"
sys.path.insert(0, str(PAIRWISE_DIR))

from run_pairwise import CACHE, GRID, INPUT, MARKERS, ROOT, TYPES, COLORS, interpolate_curve, marker_landmarks  # noqa: E402
from three_patient_joint import BETA, RHO, compile_graph, match_cost  # noqa: E402
from slope_regularization import BAND  # noqa: E402
from focused_followup import ETA  # noqa: E402
from scdebussy.align._dp import _messages  # noqa: E402

OUT = Path(__file__).resolve().parent
REFERENCES = ("RU581", "RU1083")
PATIENT = "RU831"


def infer_traced(compiled, nodes, cost):
    """Same as three_patient_joint.infer_graph, but also returns the forward trace
    annotated with each edge's patient-landmark index, so gap/plateau positions can
    be recovered exactly (mi/mj are -1 for gap-kind edges, so they alone don't say
    which patient landmark was skipped)."""
    src, dst, static, slope, kind, mi, mj = compiled
    energy = static + RHO * slope
    matched = mi >= 0
    energy[matched] += cost[mi[matched], mj[matched]]
    energy = np.ascontiguousarray(energy)
    forward, backward, _best, previous = _messages(src, dst, -BETA * energy, nodes)
    log_z = float(forward[-1])
    if not np.isfinite(log_z):
        raise RuntimeError("no path")
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

    # Walk the trace forward, tracking the patient-landmark index i explicitly.
    # kinds: 0=match, 1/2=warp (both still advance i, mi already gives the index),
    # 3=patient gap (advances i, no match -- the skipped index is the running i),
    # 4=reference gap (i unchanged), 5=initial edge (mi=0 given), 6=terminal tail.
    i_counter = 0
    gap_positions = []
    for edge in trace:
        k = kind[edge]
        if k in (0, 1, 2, 5):
            i_counter = int(mi[edge]) + 1
        elif k == 3:
            gap_positions.append(i_counter)
            i_counter += 1
        # k == 4: reference gap, i unchanged; k == 6: terminal tail, i already 100

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
        "gap_landmark_positions": gap_positions,
        "map_posterior_mean_abs_difference": float(np.mean(np.abs(curve[shared] - map_curve[shared]))),
        "posterior_mean_match_mass": float(mass.mean()),
    }


def longest_run(values):
    changes = np.r_[True, np.diff(values) != 0, True]
    return int(np.diff(np.flatnonzero(changes)).max())


def plateau_range(map_curve):
    """Landmark index range [start, end] of the longest exact-constant MAP run."""
    changes = np.r_[True, np.diff(map_curve) != 0, True]
    idx = np.flatnonzero(changes)
    lengths = np.diff(idx)
    best = int(np.argmax(lengths))
    return int(idx[best]), int(idx[best + 1] - 1)


def gap_ranges(positions):
    """Collapse a sorted list of single-landmark gap positions into contiguous ranges."""
    if not positions:
        return []
    positions = sorted(positions)
    ranges = [[positions[0], positions[0]]]
    for p in positions[1:]:
        if p == ranges[-1][1] + 1:
            ranges[-1][1] = p
        else:
            ranges.append([p, p])
    return [tuple(r) for r in ranges]


def heatmap_grid(position, cell_patients, labels, order):
    grid = np.full((len(order), 100), np.nan)
    for row, patient in enumerate(order):
        selected = cell_patients == patient
        values = position[selected]
        types = labels[selected]
        finite = np.isfinite(values)
        bins = np.clip(np.digitize(values[finite], np.linspace(0, 1, 101)) - 1, 0, 99)
        encoded = np.array([TYPES.index(v) / 2 if v in TYPES else np.nan for v in types[finite]])
        for b in range(100):
            contents = encoded[bins == b]
            contents = contents[np.isfinite(contents)]
            if len(contents) >= 3:
                grid[row, b] = contents.mean()
    return grid


def main():
    fit = np.load(CACHE, allow_pickle=False)
    panel = json.loads(
        (ROOT / "validation/v21/experiment_round/eight_patient/feature_selection/top_250.json").read_text()
    )
    assert fit["genes"].tolist() == panel
    assert not set(MARKERS).intersection(panel)
    names = fit["patient_order"].tolist()
    x = fit["landmark_expression"].reshape(8, 100, 250)
    matchable = fit["landmark_matchable"].reshape(8, 100)
    ru831 = x[names.index(PATIENT)]
    print(f"RU831 matchable landmarks: {matchable[names.index(PATIENT)].sum()}/100")

    adata = ad.read_h5ad(INPUT, backed="r")
    assert np.array_equal(fit["cell_ids"], np.asarray(adata.obs_names, str))
    patients_obs = adata.obs["patient"].astype(str).to_numpy()
    labels_obs = adata.obs["cell_type_final2"].astype(str).to_numpy()
    local = adata.obs["s_local"].to_numpy(float)

    all_summaries = {}
    for reference in REFERENCES:
        print(f"=== RU831 -> {reference} ===")
        ref_expr = x[names.index(reference)]
        cost = match_cost(ru831, ref_expr)

        compiled, nodes = compile_graph(cost, ETA)  # full terminal price: matches gate-off (ablation) state
        result = infer_traced(compiled, nodes, cost)

        gap_pos = result["gap_landmark_positions"]
        ranges = gap_ranges(gap_pos)
        p_start, p_end = plateau_range(result["map_curve"])
        # The diagnostic window of interest is whichever region is largest: the
        # longest internal-gap span (landmarks skipped entirely, MAP curve NaN
        # there) or the longest exact-constant MAP plateau (landmarks compressed
        # onto one reference position). For RU831 these turn out to usually be
        # gap-dominated, not plateau-dominated, so gaps take priority when larger.
        longest_gap_range = max(ranges, key=lambda r: r[1] - r[0]) if ranges else None
        gap_span = (longest_gap_range[1] - longest_gap_range[0] + 1) if longest_gap_range else 0
        plateau_span = p_end - p_start + 1
        if gap_span >= plateau_span and longest_gap_range is not None:
            window_start, window_end = longest_gap_range
            window_kind = "internal_gap"
        else:
            window_start, window_end = p_start, p_end
            window_kind = "map_plateau"

        # cost of the frozen MAP path within the plateau/gap window vs an unconstrained
        # open-begin/open-end DTW "reachability" path in the same cost matrix, restricted
        # to the same window -- checks whether a smoother path is available at comparable
        # expression cost, or whether the cost surface itself forces the plateau/gap.
        dtw_path = dtw(
            cost,
            step_pattern="asymmetric",
            open_begin=True,
            open_end=True,
            window_type="sakoechiba",
            window_args={"window_size": BAND},
            keep_internals=True,
        )
        dtw_pairs = np.column_stack([dtw_path.index1, dtw_path.index2])
        dtw_curve = np.full(100, np.nan)
        for i in range(100):
            js = dtw_pairs[dtw_pairs[:, 0] == i, 1]
            if len(js):
                dtw_curve[i] = GRID[js].mean()

        window = slice(max(0, window_start - 2), min(100, window_end + 3))
        map_idx = result["map_index"]
        window_matched_idx = np.flatnonzero(map_idx[window] >= 0) + window.start
        frozen_window_mean_cost = (
            float(cost[window_matched_idx, map_idx[window_matched_idx]].mean()) if len(window_matched_idx) else float("nan")
        )
        dtw_window_mask = (dtw_pairs[:, 0] >= window.start) & (dtw_pairs[:, 0] < window.stop)
        dtw_window_mean_cost = float(cost[dtw_pairs[dtw_window_mask, 0], dtw_pairs[dtw_window_mask, 1]].mean())

        # global expression-cost landscape check: is the best achievable match cost at
        # the gap/plateau landmarks unusually high relative to the rest of RU831?
        best_cost_per_landmark = cost.min(axis=1)
        window_best_cost = float(best_cost_per_landmark[window].mean())
        overall_best_cost = float(best_cost_per_landmark.mean())

        markers_a = marker_landmarks(adata, PATIENT, MARKERS)
        markers_b = marker_landmarks(adata, reference, MARKERS)
        scale = np.std(np.vstack([markers_a, markers_b]), axis=0)
        scale = np.where(scale > 1e-8, scale, 1)
        valid_map = np.isfinite(result["map_curve"])
        nearest = np.clip(np.rint(result["map_curve"][valid_map] * 99).astype(int), 0, 99)
        heldout_marker_mse = float(np.mean(((markers_a[valid_map] - markers_b[nearest]) / scale) ** 2))

        summary = {
            "reference": reference,
            "terminal_price": ETA,
            "internal_patient_gaps": result["internal_patient_gaps"],
            "internal_reference_gaps": result["internal_reference_gaps"],
            "gap_landmark_ranges": ranges,
            "longest_map_plateau_landmarks": longest_run(result["map_curve"]),
            "plateau_range": [p_start, p_end],
            "plateau_value": float(result["map_curve"][p_start]) if np.isfinite(result["map_curve"][p_start]) else None,
            "diagnostic_window_kind": window_kind,
            "map_posterior_mean_abs_difference": result["map_posterior_mean_abs_difference"],
            "posterior_mean_match_mass": result["posterior_mean_match_mass"],
            "tail_skipped_fraction": result["tail_skipped_fraction"],
            "heldout_marker_mse": heldout_marker_mse,
            "window_landmarks": [window.start, window.stop - 1],
            "frozen_map_window_mean_matched_cost": frozen_window_mean_cost,
            "unconstrained_dtw_window_mean_cost": dtw_window_mean_cost,
            "window_matched_landmark_count": int(len(window_matched_idx)),
            "window_total_landmark_count": int(window.stop - window.start),
            "window_best_achievable_cost_mean": window_best_cost,
            "overall_best_achievable_cost_mean": overall_best_cost,
            "window_cost_ratio_vs_overall": window_best_cost / overall_best_cost if overall_best_cost else float("nan"),
        }
        all_summaries[reference] = summary
        print(json.dumps(summary, indent=2))

        # --- figures (RU581 reference only, primary diagnostic) ---
        if reference == "RU581":
            fig, ax = plt.subplots(figsize=(6.3, 5.7), dpi=170)
            image = ax.imshow(cost.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="magma_r")
            ax.plot(GRID, GRID, color="white", lw=1.2, ls=":", label="identity")
            map_pairs = np.column_stack([np.flatnonzero(map_idx >= 0), map_idx[map_idx >= 0]])
            ax.plot(GRID[map_pairs[:, 0]], GRID[map_pairs[:, 1]], color="#2c7fb8", lw=2, label="joint-model MAP")
            ax.plot(GRID[dtw_pairs[:, 0]], GRID[dtw_pairs[:, 1]], color="#f03b20", lw=1.4, ls="--", label=f"open DTW band {BAND} (reachability)")
            ax.axvspan(GRID[window.start], GRID[min(window.stop, 99)], color="yellow", alpha=0.15, label="plateau/gap window")
            ax.set(xlabel="RU831 local pseudotime", ylabel=f"{reference} local pseudotime",
                   title=f"RU831→{reference} trajectory-250 unit-precision cost")
            ax.legend(fontsize=7, loc="upper left")
            fig.colorbar(image, ax=ax, label="mean squared expression difference")
            fig.tight_layout()
            fig.savefig(OUT / "cost_matrix_paths.png")
            fig.savefig(OUT / "cost_matrix_paths.pdf")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6.7, 4.4), dpi=170)
            ax.plot(GRID, GRID, color="gray", lw=1, ls=":", label="identity")
            ax.plot(GRID, result["curve"], color="#2c7fb8", lw=2, label="joint-model posterior mean")
            ax.plot(GRID, result["map_curve"], color="#2c7fb8", lw=1, ls="--", label="joint-model MAP")
            ax.plot(GRID, dtw_curve, color="#f03b20", lw=1.6, ls="-", label=f"open DTW band {BAND} (reachability)")
            ax.axvspan(GRID[window.start], GRID[min(window.stop, 99)], color="yellow", alpha=0.15)
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="RU831 s_local", ylabel=f"mapped {reference} position",
                   title=f"RU831→{reference} mapping function")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(OUT / "mapping_functions.png")
            fig.savefig(OUT / "mapping_functions.pdf")
            plt.close(fig)

            # side-by-side heatmap: Raw | joint-style pairwise mapping | DTW reachability alternative
            order = (PATIENT, reference)
            mapped_joint = local.copy()
            ru_mask = patients_obs == PATIENT
            mapped_joint[ru_mask] = interpolate_curve(result["curve"], local[ru_mask])
            mapped_dtw = local.copy()
            mapped_dtw[ru_mask] = interpolate_curve(dtw_curve, local[ru_mask])
            cmap = LinearSegmentedColormap.from_list("types", [COLORS[t] for t in TYPES])
            fig, axes = plt.subplots(1, 3, figsize=(13, 2.7), dpi=170, sharey=True)
            for ax, (title, position) in zip(
                axes,
                (("Raw", local), ("Joint-model pairwise mapping", mapped_joint), (f"Open DTW band {BAND} (best smooth alternative)", mapped_dtw)),
                strict=True,
            ):
                grid = heatmap_grid(position, patients_obs, labels_obs, order)
                ax.imshow(np.ma.masked_invalid(grid), aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1,
                           extent=[0, 1, len(order) - 0.5, -0.5])
                ax.set(yticks=np.arange(len(order)), yticklabels=order, xlim=(0, 1), title=title, xlabel="shared position [0,1]")
                ax.set_facecolor("#e8e8e8")
            fig.legend(
                handles=[mpatches.Patch(color=COLORS[t], label=t) for t in TYPES]
                + [mpatches.Patch(color="#e8e8e8", label="empty / <3 finite cells")],
                ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.1),
            )
            fig.tight_layout(rect=[0, 0.1, 1, 1])
            fig.savefig(OUT / "comparison_heatmap.png", bbox_inches="tight")
            plt.close(fig)

    (OUT / "ru831_diagnostic.json").write_text(json.dumps(all_summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
