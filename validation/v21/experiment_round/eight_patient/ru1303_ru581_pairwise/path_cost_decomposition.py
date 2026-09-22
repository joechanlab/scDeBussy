"""Score explicit RU1303-to-RU581 paths under the frozen pairwise objective."""

import csv
import json
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from focused_followup import ETA, OMEGA, build_graph, graph_infer
from matplotlib.colors import LinearSegmentedColormap
from run_pairwise import (
    CACHE,
    COLORS,
    GRID,
    INPUT,
    MARKERS,
    PATIENTS,
    ROOT,
    TYPES,
    interpolate_curve,
    label_grid,
    marker_landmarks,
)
from slope_regularization import build_soft_graph

from scdebussy.align._dp import _messages

OUT = Path(__file__).resolve().parent / "path_cost_decomposition"
RHO = 0.02
EXISTING_RHO = (0.0, 0.02, 0.1, 0.4)
GAP_SCALE = 3.0


def map_path(graph, n_nodes, required_late=None):
    """Find the MAP path, optionally fixing a smooth late tail and excluding gaps."""
    src, dst, energy, kind, mi, mj = graph
    allowed = np.ones(len(src), bool)
    if required_late is not None:
        allowed &= (kind != 3) & (kind != 4)
        late = mi >= 80
        allowed[late] &= mj[late] == required_late[mi[late] - 80]
    weights = np.where(allowed, -energy, -np.inf)
    forward, _backward, best, previous = _messages(src, dst, weights, n_nodes)
    if not np.isfinite(forward[-1]):
        raise RuntimeError("constrained smooth late path is unreachable")
    trace = []
    node = n_nodes - 1
    while node:
        edge = int(previous[node])
        if edge < 0:
            raise RuntimeError("MAP backtrace failed")
        trace.append(edge)
        node = src[edge]
    trace = np.asarray(trace[::-1], int)
    matched = mi[trace] >= 0
    curve = np.full(100, -1, int)
    curve[mi[trace][matched]] = mj[trace][matched]
    return curve, {
        "map_cost": float(-best[-1]),
        "patient_gap": int(np.sum(kind[trace] == 3)),
        "reference_gap": int(np.sum(kind[trace] == 4)),
    }


def run_penalty(steps, value):
    """Sum (ell-1)^2 over consecutive runs of one speed-move kind."""
    total = 0
    for move in (0, 2):
        mask = steps == move
        padded = np.r_[False, mask, False]
        change = np.diff(padded.astype(int))
        lengths = np.flatnonzero(change == -1) - np.flatnonzero(change == 1)
        total += int(np.sum(np.maximum(lengths - 1, 0) ** 2))
    return value * total


def longest_true_run(mask):
    """Return the longest contiguous sequence of true move indicators."""
    padded = np.r_[False, mask, False]
    change = np.diff(padded.astype(int))
    lengths = np.flatnonzero(change == -1) - np.flatnonzero(change == 1)
    return int(lengths.max()) if len(lengths) else 0


def longest_plateau(values):
    """Count the longest run of identical mapped landmark positions."""
    changes = np.r_[True, np.diff(values) != 0, True]
    return int(np.diff(np.flatnonzero(changes)).max())


def score(name, index_map, cost, local, ru, markers_a, markers_b, scale, gap_count=0, graph_cost=None):
    """Decompose a complete all-matched monotone landmark mapping."""
    if np.any(index_map < 0) or np.any(np.diff(index_map) < 0) or np.any(np.diff(index_map) > 2):
        raise ValueError(f"{name}: mapping must match all rows with DTW steps 0/1/2")
    steps = np.diff(index_map)
    expression = float(cost[np.arange(100), index_map].sum())
    expression_late = float(cost[np.arange(80, 100), index_map[80:]].sum())
    warp = float(OMEGA * (np.count_nonzero(steps == 0) + np.count_nonzero(steps == 2)))
    terminal_ref = int(index_map[0] + 99 - index_map[-1])
    terminal_cost = float(ETA * terminal_ref)
    gap = terminal_cost  # All four candidates match every patient landmark, with no internal gaps.
    regularization = float(run_penalty(steps, RHO))
    total = expression + warp + gap + regularization
    curve = GRID[index_map]
    mapped = interpolate_curve(curve, local[ru])
    displacement = np.abs(mapped - local[ru])
    marker_mse = float(np.mean(((markers_a[80:] - markers_b[index_map[80:]]) / scale) ** 2))
    row = {
        "candidate": name,
        "C_expr": expression,
        "C_expr_late_80_99": expression_late,
        "C_warp": warp,
        "C_gap": gap,
        "C_regularization_rho_0.02": regularization,
        "C_total_rho_0.02": total,
        "C_total_rho_0": expression + warp + gap,
        "C_total_rho_0.1": expression + warp + gap + run_penalty(steps, 0.1),
        "C_total_rho_0.4": expression + warp + gap + run_penalty(steps, 0.4),
        "map_position_80": float(curve[80]),
        "map_position_90": float(curve[90]),
        "map_position_99": float(curve[99]),
        "late_position_min": float(curve[80:].min()),
        "late_position_max": float(curve[80:].max()),
        "heldout_marker_mse_late": marker_mse,
        "mean_ru1303_cell_displacement": float(displacement.mean()),
        "max_ru1303_cell_displacement": float(displacement.max()),
        "longest_plateau_whole": longest_plateau(index_map),
        "longest_plateau_late": longest_plateau(index_map[80:]),
        "longest_repeat_move_run": longest_true_run(steps == 0),
        "longest_skip2_move_run": longest_true_run(steps == 2),
        "repeat_move_count": int(np.count_nonzero(steps == 0)),
        "skip2_move_count": int(np.count_nonzero(steps == 2)),
        "biological_gap_count": int(gap_count),
        "terminal_ru581_skips": terminal_ref,
        "original_graph_MAP_cost_rho0": graph_cost,
    }
    if graph_cost is not None and abs(row["C_total_rho_0"] - graph_cost) > 1e-8:
        raise AssertionError(f"{name}: decomposition does not reproduce graph MAP cost")
    return row, curve


def main():
    """Compare four prespecified complete paths and write minimal diagnostics."""
    OUT.mkdir(exist_ok=True)
    fit = np.load(CACHE, allow_pickle=False)
    panel = json.loads(
        (ROOT / "validation/v21/experiment_round/eight_patient/feature_selection/top_250.json").read_text()
    )
    assert fit["genes"].tolist() == panel
    assert not set(MARKERS).intersection(panel)
    names = fit["patient_order"].tolist()
    expression = fit["landmark_expression"].reshape(8, 100, 250)
    a = expression[names.index("RU1303")]
    b = expression[names.index("RU581")]
    cost = np.mean((a[:, None] - b[None]) ** 2, axis=2)
    adata = ad.read_h5ad(INPUT, backed="r")
    assert np.array_equal(fit["cell_ids"], np.asarray(adata.obs_names, str))
    patient = adata.obs["patient"].astype(str).to_numpy()
    cell_type = adata.obs["cell_type_final2"].astype(str).to_numpy()
    local = adata.obs["s_local"].to_numpy(float)
    ru = patient == "RU1303"
    markers_a = marker_landmarks(adata, "RU1303", MARKERS)
    markers_b = marker_landmarks(adata, "RU581", MARKERS)
    scale = np.std(np.vstack([markers_a, markers_b]), axis=0)
    scale = np.where(scale > 1e-8, scale, 1)

    graph, nodes = build_graph(cost, gap_scale=GAP_SCALE)
    current, current_stats = map_path(graph, nodes)
    soft_graph, soft_nodes = build_soft_graph(cost)
    src, dst, base, slope, kind, mi, mj = soft_graph
    required_late = np.arange(40, 60)
    smooth_energy = np.ascontiguousarray(base + RHO * slope)
    smooth, smooth_stats = map_path((src, dst, smooth_energy, kind, mi, mj), soft_nodes, required_late=required_late)
    previous = list(csv.DictReader((OUT.parent / "landmark_matches.csv").open()))
    plateau = np.rint(np.array([float(row["dtw_band_60_ru581"]) for row in previous]) * 99).astype(int)
    identity = np.arange(100)
    paths = (
        ("current_capped_MAP", current, current_stats),
        ("smooth_40_to_59", smooth, smooth_stats),
        ("plateau_DTW_band60", plateau, None),
        ("identity", identity, None),
    )
    rows, curves = [], {}
    for name, path, stats in paths:
        if stats and (stats["patient_gap"] or stats["reference_gap"]):
            raise AssertionError(f"{name}: expected no biological gaps")
        row, curve = score(
            name,
            path,
            cost,
            local,
            ru,
            markers_a,
            markers_b,
            scale,
            graph_cost=stats["map_cost"] if name == "current_capped_MAP" else None,
        )
        rows.append(row)
        curves[name] = curve
    smooth_row = next(row for row in rows if row["candidate"] == "smooth_40_to_59")
    if abs(smooth_row["C_total_rho_0.02"] - smooth_stats["map_cost"]) > 1e-8:
        raise AssertionError("forced smooth-path decomposition does not match soft graph energy")
    for rho in EXISTING_RHO[1:]:
        energy = np.ascontiguousarray(base + rho * slope)
        weighted_graph = (src, dst, energy, kind, mi, mj)
        optimum = graph_infer(weighted_graph, soft_nodes, 5)["map_cost"]
        _forced, forced_stats = map_path(weighted_graph, soft_nodes, required_late=required_late)
        forced_cost = forced_stats["map_cost"]
        if optimum > forced_cost + 1e-8:
            raise AssertionError("soft graph missed a reachable smooth candidate")
        for row in rows:
            row[f"soft_graph_MAP_optimum_rho_{rho}"] = float(optimum)
            row[f"best_forced_smooth_MAP_cost_rho_{rho}"] = float(forced_cost)
    with (OUT / "cost_decomposition.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    color = {
        "current_capped_MAP": "#2c7fb8",
        "smooth_40_to_59": "#31a354",
        "plateau_DTW_band60": "#f03b20",
        "identity": "white",
    }
    fig, (ax, bars) = plt.subplots(1, 2, figsize=(11.2, 5.1), dpi=170, gridspec_kw={"width_ratios": [1.15, 1]})
    image = ax.imshow(cost.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="magma_r")
    for name, path, _ in paths:
        ax.plot(
            GRID,
            GRID[path],
            color=color[name],
            lw=1.8,
            ls=":" if name == "identity" else "-",
            label=name.replace("_", " "),
        )
    ax.set(xlabel="RU1303 s_local", ylabel="RU581 s_local", title="Frozen pairwise cost and explicit paths")
    ax.legend(fontsize=7, loc="upper left")
    fig.colorbar(image, ax=ax, label="mean squared expression cost")
    labels = ["current", "smooth", "plateau", "identity"]
    parts = (
        ("C_expr", "#4c78a8"),
        ("C_warp", "#f58518"),
        ("C_gap", "#e45756"),
        ("C_regularization_rho_0.02", "#72b7b2"),
    )
    bottom = np.zeros(4)
    for field, shade in parts:
        values = np.array([row[field] for row in rows])
        bars.bar(labels, values, bottom=bottom, color=shade, label=field.replace("C_", ""))
        bottom += values
    for index, total in enumerate(bottom):
        bars.text(index, total + 0.8, f"{total:.1f}", ha="center", fontsize=8)
    bars.set(ylabel="total path cost", title="Cost decomposition, rho=0.02")
    bars.legend(fontsize=7, ncol=2)
    bars.set_ylim(0, bottom.max() * 1.15)
    fig.tight_layout()
    fig.savefig(OUT / "paths_and_costs.png")
    plt.close(fig)

    cmap = LinearSegmentedColormap.from_list("types", [COLORS[t] for t in TYPES])
    fig, axes = plt.subplots(1, 4, figsize=(15, 2.6), dpi=170, sharey=True)
    panels = (
        ("Raw", curves["identity"]),
        ("Current path", curves["current_capped_MAP"]),
        ("Smooth candidate", curves["smooth_40_to_59"]),
        ("Plateau candidate", curves["plateau_DTW_band60"]),
    )
    ru_a = ru & (cell_type == "SCLC-A")
    for axis, (title, curve) in zip(axes, panels, strict=True):
        position = local.copy()
        position[ru] = interpolate_curve(curve, local[ru])
        grid = label_grid(position, {"patient": patient, "type": cell_type})
        axis.imshow(
            np.ma.masked_invalid(grid),
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=1,
            extent=[0, 1, 1.5, -0.5],
        )
        axis.set(yticks=[0, 1], yticklabels=PATIENTS, xlim=(0, 1), title=title, xlabel="RU581 position [0,1]")
        axis.set_facecolor("#e8e8e8")
        axis.axvline(np.nanmedian(position[ru_a]), color="black", lw=0.8, ls="--")
    fig.legend(
        handles=[mpatches.Patch(color=COLORS[t], label=t) for t in TYPES]
        + [mpatches.Patch(color="#e8e8e8", label="empty / <3 finite cells")],
        ncol=4,
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(OUT / "comparison_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
