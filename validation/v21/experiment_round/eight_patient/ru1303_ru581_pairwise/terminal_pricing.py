"""Focused RU581 terminal-tail price diagnostic on the frozen RU1303 pair."""

import csv
import json
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from focused_followup import ETA, OMEGA, graph_infer
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
from slope_regularization import BETA, build_soft_graph, longest_run, longest_true_run

from scdebussy.align._dp import _messages

OUT = Path(__file__).resolve().parent / "terminal_pricing"
RHO = 0.02
TAIL_PRICES = (ETA, 0.25, 0.10, 0.0)


def trace_and_decompose(graph, n_nodes, cost, rho):
    """Backtrace the MAP and separate the unchanged and varied objective terms."""
    src, dst, base, slope, kind, mi, mj = graph
    energy = np.ascontiguousarray(base + rho * slope)
    _f, _b, best, previous = _messages(src, dst, -BETA * energy, n_nodes)
    edges = []
    node = n_nodes - 1
    while node:
        edge = int(previous[node])
        if edge < 0:
            raise RuntimeError("MAP backtrace failed")
        edges.append(edge)
        node = int(src[edge])
    edges = np.asarray(edges[::-1], int)
    kinds = kind[edges]
    matched = mi[edges] >= 0
    expression = float(cost[mi[edges][matched], mj[edges][matched]].sum())
    warp = float(OMEGA * np.count_nonzero(np.isin(kinds, [1, 2])))
    internal_gap = float(base[edges][np.isin(kinds, [3, 4])].sum())
    leading = float(base[edges][kinds == 5].sum() - cost[mi[edges][kinds == 5], mj[edges][kinds == 5]].sum())
    tail = float(base[edges][kinds == 6].sum())
    regularization = float(rho * slope[edges].sum())
    total = expression + warp + internal_gap + leading + tail + regularization
    np.testing.assert_allclose(total, energy[edges].sum(), rtol=1e-11, atol=1e-9)
    np.testing.assert_allclose(total, -best[-1] / BETA, rtol=1e-11, atol=1e-9)
    map_index = np.full(100, -1, int)
    map_index[mi[edges][matched]] = mj[edges][matched]
    return {
        "expression_cost": expression,
        "warp_cost": warp,
        "internal_biological_gap_cost": internal_gap,
        "ru581_leading_unmatched_cost": leading,
        "ru581_terminal_tail_cost": tail,
        "run_regularization_cost": regularization,
        "total_map_cost": total,
        "ru581_leading_unmatched_length": int(round(leading / ETA)),
        "ru581_terminal_tail_length": 100 - int(map_index[map_index >= 0].max()) - 1,
        "biological_gap_count": int(np.count_nonzero(np.isin(kinds, [3, 4]))),
        "patient_gap_count": int(np.count_nonzero(kinds == 3)),
        "reference_gap_count": int(np.count_nonzero(kinds == 4)),
        "repeat_move_count": int(np.count_nonzero(kinds == 1)),
        "skip2_move_count": int(np.count_nonzero(kinds == 2)),
        "longest_speed_run": max(longest_true_run(kinds == 1), longest_true_run(kinds == 2)),
        "longest_map_plateau": longest_run(map_index),
        "longest_late_map_plateau": longest_run(map_index[80:]),
        "map_index": map_index,
    }


def heldout_mse(curve, markers_a, markers_b, scale):
    """Score the late mapped landmarks against markers excluded from fitting."""
    late = curve[80:]
    if not np.all(np.isfinite(late)):
        return float("nan")
    nearest = np.clip(np.rint(late * 99).astype(int), 0, 99)
    return float(np.mean(((markers_a[80:] - markers_b[nearest]) / scale) ** 2))


def main():
    """Evaluate four tail prices without changing any other model parameter."""
    OUT.mkdir(exist_ok=True)
    fit = np.load(CACHE, allow_pickle=False)
    panel = json.loads(
        (ROOT / "validation/v21/experiment_round/eight_patient/feature_selection/top_250.json").read_text()
    )
    assert fit["genes"].tolist() == panel
    assert not set(MARKERS).intersection(panel)
    names = fit["patient_order"].tolist()
    x = fit["landmark_expression"].reshape(8, 100, 250)
    cost = np.mean((x[names.index("RU1303")][:, None] - x[names.index("RU581")][None]) ** 2, axis=2)
    adata = ad.read_h5ad(INPUT, backed="r")
    assert np.array_equal(fit["cell_ids"], np.asarray(adata.obs_names, str))
    patient = adata.obs["patient"].astype(str).to_numpy()
    cell_type = adata.obs["cell_type_final2"].astype(str).to_numpy()
    local = adata.obs["s_local"].to_numpy(float)
    ru = patient == "RU1303"
    ru_a = ru & (cell_type == "SCLC-A")
    markers_a = marker_landmarks(adata, "RU1303", MARKERS)
    markers_b = marker_landmarks(adata, "RU581", MARKERS)
    scale = np.std(np.vstack([markers_a, markers_b]), axis=0)
    scale = np.where(scale > 1e-8, scale, 1)
    rows = []
    landmark_rows = []
    curves = {}
    for tail_eta in TAIL_PRICES:
        graph, n_nodes = build_soft_graph(cost, tail_eta=tail_eta)
        src, dst, base, slope, kind, mi, mj = graph
        energy = np.ascontiguousarray(base + RHO * slope)
        inferred = graph_infer((src, dst, energy, kind, mi, mj), n_nodes, BETA)
        mass = inferred["matrix"].sum(1)
        curve = np.divide(inferred["matrix"] @ GRID, mass, out=np.full(100, np.nan), where=mass > 1e-12)
        decomp = trace_and_decompose(graph, n_nodes, cost, RHO)
        map_index = decomp.pop("map_index")
        map_curve = np.where(map_index >= 0, GRID[np.maximum(map_index, 0)], np.nan)
        assert decomp["patient_gap_count"] == inferred["map_counts"]["patient_gap"]
        assert decomp["reference_gap_count"] == inferred["map_counts"]["reference_gap"]
        np.testing.assert_allclose(decomp["total_map_cost"], inferred["map_cost"])
        np.testing.assert_allclose(decomp["ru581_terminal_tail_cost"], decomp["ru581_terminal_tail_length"] * tail_eta)
        row = {
            "tail_price_per_landmark": float(tail_eta),
            **decomp,
            "map_late_80": float(map_curve[80]),
            "map_late_90": float(map_curve[90]),
            "map_late_99": float(map_curve[99]),
            "map_late_min": float(np.nanmin(map_curve[80:])),
            "map_late_max": float(np.nanmax(map_curve[80:])),
            "posterior_late_80": float(curve[80]),
            "posterior_late_90": float(curve[90]),
            "posterior_late_99": float(curve[99]),
            "late_mean_match_mass": float(mass[80:].mean()),
            "heldout_marker_mse_map": heldout_mse(map_curve, markers_a, markers_b, scale),
            "heldout_marker_mse_posterior": heldout_mse(curve, markers_a, markers_b, scale),
        }
        rows.append(row)
        landmark_rows.extend(
            {
                "tail_price_per_landmark": float(tail_eta),
                "ru1303_landmark": i,
                "ru1303_s_local": float(GRID[i]),
                "map_ru581_position": float(map_curve[i]),
                "posterior_ru581_position": float(curve[i]),
                "posterior_match_mass": float(mass[i]),
            }
            for i in range(80, 100)
        )
        curves[tail_eta] = curve

    with (OUT / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "late_landmark_mapping.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(landmark_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(landmark_rows)

    cmap = LinearSegmentedColormap.from_list("types", [COLORS[t] for t in TYPES])
    fig, axes = plt.subplots(1, 4, figsize=(14, 2.6), dpi=170, sharey=True)
    panels = (
        ("Raw", GRID),
        ("Current best", curves[ETA]),
        ("Cheap open end (0.10)", curves[0.10]),
        ("Zero cost open end", curves[0.0]),
    )
    for ax, (title, curve) in zip(axes, panels, strict=True):
        position = local.copy()
        position[ru] = interpolate_curve(curve, local[ru])
        grid = label_grid(position, {"patient": patient, "type": cell_type})
        ax.imshow(
            np.ma.masked_invalid(grid),
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=1,
            extent=[0, 1, 1.5, -0.5],
        )
        ax.set(yticks=[0, 1], yticklabels=PATIENTS, xlim=(0, 1), title=title, xlabel="RU581 position [0,1]")
        ax.set_facecolor("#e8e8e8")
        ax.axvline(np.nanmedian(position[ru_a]), color="black", ls="--", lw=0.8)
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
