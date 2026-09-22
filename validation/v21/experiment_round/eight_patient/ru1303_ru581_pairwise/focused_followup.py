"""Compare v2 temperature and a pairwise-only run-limited DTW/gap graph."""

import csv
import json
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from run_pairwise import (
    CACHE,
    COLORS,
    GRID,
    INPUT,
    MARKERS,
    PATIENTS,
    TYPES,
    interpolate_curve,
    label_grid,
    marker_landmarks,
)

from scdebussy.align._dp import _messages, infer
from scdebussy.align._model import LatticeModel

OUT = Path(__file__).resolve().parent / "focused_followup"
ETA = 0.516917977676361
LAMBDA = 3 * ETA
BAND = 60
OMEGA = 0.05
REPEAT_CAP = 4


def build_graph(cost, gap_scale=1.0):
    """Compile open-end match/repeat/skip plus affine biological gaps."""
    states = [None]
    indices = {}
    for total in range(2, 201):
        for i in range(max(1, total - 100), min(100, total - 1) + 1):
            j = total - i
            if j < 1 or j > 100:
                continue
            for mode in range(REPEAT_CAP + 2):
                indices[i, j, mode] = len(states)
                states.append((i, j, mode))
    sink = len(states)
    rows = []

    def edge(src, dst, energy, kind, mi=-1, mj=-1):
        rows.append((src, dst, energy, kind, mi, mj))

    for j in range(100):
        if j <= BAND:
            edge(0, indices[1, j + 1, 1], j * ETA + cost[0, j], 5, 0, j)

    for node in range(1, sink):
        i, j, mode = states[node]
        if i < 100 and j < 100 and abs(i - j) <= BAND:
            edge(node, indices[i + 1, j + 1, 1], cost[i, j], 0, i, j)
        if 1 <= mode <= REPEAT_CAP and mode < REPEAT_CAP and i < 100 and abs(i - (j - 1)) <= BAND:
            edge(node, indices[i + 1, j, mode + 1], cost[i, j - 1] + OMEGA, 1, i, j - 1)
        if i < 100 and j < 99 and abs(i - (j + 1)) <= BAND:
            edge(node, indices[i + 1, j + 2, 1], cost[i, j + 1] + OMEGA, 2, i, j + 1)
        if i < 100:
            edge(node, indices[i + 1, j, REPEAT_CAP + 1], gap_scale * (ETA + (mode != REPEAT_CAP + 1) * LAMBDA), 3)
        if j < 100:
            edge(node, indices[i, j + 1, 0], gap_scale * (ETA + (mode != 0) * LAMBDA), 4)
        if 1 <= mode <= REPEAT_CAP and i == 100:
            edge(node, sink, (100 - j) * ETA, 6)

    data = np.asarray(rows)
    return tuple(np.ascontiguousarray(data[:, k], dtype=float if k == 2 else np.int64) for k in range(6)), sink + 1


def graph_infer(graph, n_nodes, beta):
    """Return posterior correspondence, entropy, MAP path, and edge counts."""
    src, dst, energy, kind, mi, mj = graph
    f, b, _best, previous = _messages(src, dst, -beta * energy, n_nodes)
    log_z = f[-1]
    if not np.isfinite(log_z):
        raise RuntimeError("run-limited graph has no path")
    edge_mass = np.exp(f[src] - beta * energy + b[dst] - log_z)
    matrix = np.zeros((100, 100))
    matched = mi >= 0
    np.add.at(matrix, (mi[matched], mj[matched]), edge_mass[matched])
    trace = []
    node = n_nodes - 1
    while node:
        e = int(previous[node])
        trace.append(e)
        node = src[e]
    trace = trace[::-1]
    map_curve = np.full(100, np.nan)
    for e in trace:
        if mi[e] >= 0:
            map_curve[mi[e]] = GRID[mj[e]]
    return {
        "matrix": matrix,
        "map": map_curve,
        "path_entropy": float(beta * np.dot(edge_mass, energy) + log_z),
        "map_cost": float(energy[trace].sum()),
        "map_counts": {
            name: int(np.sum(kind[trace] == code))
            for name, code in (("ordinary", 0), ("repeat", 1), ("skip2", 2), ("patient_gap", 3), ("reference_gap", 4))
        },
    }


def summarize(name, matrix, map_curve, path_entropy, local, markers_a, markers_b, scale, ru_a, ru581_n, extras=None):
    """Summarize late mapped positions and support without fitting labels."""
    mass = matrix.sum(1)
    curve = np.divide(matrix @ GRID, mass, out=np.full(100, np.nan), where=mass > 1e-12)
    late = slice(80, 100)
    conditional = np.divide(matrix, mass[:, None], out=np.zeros_like(matrix), where=mass[:, None] > 1e-12)
    row_entropy = -np.sum(np.where(conditional > 0, conditional * np.log(np.maximum(conditional, 1e-300)), 0), axis=1)
    shared = np.isfinite(map_curve[late]) & np.isfinite(curve[late])
    delta = np.abs(map_curve[late][shared] - curve[late][shared])
    nearest = np.clip(np.rint(curve[late] * 99).astype(int), 0, 99)
    marker_mse = float(np.mean(((markers_a[late] - markers_b[nearest]) / scale) ** 2))
    mapped_cells = interpolate_curve(curve, local[ru_a])
    nlo, nhi = np.quantile(local[ru581_n], [0.1, 0.9])
    changes = np.r_[True, np.diff(map_curve) != 0, True]
    lengths = np.diff(np.flatnonzero(changes))
    row = {
        "method": name,
        "late_position_80": float(curve[80]),
        "late_position_90": float(curve[90]),
        "late_position_99": float(curve[99]),
        "map_position_80": float(map_curve[80]) if np.isfinite(map_curve[80]) else None,
        "map_position_90": float(map_curve[90]) if np.isfinite(map_curve[90]) else None,
        "map_position_99": float(map_curve[99]) if np.isfinite(map_curve[99]) else None,
        "late_mean_match_mass": float(mass[late].mean()),
        "late_min_match_mass": float(mass[late].min()),
        "late_mean_row_entropy": float(row_entropy[late].mean()),
        "full_path_entropy": float(path_entropy),
        "late_map_posterior_mean_abs_difference": float(delta.mean()) if len(delta) else None,
        "late_map_posterior_max_abs_difference": float(delta.max()) if len(delta) else None,
        "late_map_matched_count": int(shared.sum()),
        "late_heldout_marker_mse": marker_mse,
        "longest_exact_map_plateau": int(lengths.max()),
        "late_exact_map_plateau": int(np.diff(np.flatnonzero(np.r_[True, np.diff(map_curve[late]) != 0, True])).max()),
        "ru1303_sclc_a_overlap_ru581_sclc_n": float(np.mean((mapped_cells >= nlo) & (mapped_cells <= nhi))),
    }
    if extras:
        row.update(extras)
    return row, curve, mass, row_entropy


def main():
    """Run the small temperature and recurrence diagnostic."""
    OUT.mkdir(exist_ok=True)
    fit = np.load(CACHE, allow_pickle=False)
    names = fit["patient_order"].tolist()
    x = fit["landmark_expression"].reshape(8, 100, 250)
    cost = np.mean((x[names.index("RU1303")][:, None] - x[names.index("RU581")][None]) ** 2, axis=2)
    adata = ad.read_h5ad(INPUT, backed="r")
    assert np.array_equal(fit["cell_ids"], np.asarray(adata.obs_names, str))
    obs = adata.obs
    patient = obs["patient"].astype(str).to_numpy()
    typ = obs["cell_type_final2"].astype(str).to_numpy()
    local = obs["s_local"].to_numpy(float)
    ru_a = (patient == "RU1303") & (typ == "SCLC-A")
    ru581_n = (patient == "RU581") & (typ == "SCLC-N")
    markers_a = marker_landmarks(adata, "RU1303", MARKERS)
    markers_b = marker_landmarks(adata, "RU581", MARKERS)
    scale = np.std(np.vstack([markers_a, markers_b]), axis=0)
    scale = np.where(scale > 1e-8, scale, 1)
    rows, curves, maps = [], {}, {}
    for beta in (0.5, 1, 2, 5, 10):
        model = LatticeModel(cost, beta, ETA, ETA, LAMBDA, LAMBDA, mode="open")
        p = infer(model, backend="numba", count_distribution=False)
        g = model.graph
        curve_map = np.full(100, np.nan)
        edges = p.map_edges
        good = g.match_i[edges] >= 0
        curve_map[g.match_i[edges][good]] = GRID[g.match_j[edges][good]]
        name = f"v2_beta_{beta:g}"
        row, curve, _mass, _entropy = summarize(
            name,
            p.a,
            curve_map,
            p.entropy,
            local,
            markers_a,
            markers_b,
            scale,
            ru_a,
            ru581_n,
            {
                "beta": beta,
                "map_cost": p.map_energy,
                "patient_gap_MAP": int(g.insertions[edges].sum()),
                "reference_gap_MAP": int(g.deletions[edges].sum()),
            },
        )
        rows.append(row)
        curves[name] = curve
        maps[name] = curve_map
    graph, n_nodes = build_graph(cost)
    for beta in (1, 5):
        result = graph_infer(graph, n_nodes, beta)
        name = f"constrained_beta_{beta:g}"
        row, curve, _mass, _entropy = summarize(
            name,
            result["matrix"],
            result["map"],
            result["path_entropy"],
            local,
            markers_a,
            markers_b,
            scale,
            ru_a,
            ru581_n,
            {"beta": beta, "map_cost": result["map_cost"], **result["map_counts"]},
        )
        rows.append(row)
        curves[name] = curve
        maps[name] = result["map"]
    stronger, stronger_nodes = build_graph(cost, gap_scale=3.0)
    result = graph_infer(stronger, stronger_nodes, 5)
    name = "constrained_gap3_beta_5"
    row, curve, _mass, _entropy = summarize(
        name,
        result["matrix"],
        result["map"],
        result["path_entropy"],
        local,
        markers_a,
        markers_b,
        scale,
        ru_a,
        ru581_n,
        {"beta": 5, "gap_scale": 3.0, "map_cost": result["map_cost"], **result["map_counts"]},
    )
    rows.append(row)
    curves[name] = curve
    maps[name] = result["map"]
    with (OUT / "metrics.csv").open("w", newline="") as handle:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    # Keep the original unconstrained DTW band-60 MAP from the previous pairwise run.
    previous = list(csv.DictReader((OUT.parent / "landmark_matches.csv").open()))
    dtw_unconstrained = np.array([float(r["dtw_band_60_ru581"]) for r in previous])
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.2, 4.1), dpi=170)
    image = ax0.imshow(cost.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="magma_r")
    ax0.plot(GRID, GRID, color="white", ls=":", lw=1, label="identity")
    ax0.plot(GRID, dtw_unconstrained, color="red", lw=1.4, label="unconstrained DTW MAP")
    ax0.plot(GRID, maps["constrained_gap3_beta_5"], color="#31a354", lw=1.6, label="constrained MAP, high gap cost")
    ax0.set(xlabel="RU1303 s_local", ylabel="RU581 s_local", title="Frozen pairwise cost and paths")
    ax0.legend(fontsize=7)
    fig.colorbar(image, ax=ax0, label="unit-precision cost")
    ax1.plot(GRID, GRID, color="gray", ls=":", lw=1, label="identity")
    ax1.plot(GRID, curves["v2_beta_1"], color="#2c7fb8", lw=1.6, label="v2 posterior β=1")
    ax1.plot(GRID, curves["v2_beta_5"], color="#2c7fb8", lw=1.3, ls="--", label="v2 posterior β=5")
    ax1.plot(GRID, dtw_unconstrained, color="red", lw=1.2, label="unconstrained DTW MAP")
    ax1.plot(GRID, maps["constrained_gap3_beta_5"], color="#31a354", lw=1.3, ls="--", label="constrained MAP")
    ax1.plot(GRID, curves["constrained_gap3_beta_5"], color="#006d2c", lw=1.6, label="constrained posterior β=5")
    ax1.axhspan(0.421, 0.531, color="red", alpha=0.08)
    ax1.axhspan(0.709, 1, color="cyan", alpha=0.1)
    ax1.set(
        xlim=(0, 1), ylim=(0, 1), xlabel="RU1303 s_local", ylabel="RU581 matched position", title="MAP versus posterior"
    )
    ax1.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "cost_mapping.png")
    plt.close(fig)

    cmap = LinearSegmentedColormap.from_list("types", [COLORS[t] for t in TYPES])
    fig, axes = plt.subplots(1, 4, figsize=(15, 2.6), dpi=170, sharey=True)
    panels = (
        ("Raw", GRID),
        ("Current v2 posterior", curves["v2_beta_1"]),
        ("Unconstrained DTW", dtw_unconstrained),
        ("Constrained DTW posterior", curves["constrained_gap3_beta_5"]),
    )
    ru = patient == "RU1303"
    for ax, (title, curve) in zip(axes, panels, strict=True):
        position = local.copy()
        position[ru] = interpolate_curve(curve, local[ru])
        grid = label_grid(position, {"patient": patient, "type": typ})
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
