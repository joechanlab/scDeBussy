"""Small pairwise-only soft run-penalty diagnostic for RU1303 to RU581."""

import csv
import json
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
from dtw import dtw

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from focused_followup import BAND, ETA, LAMBDA, OMEGA, build_graph, graph_infer
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

OUT = Path(__file__).resolve().parent / "slope_regularization"
RHO_VALUES = (0.02, 0.1, 0.4)
MAX_RUN = 12  # technical state bound; report if any MAP path reaches it
GAP_SCALE = 3.0
BETA = 5.0


def build_soft_graph(cost):
    """Compile one graph with edge coefficients for quadratic speed-run cost."""
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
            edge(node, sink, (100 - j) * ETA, 0, 6)

    data = np.asarray(rows)
    arrays = tuple(np.ascontiguousarray(data[:, k], dtype=float if k in (2, 3) else np.int64) for k in range(7))
    return arrays, sink + 1


def longest_run(values):
    """Return the largest exact constant run in a landmark map."""
    changes = np.r_[True, np.diff(values) != 0, True]
    return int(np.diff(np.flatnonzero(changes)).max())


def longest_true_run(mask):
    """Count the longest contiguous sequence satisfying a move predicate."""
    padded = np.r_[False, mask, False]
    starts = np.flatnonzero(np.diff(padded.astype(int)) == 1)
    ends = np.flatnonzero(np.diff(padded.astype(int)) == -1)
    return int(np.max(ends - starts)) if len(starts) else 0


def evaluate(name, curve, map_curve, local, ru, ru_a, markers_a, markers_b, scale, gap_count, extra=None):
    """Score movement and markers after the path has been selected without labels."""
    mapped = interpolate_curve(curve, local[ru])
    valid = np.isfinite(mapped)
    displacement = np.abs(mapped[valid] - local[ru][valid])
    nearest = np.clip(np.rint(curve[80:] * 99).astype(int), 0, 99)
    late = curve[80:]
    steps = np.diff(map_curve)
    row = {
        "method": name,
        "late_position_80": float(curve[80]),
        "late_position_90": float(curve[90]),
        "late_position_99": float(curve[99]),
        "late_position_min": float(np.min(late)),
        "late_position_max": float(np.max(late)),
        "late_marker_mse": float(np.mean(((markers_a[80:] - markers_b[nearest]) / scale) ** 2)),
        "ru1303_mean_displacement": float(displacement.mean()),
        "ru1303_max_displacement": float(displacement.max()),
        "longest_map_plateau": longest_run(map_curve),
        "longest_late_map_plateau": longest_run(map_curve[80:]),
        "max_consecutive_repeat_moves": longest_true_run(np.isclose(steps, 0)),
        "max_consecutive_skip2_moves": longest_true_run(np.isclose(steps, 2 / 99)),
        "biological_patient_gaps_map": int(gap_count),
        "terminal_patient_skips_map": 0,
    }
    if extra:
        row.update(extra)
    return row


def main():
    """Run three fixed slope penalties and regenerate the requested comparisons."""
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

    rows = [evaluate("raw_identity", GRID, GRID, local, ru, ru_a, markers_a, markers_b, scale, 0)]
    previous = list(csv.DictReader((OUT.parent / "focused_followup" / "metrics.csv").open()))
    v2_beta1 = next(row for row in previous if row["method"] == "v2_beta_1")
    from scdebussy.align._dp import infer
    from scdebussy.align._model import LatticeModel

    v2 = infer(LatticeModel(cost, 1, ETA, ETA, LAMBDA, LAMBDA, mode="open"), backend="numba", count_distribution=False)
    mass = v2.a.sum(1)
    v2_curve = v2.a @ GRID / mass
    v2_map = np.full(100, np.nan)
    vg = LatticeModel(cost, 1, ETA, ETA, LAMBDA, LAMBDA, mode="open").graph
    e = v2.map_edges
    good = vg.match_i[e] >= 0
    v2_map[vg.match_i[e][good]] = GRID[vg.match_j[e][good]]
    matched_i = vg.match_i[e][good]
    terminal_patient = int(matched_i.min() + 99 - matched_i.max())
    internal_patient = int(vg.insertions[e].sum() - terminal_patient)
    rows.append(
        evaluate(
            "current_v2_posterior",
            v2_curve,
            v2_map,
            local,
            ru,
            ru_a,
            markers_a,
            markers_b,
            scale,
            internal_patient,
            {
                "terminal_patient_skips_map": terminal_patient,
                "late_mean_match_mass": float(mass[80:].mean()),
                "late_path_entropy": float(v2_beta1["full_path_entropy"]),
            },
        )
    )

    dtw_path = dtw(
        cost,
        step_pattern="asymmetric",
        open_begin=True,
        open_end=True,
        window_type="sakoechiba",
        window_args={"window_size": BAND},
    )
    dtw_curve = np.full(100, np.nan)
    for i in range(100):
        dtw_curve[i] = GRID[np.asarray(dtw_path.index2)[np.asarray(dtw_path.index1) == i]].mean()
    rows.append(
        evaluate("unconstrained_open_dtw_band60", dtw_curve, dtw_curve, local, ru, ru_a, markers_a, markers_b, scale, 0)
    )

    capped, capped_nodes = build_graph(cost, gap_scale=GAP_SCALE)
    capped_result = graph_infer(capped, capped_nodes, BETA)
    capped_mass = capped_result["matrix"].sum(1)
    capped_curve = capped_result["matrix"] @ GRID / capped_mass
    rows.append(
        evaluate(
            "capped_repeat4_posterior",
            capped_curve,
            capped_result["map"],
            local,
            ru,
            ru_a,
            markers_a,
            markers_b,
            scale,
            capped_result["map_counts"]["patient_gap"],
            {
                "late_mean_match_mass": float(capped_mass[80:].mean()),
                "late_path_entropy": capped_result["path_entropy"],
            },
        )
    )

    soft, soft_nodes = build_soft_graph(cost)
    src, dst, base, slope, kind, mi, mj = soft
    candidates = []
    for rho in RHO_VALUES:
        energy = np.ascontiguousarray(base + rho * slope)
        result = graph_infer((src, dst, energy, kind, mi, mj), soft_nodes, BETA)
        mass = result["matrix"].sum(1)
        curve = result["matrix"] @ GRID / mass
        name = f"run_penalty_rho_{rho:g}"
        row = evaluate(
            name,
            curve,
            result["map"],
            local,
            ru,
            ru_a,
            markers_a,
            markers_b,
            scale,
            result["map_counts"]["patient_gap"],
            {
                "rho": rho,
                "late_mean_match_mass": float(mass[80:].mean()),
                "late_path_entropy": result["path_entropy"],
                "warp_repeat_map": result["map_counts"]["repeat"],
                "warp_skip2_map": result["map_counts"]["skip2"],
            },
        )
        rows.append(row)
        candidates.append((row, curve, result["map"]))

    # Select on geometry and gaps only, never on labels or held-out markers.
    def selection_key(candidate):
        row, curve, _ = candidate
        late = curve[80:]
        outside = np.maximum(0.4 - late, 0) + np.maximum(late - 0.6, 0)
        return (row["biological_patient_gaps_map"] > 0, float(outside.mean()), row["longest_late_map_plateau"])

    selected_row, selected_curve, selected_map = min(candidates, key=selection_key)
    selected_row["selected_for_figure"] = True
    for row in rows:
        row.setdefault("selected_for_figure", False)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(6.7, 4.5), dpi=170)
    for label, curve, color, style in (
        ("raw identity", GRID, "gray", ":"),
        ("v2 posterior", v2_curve, "#2c7fb8", "-"),
        ("unconstrained DTW", dtw_curve, "#f03b20", "-"),
        ("capped posterior", capped_curve, "#756bb1", "-"),
        (f"regularized posterior ρ={selected_row['rho']}", selected_curve, "#238b45", "-"),
        ("regularized MAP", selected_map, "#238b45", "--"),
    ):
        ax.plot(GRID, curve, color=color, ls=style, lw=1.7, label=label)
    ax.axhspan(0.421, 0.531, color="red", alpha=0.1, label="RU581 SCLC-A q10–q90")
    ax.axhspan(0.709, 1, color="cyan", alpha=0.08, label="RU581 SCLC-N q10–q90")
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="RU1303 s_local",
        ylabel="matched RU581 position",
        title="Pairwise mapping with soft speed-run penalty",
    )
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "mapping_functions.png")
    plt.close(fig)

    cmap = LinearSegmentedColormap.from_list("types", [COLORS[t] for t in TYPES])
    fig, axes = plt.subplots(1, 5, figsize=(17, 2.6), dpi=170, sharey=True)
    panels = (
        ("Raw", GRID),
        ("v2 posterior", v2_curve),
        ("Unconstrained DTW", dtw_curve),
        ("Capped DTW", capped_curve),
        ("Run-regularized DTW", selected_curve),
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
    print(json.dumps({"selected": selected_row, "candidates": [r for r, _, _ in candidates]}, indent=2))


if __name__ == "__main__":
    main()
