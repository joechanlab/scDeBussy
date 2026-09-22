"""Fixed-price pairwise HTAN checks for asymmetric reference terminal non-overlap."""

import csv
import json
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from focused_followup import ETA, graph_infer
from matplotlib.colors import LinearSegmentedColormap
from run_pairwise import CACHE, COLORS, GRID, INPUT, MARKERS, ROOT, TYPES, interpolate_curve, marker_landmarks
from slope_regularization import BETA, build_soft_graph
from terminal_pricing import RHO, trace_and_decompose

OUT = Path(__file__).resolve().parent / "terminal_generalization"
PAIRS = (
    ("RU1303", "RU1083"),
    ("RU1303", "RU1444"),
    ("RU581", "RU1083"),
    ("RU831", "RU581"),
    ("RU581", "RU1303"),
)
PRICES = (("previous_current", ETA), ("terminal_0.10", 0.10))


def pair_grid(position, patient, cell_type, source, reference):
    """Apply the previous 100-bin/three-cell gray convention in pair-specific row order."""
    grid = np.full((2, 100), np.nan)
    for row, name in enumerate((source, reference)):
        keep = patient == name
        pos = position[keep]
        typ = cell_type[keep]
        finite = np.isfinite(pos)
        bins = np.clip(np.digitize(pos[finite], np.linspace(0, 1, 101)) - 1, 0, 99)
        encoded = np.array([TYPES.index(value) / 2 if value in TYPES else np.nan for value in typ[finite]])
        for b in range(100):
            values = encoded[bins == b]
            values = values[np.isfinite(values)]
            if len(values) >= 3:
                grid[row, b] = values.mean()
    return grid


def marker_mse(curve, source_markers, reference_markers, scale, start=0):
    """Score held-out marker correspondence where the supplied curve is finite."""
    indices = np.arange(start, 100)
    valid = np.isfinite(curve[indices])
    if not valid.any():
        return float("nan")
    nearest = np.clip(np.rint(curve[indices[valid]] * 99).astype(int), 0, 99)
    return float(np.mean(((source_markers[indices[valid]] - reference_markers[nearest]) / scale) ** 2))


def longest_matched_plateau(map_index):
    """Count equal matched positions, breaking runs at biological gaps."""
    best = run = 0
    previous = -1
    for value in map_index:
        if value < 0:
            run = 0
            previous = -1
        else:
            run = run + 1 if value == previous else 1
            best = max(best, run)
            previous = value
    return best


def evaluate(source, reference, label, price, cost, local, patient, source_markers, reference_markers, scale):
    """Evaluate one unchanged graph at the specified terminal-tail price."""
    graph, n_nodes = build_soft_graph(cost, tail_eta=price)
    src, dst, base, slope, kind, mi, mj = graph
    energy = np.ascontiguousarray(base + RHO * slope)
    inferred = graph_infer((src, dst, energy, kind, mi, mj), n_nodes, BETA)
    matrix = inferred["matrix"]
    mass = matrix.sum(1)
    curve = np.divide(matrix @ GRID, mass, out=np.full(100, np.nan), where=mass > 1e-12)
    decomposition = trace_and_decompose(graph, n_nodes, cost, RHO)
    map_index = decomposition.pop("map_index")
    decomposition["longest_map_plateau"] = longest_matched_plateau(map_index)
    decomposition["longest_late_map_plateau"] = longest_matched_plateau(map_index[80:])
    map_curve = inferred["map"]
    np.testing.assert_allclose(decomposition["total_map_cost"], inferred["map_cost"])
    np.testing.assert_allclose(
        decomposition["ru581_terminal_tail_cost"], decomposition["ru581_terminal_tail_length"] * price
    )
    conditional = np.divide(matrix, mass[:, None], out=np.zeros_like(matrix), where=mass[:, None] > 1e-12)
    entropy = -np.sum(conditional * np.log(np.maximum(conditional, 1e-300)), axis=1)
    shared = np.isfinite(map_curve) & np.isfinite(curve)
    source_cells = patient == source
    mapped_cells = interpolate_curve(curve, local[source_cells])
    valid_cells = np.isfinite(mapped_cells)
    displacement = np.abs(mapped_cells[valid_cells] - local[source_cells][valid_cells])
    matched = map_index >= 0
    row = {
        "source": source,
        "reference": reference,
        "setting": label,
        "terminal_price_per_landmark": price,
        "matched_reference_start": int(map_index[matched].min()),
        "matched_reference_end": int(map_index[matched].max()),
        "matched_reference_span_fraction": float((map_index[matched].max() - map_index[matched].min() + 1) / 100),
        "terminal_skipped_fraction": float(decomposition["ru581_terminal_tail_length"] / 100),
        "mean_source_cell_displacement": float(displacement.mean()),
        "max_source_cell_displacement": float(displacement.max()),
        "map_posterior_mean_abs_difference": float(np.mean(np.abs(map_curve[shared] - curve[shared]))),
        "map_posterior_late_mean_abs_difference": float(
            np.mean(np.abs(map_curve[80:][shared[80:]] - curve[80:][shared[80:]]))
        ),
        "posterior_mean_row_entropy_nats": float(entropy.mean()),
        "posterior_late_mean_row_entropy_nats": float(entropy[80:].mean()),
        "posterior_mean_match_mass": float(mass.mean()),
        "posterior_late_mean_match_mass": float(mass[80:].mean()),
        "posterior_path_entropy_nats": float(inferred["path_entropy"]),
        "heldout_marker_mse_map_full": marker_mse(map_curve, source_markers, reference_markers, scale),
        "heldout_marker_mse_posterior_full": marker_mse(curve, source_markers, reference_markers, scale),
        "heldout_marker_mse_map_late": marker_mse(map_curve, source_markers, reference_markers, scale, 80),
        "heldout_marker_mse_posterior_late": marker_mse(curve, source_markers, reference_markers, scale, 80),
        "late_map_80": float(map_curve[80]),
        "late_map_90": float(map_curve[90]),
        "late_map_99": float(map_curve[99]),
        "late_posterior_80": float(curve[80]),
        "late_posterior_90": float(curve[90]),
        "late_posterior_99": float(curve[99]),
        **decomposition,
    }
    return row, curve


def save_heatmap(source, reference, curves, local, patient, cell_type):
    """Plot raw, current-price, and cheap-tail results with identical axes and rules."""
    cmap = LinearSegmentedColormap.from_list("types", [COLORS[t] for t in TYPES])
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.6), dpi=170, sharey=True)
    source_cells = patient == source
    source_a = source_cells & (cell_type == "SCLC-A")
    for ax, (title, curve) in zip(
        axes,
        (("Raw", GRID), ("Previous/current", curves["previous_current"]), ("Terminal 0.10", curves["terminal_0.10"])),
        strict=True,
    ):
        position = local.copy()
        position[source_cells] = interpolate_curve(curve, local[source_cells])
        grid = pair_grid(position, patient, cell_type, source, reference)
        ax.imshow(
            np.ma.masked_invalid(grid),
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=1,
            extent=[0, 1, 1.5, -0.5],
        )
        ax.set(
            yticks=[0, 1], yticklabels=[source, reference], xlim=(0, 1), title=title, xlabel="reference position [0,1]"
        )
        ax.set_facecolor("#e8e8e8")
        if source_a.any():
            ax.axvline(np.nanmedian(position[source_a]), color="black", ls="--", lw=0.8)
    fig.legend(
        handles=[mpatches.Patch(color=COLORS[t], label=t) for t in TYPES]
        + [mpatches.Patch(color="#e8e8e8", label="empty / <3 finite cells")],
        ncol=4,
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(OUT / f"{source}_to_{reference}_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def main():
    """Run the prespecified four checks and inexpensive reverse direction."""
    OUT.mkdir(exist_ok=True)
    fit = np.load(CACHE, allow_pickle=False)
    panel = json.loads(
        (ROOT / "validation/v21/experiment_round/eight_patient/feature_selection/top_250.json").read_text()
    )
    assert fit["genes"].tolist() == panel
    assert not set(MARKERS).intersection(panel)
    names = fit["patient_order"].tolist()
    x = fit["landmark_expression"].reshape(8, 100, 250)
    adata = ad.read_h5ad(INPUT, backed="r")
    assert np.array_equal(fit["cell_ids"], np.asarray(adata.obs_names, str))
    patient = adata.obs["patient"].astype(str).to_numpy()
    cell_type = adata.obs["cell_type_final2"].astype(str).to_numpy()
    local = adata.obs["s_local"].to_numpy(float)
    marker_cache = {name: marker_landmarks(adata, name, MARKERS) for pair in PAIRS for name in pair}
    rows = []
    for source, reference in PAIRS:
        cost = np.mean((x[names.index(source)][:, None] - x[names.index(reference)][None]) ** 2, axis=2)
        source_markers, reference_markers = marker_cache[source], marker_cache[reference]
        scale = np.std(np.vstack([source_markers, reference_markers]), axis=0)
        scale = np.where(scale > 1e-8, scale, 1)
        curves = {}
        for label, price in PRICES:
            row, curve = evaluate(
                source, reference, label, price, cost, local, patient, source_markers, reference_markers, scale
            )
            rows.append(row)
            curves[label] = curve
        save_heatmap(source, reference, curves, local, patient, cell_type)
    with (OUT / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
