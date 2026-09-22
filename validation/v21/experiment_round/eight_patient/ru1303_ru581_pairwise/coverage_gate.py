"""Expression-only trajectory-coverage gate on five frozen HTAN pairwise cases."""

import csv
import json
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from focused_followup import ETA
from matplotlib.colors import LinearSegmentedColormap
from run_pairwise import CACHE, COLORS, GRID, INPUT, MARKERS, ROOT, TYPES, interpolate_curve, marker_landmarks
from terminal_generalization import evaluate, pair_grid

OUT = Path(__file__).resolve().parent / "coverage_gate"
PATIENTS = ("RU1303", "RU581", "RU1083", "RU1444")
PAIRS = (
    ("RU1303", "RU581"),
    ("RU1303", "RU1083"),
    ("RU1303", "RU1444"),
    ("RU581", "RU1083"),
    ("RU581", "RU1303"),
)
PLOT_PAIRS = (("RU1303", "RU581"), ("RU581", "RU1303"))
TERMINAL_K = 10
COVERAGE_MARGIN = 0.10
CHEAP_PRICE = 0.10


def infer_coverage(x, names):
    """Median nearest position of each terminal expression centroid in peer trajectories."""
    result = {}
    for patient in PATIENTS:
        terminal = x[names.index(patient), -TERMINAL_K:].mean(axis=0)
        peer_positions = {}
        peer_costs = {}
        for peer in PATIENTS:
            if peer == patient:
                continue
            distances = np.mean((x[names.index(peer)] - terminal) ** 2, axis=1)
            index = int(np.argmin(distances))
            peer_positions[peer] = float(GRID[index])
            peer_costs[peer] = float(distances[index])
        result[patient] = {
            "coverage_score": float(np.median(list(peer_positions.values()))),
            "peer_nearest_positions": peer_positions,
            "peer_nearest_costs": peer_costs,
        }
    return result


def save_heatmap(source, reference, curves, local, patient, cell_type):
    """Compare fixed, endpoint-adaptive, and gated maps under prior plotting rules."""
    cmap = LinearSegmentedColormap.from_list("types", [COLORS[t] for t in TYPES])
    fig, axes = plt.subplots(1, 4, figsize=(14, 2.6), dpi=170, sharey=True)
    source_cells = patient == source
    source_a = source_cells & (cell_type == "SCLC-A")
    panels = (
        ("Raw", GRID),
        ("Fixed 0.10", curves["fixed_0.10"]),
        ("Endpoint adaptive", curves["endpoint_adaptive"]),
        ("Coverage gate", curves["coverage_gate"]),
    )
    for ax, (title, curve) in zip(axes, panels, strict=True):
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
    """Apply one fixed gate with no fitting to labels or held-out markers."""
    OUT.mkdir(exist_ok=True)
    fit = np.load(CACHE, allow_pickle=False)
    panel = json.loads(
        (ROOT / "validation/v21/experiment_round/eight_patient/feature_selection/top_250.json").read_text()
    )
    assert fit["genes"].tolist() == panel
    assert not set(MARKERS).intersection(panel)
    names = fit["patient_order"].tolist()
    x = fit["landmark_expression"].reshape(8, 100, 250)
    coverage = infer_coverage(x, names)
    with (OUT / "patient_coverage.json").open("w") as handle:
        json.dump(coverage, handle, indent=2)
        handle.write("\n")
    adata = ad.read_h5ad(INPUT, backed="r")
    assert np.array_equal(fit["cell_ids"], np.asarray(adata.obs_names, str))
    patient = adata.obs["patient"].astype(str).to_numpy()
    cell_type = adata.obs["cell_type_final2"].astype(str).to_numpy()
    local = adata.obs["s_local"].to_numpy(float)
    markers = {name: marker_landmarks(adata, name, MARKERS) for name in PATIENTS}
    with (Path(__file__).resolve().parent / "terminal_generalization/metrics.csv").open() as handle:
        previous = list(csv.DictReader(handle))
    fixed_by_pair = {(row["source"], row["reference"]): row for row in previous if row["setting"] == "terminal_0.10"}
    with (Path(__file__).resolve().parent / "terminal_pricing/metrics.csv").open() as handle:
        original_pair = list(csv.DictReader(handle))
    original_cheap = next(row for row in original_pair if np.isclose(float(row["tail_price_per_landmark"]), 0.10))
    with (Path(__file__).resolve().parent / "adaptive_terminal/metrics.csv").open() as handle:
        adaptive_by_pair = {(row["source"], row["reference"]): row for row in csv.DictReader(handle)}
    rows = []
    for source, reference in PAIRS:
        cost = np.mean((x[names.index(source)][:, None] - x[names.index(reference)][None]) ** 2, axis=2)
        source_markers, reference_markers = markers[source], markers[reference]
        scale = np.std(np.vstack([source_markers, reference_markers]), axis=0)
        scale = np.where(scale > 1e-8, scale, 1)
        difference = coverage[source]["coverage_score"] - coverage[reference]["coverage_score"]
        gate_on = difference <= -COVERAGE_MARGIN
        price = CHEAP_PRICE if gate_on else ETA
        gated, gated_curve = evaluate(
            source,
            reference,
            "coverage_gate",
            price,
            cost,
            local,
            patient,
            source_markers,
            reference_markers,
            scale,
        )
        fixed = fixed_by_pair.get((source, reference))
        adaptive = adaptive_by_pair[source, reference]
        comparison = {
            "query_coverage_score": coverage[source]["coverage_score"],
            "reference_coverage_score": coverage[reference]["coverage_score"],
            "coverage_difference_query_minus_reference": difference,
            "coverage_margin": COVERAGE_MARGIN,
            "cheap_terminal_gate_on": gate_on,
            "fixed_0.10_tail_skipped_fraction": (
                float(fixed["terminal_skipped_fraction"])
                if fixed
                else int(original_cheap["ru581_terminal_tail_length"]) / 100
            ),
            "fixed_0.10_posterior_full_marker_mse": (
                float(fixed["heldout_marker_mse_posterior_full"]) if fixed else None
            ),
            "adaptive_tail_skipped_fraction": float(adaptive["terminal_skipped_fraction"]),
            "adaptive_posterior_full_marker_mse": float(adaptive["heldout_marker_mse_posterior_full"]),
        }
        curves = {"coverage_gate": gated_curve}
        if (source, reference) in PLOT_PAIRS:
            if gate_on:
                curves["fixed_0.10"] = gated_curve
            else:
                fixed_row, curves["fixed_0.10"] = evaluate(
                    source,
                    reference,
                    "fixed_0.10",
                    CHEAP_PRICE,
                    cost,
                    local,
                    patient,
                    source_markers,
                    reference_markers,
                    scale,
                )
                comparison["fixed_0.10_posterior_full_marker_mse"] = fixed_row["heldout_marker_mse_posterior_full"]
            _adaptive_row, curves["endpoint_adaptive"] = evaluate(
                source,
                reference,
                "endpoint_adaptive",
                float(adaptive["terminal_price_per_landmark"]),
                cost,
                local,
                patient,
                source_markers,
                reference_markers,
                scale,
            )
            save_heatmap(source, reference, curves, local, patient, cell_type)
        if gate_on and (source, reference) == ("RU1303", "RU581"):
            comparison["fixed_0.10_posterior_full_marker_mse"] = gated["heldout_marker_mse_posterior_full"]
        rows.append({**gated, **comparison})
    with (OUT / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
