"""One fixed endpoint-compatibility rule on the five prespecified HTAN pairs."""

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

OUT = Path(__file__).resolve().parent / "adaptive_terminal"
K = 10
TAU = 0.25
PLOT_PAIRS = (("RU1303", "RU581"), ("RU581", "RU1303"))
TEST_PAIRS = (
    ("RU1303", "RU581"),
    ("RU1303", "RU1083"),
    ("RU1303", "RU1444"),
    ("RU581", "RU1083"),
    ("RU581", "RU1303"),
)


def prior_endpoints():
    """Read the already-computed fixed-0.10 endpoints; do not refit them."""
    base = Path(__file__).resolve().parent
    with (base / "terminal_generalization/metrics.csv").open() as handle:
        previous = list(csv.DictReader(handle))
    endpoints = {
        (row["source"], row["reference"]): int(row["matched_reference_end"])
        for row in previous
        if row["setting"] == "terminal_0.10"
    }
    with (base / "terminal_pricing/metrics.csv").open() as handle:
        original = list(csv.DictReader(handle))
    cheap = next(row for row in original if np.isclose(float(row["tail_price_per_landmark"]), 0.10))
    endpoints["RU1303", "RU581"] = 99 - int(cheap["ru581_terminal_tail_length"])
    return endpoints


def compatibility(cost, endpoint):
    """Compare the query's last K landmarks to retained-window and skipped-suffix costs."""
    retained = float(cost[-K:, endpoint - K + 1 : endpoint + 1].mean())
    suffix = float(cost[-K:, endpoint + 1 :].mean())
    return retained, suffix, retained - suffix


def adaptive_price(delta):
    """Map cost difference to a terminal price capped by the previous price."""
    return float(ETA / (1 + np.exp(-delta / TAU)))


def save_heatmap(source, reference, curves, local, patient, cell_type):
    """Use the established row order, colors, 100 bins, axes, and gray rule."""
    cmap = LinearSegmentedColormap.from_list("types", [COLORS[t] for t in TYPES])
    fig, axes = plt.subplots(1, 4, figsize=(14, 2.6), dpi=170, sharey=True)
    source_cells = patient == source
    source_a = source_cells & (cell_type == "SCLC-A")
    panels = (
        ("Raw", GRID),
        ("Previous/current", curves["previous_current"]),
        ("Fixed 0.10", curves["fixed_0.10"]),
        ("Adaptive", curves["adaptive"]),
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


def save_scatter(rows):
    """Show the chosen price and resulting truncation against the compatibility score."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=170, sharex=True)
    for index, row in enumerate(rows):
        delta = row["delta_suffix"]
        label = f"{row['source']}→{row['reference']}"
        color = plt.get_cmap("tab10")(index)
        axes[0].scatter(delta, row["terminal_price_per_landmark"], s=45, color=color, label=label)
        axes[1].scatter(delta, row["terminal_skipped_fraction"], s=45, color=color)
    delta_axis = np.linspace(
        min(row["delta_suffix"] for row in rows) - 0.03, max(row["delta_suffix"] for row in rows) + 0.03, 200
    )
    axes[0].plot(delta_axis, ETA / (1 + np.exp(-delta_axis / TAU)), color="gray", lw=1)
    axes[0].set(ylabel="adaptive terminal cost / landmark", ylim=(0, ETA), xlabel="Δ suffix")
    axes[1].set(ylabel="reference terminal skipped fraction", ylim=(0, 0.5), xlabel="Δ suffix")
    fig.legend(loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    fig.savefig(OUT / "compatibility_vs_terminal_behavior.png", bbox_inches="tight")
    plt.close(fig)


def main():
    """Apply one predeclared rule without fitting or pair-specific tuning."""
    OUT.mkdir(exist_ok=True)
    endpoints = prior_endpoints()
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
    markers = {name: marker_landmarks(adata, name, MARKERS) for pair in TEST_PAIRS for name in pair}
    rows = []
    for source, reference in TEST_PAIRS:
        cost = np.mean((x[names.index(source)][:, None] - x[names.index(reference)][None]) ** 2, axis=2)
        source_markers, reference_markers = markers[source], markers[reference]
        scale = np.std(np.vstack([source_markers, reference_markers]), axis=0)
        scale = np.where(scale > 1e-8, scale, 1)
        endpoint = endpoints[source, reference]
        retained, suffix, delta = compatibility(cost, endpoint)
        price = adaptive_price(delta)
        adaptive_row, adaptive_curve = evaluate(
            source, reference, "adaptive", price, cost, local, patient, source_markers, reference_markers, scale
        )
        adaptive_row.update(
            {
                "candidate_reference_endpoint_from_fixed_0.10": endpoint,
                "final_K": K,
                "score_retained_expression_cost": retained,
                "score_skipped_suffix_expression_cost": suffix,
                "delta_suffix": delta,
                "sigmoid_scale_tau": TAU,
            }
        )
        rows.append(adaptive_row)
        if (source, reference) in PLOT_PAIRS:
            curves = {"adaptive": adaptive_curve}
            for label, other_price in (("previous_current", ETA), ("fixed_0.10", 0.10)):
                _row, curves[label] = evaluate(
                    source,
                    reference,
                    label,
                    other_price,
                    cost,
                    local,
                    patient,
                    source_markers,
                    reference_markers,
                    scale,
                )
            save_heatmap(source, reference, curves, local, patient, cell_type)
    with (OUT / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    save_scatter(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
