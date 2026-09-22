"""Rerun the RU831 pairwise diagnostic before vs after the landmark_matchable fix.

Before = matchable=None (identical to the original ru831_pairwise.py run: RU831's
20 unmatchable landmarks still contribute ordinary match edges via their raw,
unreliable landmark_expression). After = matchable propagated into
compile_graph, so those same landmarks can only be crossed via gap edges. Same
frozen trajectory-250 genes, unit precision, warp/gap mechanics, BETA/RHO, and
full terminal price (ETA, matching the gate-ablation gate-off state) as before;
no parameter retuned -- only the matchable fix from three_patient_joint.py is
new.
"""

import json
import sys
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ru1303_ru581_pairwise"))

from ru831_pairwise import (  # noqa: E402
    PATIENT,
    REFERENCES,
    gap_ranges,
    heatmap_grid,
    infer_traced,
    longest_run,
    plateau_range,
)
from run_pairwise import CACHE, COLORS, GRID, INPUT, MARKERS, ROOT, TYPES, interpolate_curve, marker_landmarks  # noqa: E402
from three_patient_joint import compile_graph, match_cost  # noqa: E402
from focused_followup import ETA  # noqa: E402

OUT = Path(__file__).resolve().parent


def fit(cost, matchable):
    compiled, nodes = compile_graph(cost, ETA, matchable=matchable)
    return infer_traced(compiled, nodes, cost)


def summarize(result, markers_a, markers_b, scale):
    gap_pos = result["gap_landmark_positions"]
    ranges = gap_ranges(gap_pos)
    map_idx = result["map_index"]
    matched = np.flatnonzero(map_idx >= 0)
    matched_span = [int(matched.min()), int(matched.max())] if len(matched) else None
    valid_map = np.isfinite(result["map_curve"])
    nearest = np.clip(np.rint(result["map_curve"][valid_map] * 99).astype(int), 0, 99)
    heldout_marker_mse = float(np.mean(((markers_a[valid_map] - markers_b[nearest]) / scale) ** 2))
    return {
        "internal_patient_gaps": result["internal_patient_gaps"],
        "internal_reference_gaps": result["internal_reference_gaps"],
        "gap_landmark_ranges": ranges,
        "longest_map_plateau_landmarks": longest_run(result["map_curve"]),
        "matched_span": matched_span,
        "matched_landmark_count": int(len(matched)),
        "map_posterior_mean_abs_difference": result["map_posterior_mean_abs_difference"],
        "posterior_mean_match_mass": result["posterior_mean_match_mass"],
        "tail_skipped_fraction": result["tail_skipped_fraction"],
        "heldout_marker_mse": heldout_marker_mse,
    }


def main():
    fit_cache = np.load(CACHE, allow_pickle=False)
    panel = json.loads(
        (ROOT / "validation/v21/experiment_round/eight_patient/feature_selection/top_250.json").read_text()
    )
    assert fit_cache["genes"].tolist() == panel
    names = fit_cache["patient_order"].tolist()
    x = fit_cache["landmark_expression"].reshape(8, 100, 250)
    matchable_all = fit_cache["landmark_matchable"].reshape(8, 100)
    ru831 = x[names.index(PATIENT)]
    ru831_matchable = matchable_all[names.index(PATIENT)]
    print(f"RU831 matchable landmarks: {ru831_matchable.sum()}/100")
    print(f"unmatchable landmark indices: {np.flatnonzero(~ru831_matchable).tolist()}")

    adata = ad.read_h5ad(INPUT, backed="r")
    assert np.array_equal(fit_cache["cell_ids"], np.asarray(adata.obs_names, str))
    patients_obs = adata.obs["patient"].astype(str).to_numpy()
    labels_obs = adata.obs["cell_type_final2"].astype(str).to_numpy()
    local = adata.obs["s_local"].to_numpy(float)

    all_summaries = {}
    for reference in REFERENCES:
        print(f"=== RU831 -> {reference} ===")
        ref_expr = x[names.index(reference)]
        cost = match_cost(ru831, ref_expr)

        before = fit(cost, matchable=None)
        after = fit(cost, matchable=ru831_matchable)

        markers_a = marker_landmarks(adata, PATIENT, MARKERS)
        markers_b = marker_landmarks(adata, reference, MARKERS)
        scale = np.std(np.vstack([markers_a, markers_b]), axis=0)
        scale = np.where(scale > 1e-8, scale, 1)

        summary = {
            "before_matchability_fix": summarize(before, markers_a, markers_b, scale),
            "after_matchability_fix": summarize(after, markers_a, markers_b, scale),
        }
        all_summaries[reference] = summary
        print(json.dumps(summary, indent=2))

        if reference == "RU581":
            fig, ax = plt.subplots(figsize=(6.7, 4.4), dpi=170)
            ax.plot(GRID, GRID, color="gray", lw=1, ls=":", label="identity")
            ax.plot(GRID, before["curve"], color="#f03b20", lw=1.6, ls="-", label="before fix: posterior mean")
            ax.plot(GRID, before["map_curve"], color="#f03b20", lw=1, ls="--", label="before fix: MAP")
            ax.plot(GRID, after["curve"], color="#2c7fb8", lw=2, label="after fix: posterior mean")
            ax.plot(GRID, after["map_curve"], color="#2c7fb8", lw=1, ls="--", label="after fix: MAP")
            unmatchable_idx = np.flatnonzero(~ru831_matchable)
            for i in unmatchable_idx:
                ax.axvspan(GRID[i], GRID[min(i + 1, 99)], color="yellow", alpha=0.12, lw=0)
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="RU831 s_local", ylabel="mapped RU581 position",
                   title="RU831→RU581 mapping: before vs after matchability fix")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(OUT / "mapping_functions_after_fix.png")
            fig.savefig(OUT / "mapping_functions_after_fix.pdf")
            plt.close(fig)

            order = (PATIENT, reference)
            mapped_before = local.copy()
            mapped_after = local.copy()
            ru_mask = patients_obs == PATIENT
            mapped_before[ru_mask] = interpolate_curve(before["curve"], local[ru_mask])
            mapped_after[ru_mask] = interpolate_curve(after["curve"], local[ru_mask])
            cmap = LinearSegmentedColormap.from_list("types", [COLORS[t] for t in TYPES])
            fig, axes = plt.subplots(1, 3, figsize=(13, 2.7), dpi=170, sharey=True)
            for ax, (title, position) in zip(
                axes,
                (
                    ("Raw", local),
                    ("Before matchability fix", mapped_before),
                    ("After matchability fix", mapped_after),
                ),
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
            fig.savefig(OUT / "comparison_heatmap_before_after_fix.png", bbox_inches="tight")
            plt.close(fig)

    (OUT / "ru831_matchability_fix_rerun.json").write_text(json.dumps(all_summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
