"""Focused RU1303-to-RU581 path diagnostic using frozen HTAN landmarks."""

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
from matplotlib.colors import LinearSegmentedColormap

from scdebussy.align._dp import infer
from scdebussy.align._model import LatticeModel

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
INPUT = Path("/scratch/chanj3/wangm10/HTAN/results/aligned_scdebussy_preprocessed_v2_ru942_deg_all.h5ad")
CACHE = Path(
    "/data1/chanj3/wangm10/scDeBussy-v2/.htan_experiment_cache/exp-htan-eight-patient/eight_patient/results/trajectory_250_unit.npz"
)
MARKERS = ("ASCL1", "NEUROD1", "INSM1", "CHGA", "CHGB", "NCAM1", "DLL3")
PATIENTS = ("RU1303", "RU581")
TYPES = ("LUAD", "SCLC-A", "SCLC-N")
COLORS = {"LUAD": "gold", "SCLC-A": "tab:red", "SCLC-N": "tab:cyan"}
GRID = np.linspace(0, 1, 100)


def path_summary(pairs, cost, method, native_cost, gap_patient=0, gap_ref=0):
    """Summarize matched spans, transition counts, and path costs."""
    ii, jj = np.asarray(pairs, int).T
    di, dj = np.diff(ii), np.diff(jj)
    terminal_patient = int(ii.min() + 99 - ii.max())
    terminal_ref = int(jj.min() + 99 - jj.max())
    return {
        "method": method,
        "native_total_cost": float(native_cost),
        "matched_expression_cost": float(cost[ii, jj].sum()),
        "matched_span_ru1303": [int(ii.min()), int(ii.max())],
        "matched_span_ru581": [int(jj.min()), int(jj.max())],
        "ru1303_unmatched_begin_fraction": float(ii.min() / 100),
        "ru1303_unmatched_end_fraction": float((99 - ii.max()) / 100),
        "ru581_unmatched_begin_fraction": float(jj.min() / 100),
        "ru581_unmatched_end_fraction": float((99 - jj.max()) / 100),
        "warp_stretch_moves": int(np.sum((di == 1) & (dj == 0))),
        "warp_compression_moves": int(np.sum((di == 1) & (dj == 2))),
        "ordinary_11_moves": int(np.sum((di == 1) & (dj == 1))),
        "internal_patient_gap_skips": int(max(0, gap_patient - terminal_patient)),
        "internal_ru581_gap_skips": int(max(0, gap_ref - terminal_ref)),
        "terminal_patient_skips": terminal_patient,
        "terminal_ru581_skips": terminal_ref,
    }


def interpolate_curve(curve, local):
    """Interpolate a landmark mapping at observed local pseudotimes."""
    good = np.isfinite(curve)
    return np.interp(local, GRID[good], curve[good], left=np.nan, right=np.nan)


def marker_landmarks(adata, patient, names):
    """Smooth evaluation-only marker expression on the frozen landmark grid."""
    obs = adata.obs
    mask = obs["patient"].astype(str).to_numpy() == patient
    local = obs.loc[mask, "s_local"].to_numpy(float)
    values = adata[mask, list(names)].X
    values = np.asarray(values.toarray() if hasattr(values, "toarray") else values, float)
    weight = np.exp(-0.5 * ((local[:, None] - GRID[None, :]) / 0.05) ** 2)
    weight[np.abs(local[:, None] - GRID[None, :]) > 0.1] = 0
    return weight.T @ values / np.maximum(weight.sum(axis=0)[:, None], 1e-12)


def label_grid(local, labels):
    """Build the original 100-bin, three-cell minimum heatmap grid."""
    out = np.full((2, 100), np.nan)
    for row, patient in enumerate(PATIENTS):
        mask = labels["patient"] == patient
        pos = local[mask]
        typ = labels["type"][mask]
        finite = np.isfinite(pos)
        bins = np.clip(np.digitize(pos[finite], np.linspace(0, 1, 101)) - 1, 0, 99)
        encoded = np.array([TYPES.index(x) / 2 if x in TYPES else np.nan for x in typ[finite]])
        for b in range(100):
            vals = encoded[bins == b]
            vals = vals[np.isfinite(vals)]
            if len(vals) >= 3:
                out[row, b] = vals.mean()
    return out


def main():
    """Run the focused pairwise comparison and write its artifacts."""
    OUT.mkdir(parents=True, exist_ok=True)
    fit = np.load(CACHE, allow_pickle=False)
    panel = json.loads(
        (ROOT / "validation/v21/experiment_round/eight_patient/feature_selection/top_250.json").read_text()
    )
    assert fit["genes"].tolist() == panel
    assert not set(MARKERS).intersection(panel)
    names = fit["patient_order"].tolist()
    x = fit["landmark_expression"].reshape(8, 100, 250)
    matchable = fit["landmark_matchable"].reshape(8, 100)
    a, b = x[names.index("RU1303")], x[names.index("RU581")]
    assert matchable[names.index("RU1303")].all() and matchable[names.index("RU581")].all()
    cost = np.mean((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
    cfg = json.loads(str(fit["resolved_fit_config"]))
    eta = cfg["eta_i"]
    model = LatticeModel(cost, 1.0, eta, eta, 3 * eta, 3 * eta, mode="open")
    posterior = infer(model, backend="numba")
    graph = model.graph
    edges = posterior.map_edges
    valid = graph.match_i[edges] >= 0
    v2pairs = np.column_stack([graph.match_i[edges][valid], graph.match_j[edges][valid]])
    v2map = np.full(100, np.nan)
    mass = posterior.a.sum(1)
    v2map[mass > 1e-10] = (posterior.a @ GRID)[mass > 1e-10] / mass[mass > 1e-10]
    v2map_supported = v2map.copy()
    v2map_supported[mass < 0.1] = np.nan
    v2map_map = np.interp(GRID, GRID[v2pairs[:, 0]], GRID[v2pairs[:, 1]], left=np.nan, right=np.nan)
    rawpairs = np.column_stack([np.arange(100), np.arange(100)])
    paths = {"raw": rawpairs, "v2": v2pairs}
    curves = {"raw": GRID, "v2": v2map_supported}
    summaries = {
        "raw": path_summary(rawpairs, cost, "raw_s_local", np.trace(cost)),
        "v2": path_summary(
            v2pairs,
            cost,
            "v2_pairwise_MAP",
            posterior.map_energy,
            graph.insertions[edges].sum(),
            graph.deletions[edges].sum(),
        ),
    }
    summaries["v2"]["posterior_free_energy"] = float(-posterior.log_z)
    summaries["v2"]["expected_patient_gap_count"] = float(posterior.expected_insertions)
    summaries["v2"]["expected_reference_gap_count"] = float(posterior.expected_deletions)
    summaries["v2"]["posterior_support_late_80_99"] = [float(v) for v in mass[80:]]
    for width in (20, 60):
        path = dtw(
            cost,
            step_pattern="asymmetric",
            open_begin=True,
            open_end=True,
            window_type="sakoechiba",
            window_args={"window_size": width},
            keep_internals=True,
        )
        pairs = np.column_stack([path.index1, path.index2])
        key = f"dtw_band_{width}"
        paths[key] = pairs
        curve = np.full(100, np.nan)
        for i in range(100):
            curve[i] = GRID[pairs[pairs[:, 0] == i, 1]].mean()
        curves[key] = curve
        summaries[key] = path_summary(pairs, cost, key, path.distance)
        summaries[key]["native_normalized_cost"] = float(path.normalizedDistance)
        summaries[key]["reachability_note"] = (
            "original 0.2 band" if width == 20 else "single widened-band reachability check"
        )
    adata = ad.read_h5ad(INPUT, backed="r")
    assert np.array_equal(fit["cell_ids"], np.asarray(adata.obs_names, str))
    obs = adata.obs
    patient = obs["patient"].astype(str).to_numpy()
    typ = obs["cell_type_final2"].astype(str).to_numpy()
    local = obs["s_local"].to_numpy(float)
    labels = {"patient": patient, "type": typ}
    ru = patient == "RU1303"
    ru_a = ru & (typ == "SCLC-A")
    ru581_n = (patient == "RU581") & (typ == "SCLC-N")
    ru581_a = (patient == "RU581") & (typ == "SCLC-A")
    nlo, nhi = np.quantile(local[ru581_n], [0.1, 0.9])
    markers_a = marker_landmarks(adata, "RU1303", MARKERS)
    markers_b = marker_landmarks(adata, "RU581", MARKERS)
    scale = np.std(np.vstack([markers_a, markers_b]), axis=0)
    scale = np.where(scale > 1e-8, scale, 1)
    landmark_rows = []
    for i in range(100):
        row = {
            "ru1303_landmark": i,
            "ru1303_s_local": GRID[i],
            "v2_posterior_match_mass": mass[i],
            "v2_posterior_mean_ru581": v2map[i],
            "v2_supported_ru581": v2map_supported[i],
            "v2_map_ru581": v2map_map[i],
        }
        for width in (20, 60):
            row[f"dtw_band_{width}_ru581"] = curves[f"dtw_band_{width}"][i]
        row["ru581_pairwise_cost_min_index"] = int(np.argmin(cost[i]))
        row["ru581_pairwise_cost_min"] = float(np.min(cost[i]))
        landmark_rows.append(row)
    with (OUT / "landmark_matches.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=landmark_rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(landmark_rows)
    for key, curve in curves.items():
        mapped = interpolate_curve(curve, local[ru])
        valid = np.isfinite(mapped)
        disp = np.abs(mapped[valid] - local[ru][valid])
        a_mapped = interpolate_curve(curve, local[ru_a])
        a_mapped = a_mapped[np.isfinite(a_mapped)]
        summaries[key]["ru1303_cell_mapped_fraction"] = float(valid.mean())
        summaries[key]["mean_cell_displacement"] = float(disp.mean()) if len(disp) else None
        summaries[key]["max_cell_displacement"] = float(disp.max()) if len(disp) else None
        summaries[key]["ru1303_sclc_a_overlap_ru581_sclc_n_q10_q90"] = (
            float(np.mean((a_mapped >= nlo) & (a_mapped <= nhi))) if len(a_mapped) else None
        )
        summaries[key]["ru1303_sclc_a_median_on_ru581"] = float(np.median(a_mapped)) if len(a_mapped) else None
        summaries[key]["late_80_99_ru581_range"] = [float(np.nanmin(curve[80:])), float(np.nanmax(curve[80:]))]
        changes = np.r_[True, np.diff(curve) != 0, True]
        summaries[key]["longest_exact_plateau_landmarks"] = int(np.diff(np.flatnonzero(changes)).max())
        nearest = np.clip(np.rint(curve[80:] * 99).astype(int), 0, 99)
        summaries[key]["heldout_marker_late_standardized_mse"] = float(
            np.mean(((markers_a[80:] - markers_b[nearest]) / scale) ** 2)
        )
    summaries["reference"] = {
        "ru581_sclc_a_q10_median_q90": [float(v) for v in np.quantile(local[ru581_a], [0.1, 0.5, 0.9])],
        "ru581_sclc_n_q10_median_q90": [float(v) for v in np.quantile(local[ru581_n], [0.1, 0.5, 0.9])],
        "ru1303_sclc_a_q10_median_q90": [float(v) for v in np.quantile(local[ru_a], [0.1, 0.5, 0.9])],
        "heldout_markers": MARKERS,
        "cost": "mean squared expression difference over frozen trajectory-250 genes, unit precision",
        "v2_parameters": {k: cfg[k] for k in ("beta", "eta_i", "eta_d", "lambda_i", "lambda_d", "mode")},
    }
    (OUT / "pairwise_metrics.json").write_text(json.dumps(summaries, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(6.3, 5.7), dpi=170)
    image = ax.imshow(cost.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="magma_r")
    ax.plot(GRID, GRID, color="white", lw=1.2, ls=":", label="identity")
    ax.plot(GRID[v2pairs[:, 0]], GRID[v2pairs[:, 1]], color="#2c7fb8", lw=2, label="v2 MAP")
    for width, color, style in ((20, "#f03b20", "-"), (60, "#31a354", "--")):
        pp = paths[f"dtw_band_{width}"]
        ax.plot(GRID[pp[:, 0]], GRID[pp[:, 1]], color=color, lw=1.6, ls=style, label=f"open DTW band {width}")
    ax.axhspan(nlo, nhi, color="cyan", alpha=0.12, label="RU581 SCLC-N q10–q90")
    ax.axhspan(*np.quantile(local[ru581_a], [0.1, 0.9]), color="red", alpha=0.08, label="RU581 SCLC-A q10–q90")
    ax.set(
        xlabel="RU1303 local pseudotime",
        ylabel="RU581 local pseudotime",
        title="Pairwise trajectory-250 unit-precision cost",
    )
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.colorbar(image, ax=ax, label="mean squared expression difference")
    fig.tight_layout()
    fig.savefig(OUT / "cost_matrix_paths.png")
    fig.savefig(OUT / "cost_matrix_paths.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.3, 4.2), dpi=170)
    ax.plot(GRID, GRID, color="gray", lw=1, ls=":", label="identity")
    ax.plot(GRID, v2map_supported, color="#2c7fb8", lw=2, label="v2 posterior mean (mass ≥0.1)")
    ax.plot(GRID, v2map_map, color="#2c7fb8", lw=1, ls="--", label="v2 MAP")
    ax.plot(GRID, curves["dtw_band_20"], color="#f03b20", lw=2, label="open DTW band 20")
    ax.plot(GRID, curves["dtw_band_60"], color="#31a354", lw=1.6, ls="--", label="open DTW band 60")
    ax.axhspan(nlo, nhi, color="cyan", alpha=0.12)
    ax.axhspan(*np.quantile(local[ru581_a], [0.1, 0.9]), color="red", alpha=0.08)
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="RU1303 s_local",
        ylabel="mapped RU581 position",
        title="Pairwise mapping; shaded: RU581 SCLC-A / SCLC-N ranges",
    )
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "mapping_functions.png")
    fig.savefig(OUT / "mapping_functions.pdf")
    plt.close(fig)

    cmap = LinearSegmentedColormap.from_list("types", [COLORS[t] for t in TYPES])
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 2.7), dpi=170, sharey=True)
    for ax, (key, title) in zip(
        axes,
        (
            ("raw", "Unaligned"),
            ("v2", "Current v2 pairwise posterior"),
            ("dtw_band_20", "Open-begin/open-end DTW, original band 20"),
        ),
        strict=True,
    ):
        mapped = local.copy()
        mapped[ru] = interpolate_curve(curves[key], local[ru])
        grid = label_grid(mapped, labels)
        ax.imshow(
            np.ma.masked_invalid(grid),
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=1,
            extent=[0, 1, 1.5, -0.5],
        )
        ax.set(yticks=[0, 1], yticklabels=PATIENTS, xlim=(0, 1), title=title, xlabel="RU581 reference position [0,1]")
        ax.set_facecolor("#e8e8e8")
        ru_median = np.nanmedian(mapped[ru_a])
        ax.axvline(ru_median, color="black", ls="--", lw=0.8)
        ax.text(ru_median, -0.38, f"A {ru_median:.2f}", fontsize=7, ha="left", va="center")
        for group, tag, color in ((ru581_a, "A", "red"), (ru581_n, "N", "cyan")):
            m = np.nanmedian(mapped[group])
            ax.plot(m, 1, marker="v", ms=5, color=color)
            ax.text(m, 1.37, f"{tag} {m:.2f}", fontsize=7, ha="center")
    fig.legend(
        handles=[mpatches.Patch(color=COLORS[t], label=t) for t in TYPES]
        + [mpatches.Patch(color="#e8e8e8", label="empty / <3 finite cells")],
        ncol=4,
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.09),
    )
    fig.text(
        0.5,
        -0.14,
        "100 bins; RU1303 SCLC-A median is dashed; RU581 A/N medians are triangles.",
        fontsize=8,
        ha="center",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(OUT / "pairwise_heatmap.png", bbox_inches="tight")
    fig.savefig(OUT / "pairwise_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
