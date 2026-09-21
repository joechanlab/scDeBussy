"""Create a fixed-layout HTAN comparison figure and machine-readable metrics."""

import argparse
import csv
import json
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
from sklearn.neighbors import NearestNeighbors

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


PATIENT_KEY = "patient"
TYPE_KEY = "cell_type_final2"
PATIENTS = ["RU1083", "RU1303", "RU1444", "RU151", "RU263", "RU581", "RU831", "RU942"]
TYPES = ["LUAD", "SCLC-A", "SCLC-N"]
COLORS = {"LUAD": "gold", "SCLC-A": "tab:red", "SCLC-N": "tab:cyan"}
N_BINS, MIN_CELLS = 100, 3


def robust_rescale(x):
    lo, hi = np.nanquantile(x, [0.01, 0.99])
    return np.clip((x - lo) / (hi - lo), 0, 1)


def label_grid(obs, position, rescale):
    target = obs[TYPE_KEY].astype(str).map({k: i / 2 for i, k in enumerate(TYPES)}).to_numpy(float)
    patients = obs[PATIENT_KEY].astype(str).to_numpy()
    grid = np.full((len(PATIENTS), N_BINS), np.nan)
    for i, patient in enumerate(PATIENTS):
        mask = patients == patient
        p = np.asarray(position)[mask]
        finite = np.isfinite(p)
        if not finite.any():
            continue
        values = robust_rescale(p[finite]) if rescale else np.clip(p[finite], 0, 1)
        bins = np.clip(np.digitize(values, np.linspace(0, 1, N_BINS + 1)) - 1, 0, N_BINS - 1)
        for b in range(N_BINS):
            y = target[mask][finite][bins == b]
            if len(y) >= MIN_CELLS:
                grid[i, b] = y.mean()
    return grid


def score(obs, position, status, unprojected, method):
    patients = obs[PATIENT_KEY].astype(str).to_numpy()
    labels = obs[TYPE_KEY].astype(str).to_numpy()
    local = obs["s_local"].to_numpy(float)
    position = np.asarray(position, float)
    status = np.asarray(status, str)
    unprojected = np.asarray(unprojected, float)
    finite = np.isfinite(position)
    neighbor_score = np.full(len(obs), np.nan)
    if finite.sum() >= 50:
        ix = np.flatnonzero(finite)
        nn = NearestNeighbors(n_neighbors=50, algorithm="kd_tree").fit(position[ix, None])
        _, neighbors = nn.kneighbors(position[ix, None])
        for local_ix, original_ix in enumerate(ix):
            candidate_ix = ix[neighbors[local_ix]]
            cross = patients[candidate_ix] != patients[original_ix]
            neighbor_score[original_ix] = np.mean(labels[candidate_ix][cross] == labels[original_ix]) if cross.any() else 0.0
    bin_index = np.zeros(len(position), dtype=int)
    bin_index[finite] = np.clip((position[finite] * 99).astype(int), 0, 99)
    per_bin = np.zeros(N_BINS, float)
    for b in range(N_BINS):
        in_bin = finite & (bin_index == b)
        per_bin[b] = len(np.unique(patients[in_bin])) / len(PATIENTS) if in_bin.any() else 0
    rows = []
    for patient in PATIENTS:
        mask = patients == patient
        valid = mask & finite
        occupied = np.zeros(N_BINS, bool)
        if valid.any():
            occupied[np.clip((position[valid] * 99).astype(int), 0, 99)] = True
        def scalar(fun):
            values = fun[mask]
            values = values[np.isfinite(values)]
            return float(np.mean(values)) if len(values) else float("nan")
        rows.append({
            "method": method, "scope": "patient", "patient": patient, "cells": int(mask.sum()),
            "coverage": float(occupied.mean()), "purity": scalar(neighbor_score),
            "rejection_rate": float((mask & ~finite).sum() / mask.sum()),
            "unsupported_rate": float((status[mask] == "unsupported").mean()),
            "mean_unprojected_mass": float(unprojected[mask].mean()),
            "mean_abs_displacement": float(np.abs(position[valid] - local[valid]).mean()) if valid.any() else float("nan"),
            "max_abs_displacement": float(np.abs(position[valid] - local[valid]).max()) if valid.any() else float("nan"),
        })
    ru = patients == "RU1303"
    other_n = position[(~ru) & (labels == "SCLC-N") & finite]
    ru_a = position[ru & (labels == "SCLC-A") & finite]
    ru_overlap = float(np.mean((ru_a >= np.quantile(other_n, .1)) & (ru_a <= np.quantile(other_n, .9)))) if len(other_n) and len(ru_a) else float("nan")
    rows.append({
        "method": method, "scope": "overall", "patient": "all", "cells": int(len(obs)),
        "coverage": float(per_bin.mean()), "purity": float(np.nanmean(neighbor_score)),
        "rejection_rate": float((~finite).mean()), "unsupported_rate": float((status == "unsupported").mean()),
        "mean_unprojected_mass": float(unprojected.mean()),
        "mean_abs_displacement": float(np.abs(position[finite] - local[finite]).mean()),
        "max_abs_displacement": float(np.abs(position[finite] - local[finite]).max()),
        "median_abs_displacement": float(np.median(np.abs(position[finite] - local[finite]))),
        "p95_abs_displacement": float(np.quantile(np.abs(position[finite] - local[finite]), .95)),
        "ru1303_sclc_a_in_other_sclc_n_q10_q90": ru_overlap,
    })
    overall = rows[-1]
    for cell_type in TYPES:
        patient_medians = []
        for patient in PATIENTS:
            mask = (patients == patient) & (labels == cell_type) & finite
            if mask.any():
                median = float(np.median(position[mask]))
                overall[f"median_{cell_type}_{patient}"] = median
                patient_medians.append(median)
        overall[f"median_{cell_type}_spread"] = float(max(patient_medians) - min(patient_medians)) if patient_medians else float("nan")
    return rows


def load_candidate(path, data):
    d = np.load(path, allow_pickle=False)
    if not np.array_equal(d["cell_ids"], np.asarray(data.obs_names, str)):
        raise ValueError(f"cell identity/order mismatch in {path}")
    return d["position"], d["status"], d["unprojected"], json.loads(str(d["config"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=Path, help="Explicit cache selection and order")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = ad.read_h5ad(args.adata)
    expected = ["RU1083", "RU1303", "RU1444", "RU151", "RU263", "RU581", "RU831", "RU942"]
    if sorted(data.obs[PATIENT_KEY].astype(str).unique()) != sorted(expected):
        raise ValueError("the input must contain exactly the frozen eight patients")
    methods = [("Unaligned input ($s_{local}$)", data.obs["s_local"].to_numpy(float), np.full(data.n_obs, "input"), np.zeros(data.n_obs), False, {"input": str(args.adata)})]
    candidate_paths = args.candidate or sorted(args.candidate_dir.glob("*.npz"), key=lambda p: (p.stem != "original_v2", p.stem != "centered_default", p.stem))
    for path in candidate_paths:
        position, status, unprojected, cfg = load_candidate(path, data)
        methods.append((cfg["name"], position, status, unprojected, False, cfg))
    if len(methods) < 3:
        raise ValueError("expected original v2 plus at least one candidate")
    all_rows = []
    configs = {}
    for name, position, status, unprojected, rescale, cfg in methods:
        all_rows.extend(score(data.obs, position, status, unprojected, name))
        configs[name] = cfg
    fields = sorted({key for row in all_rows for key in row})
    with (args.output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(all_rows)
    (args.output_dir / "configurations.json").write_text(json.dumps(configs, indent=2) + "\n")
    cmap = LinearSegmentedColormap.from_list("types", [COLORS[x] for x in TYPES])
    fig = plt.figure(figsize=(10.2, 2.1 * len(methods)), dpi=180)
    gs = gridspec.GridSpec(len(methods), 2, width_ratios=[40, 1.65], hspace=.34, wspace=.04, left=.10, right=.77, top=.95, bottom=.07)
    for row, (name, position, status, unprojected, rescale, _) in enumerate(methods):
        ax = fig.add_subplot(gs[row, 0]); strip = fig.add_subplot(gs[row, 1])
        grid = label_grid(data.obs, position, rescale)
        ax.imshow(np.ma.masked_invalid(grid), aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1, extent=[0, 1, len(PATIENTS)-.5, -.5])
        ax.set_facecolor("#e8e8e8")
        ax.set_yticks(range(len(PATIENTS))); ax.set_yticklabels(PATIENTS, fontsize=9)
        ax.set_title(name, loc="left", fontsize=11, fontweight="bold", pad=4)
        for y in np.arange(.5, len(PATIENTS)-.5): ax.axhline(y, color="white", lw=1)
        if row < len(methods)-1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("pseudotime", fontsize=10)
        patients = data.obs[PATIENT_KEY].astype(str).to_numpy(); position = np.asarray(position); status = np.asarray(status, str)
        for i, patient in enumerate(PATIENTS):
            mask = patients == patient
            rejected = np.mean(~np.isfinite(position[mask])); unsupported_finite = np.mean((status[mask] == "unsupported") & np.isfinite(position[mask]))
            strip.barh(i, rejected, color="black", height=.78, align="center")
            strip.barh(i, unsupported_finite, left=rejected, color="#f28e2b", height=.78, align="center")
        strip.set_xlim(0, 1); strip.set_ylim(len(PATIENTS)-.5, -.5); strip.set_yticks([]); strip.set_xticks([])
        for spine in strip.spines.values(): spine.set_visible(False)
        if row == 0: strip.set_title("rejected", fontsize=7.5, pad=5)
    fig.legend(handles=[mpatches.Patch(color=COLORS[x], label=x) for x in TYPES] + [mpatches.Patch(color="#e8e8e8", label="empty / <3 cells"), mpatches.Patch(color="black", label="nonfinite projection"), mpatches.Patch(color="#f28e2b", label="finite but unsupported")], loc="upper left", bbox_to_anchor=(.79, .95), fontsize=8, frameon=False, title="Cell type / status", title_fontsize=8)
    fig.suptitle("HTAN alignment experiment: fixed patient order, bins, and axes", x=.10, ha="left", fontsize=12)
    fig.text(.10, .012, "Gray bins are empty or have fewer than 3 finite cells. The separate bar at right shows nonfinite cells (black) and finite unsupported cells (orange).", fontsize=7.3)
    fig.savefig(args.output_dir / "htan_comparison.png", bbox_inches="tight", facecolor="white")
    fig.savefig(args.output_dir / "htan_comparison.pdf", bbox_inches="tight", facecolor="white")
    print(args.output_dir / "htan_comparison.png")


if __name__ == "__main__":
    main()
