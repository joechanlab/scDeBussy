"""Two-panel Raw | Final summary heatmap for a stakeholder update.

Reuses the exact plotting conventions from ../../summarize_and_plot.py
(patient order, cell-type colors, 100 bins, [0,1] axis, gray empty/<3-cell
rule, status strip for nonfinite/unsupported cells) but restricted to two
panels: raw s_local and the final validated result (post-matchability-fix,
RU1303-only gate == the label-free auto gate, bit-identical). No new
analysis; purely a presentation cut of already-computed, already-verified
results.
"""

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
OUT = Path(__file__).resolve().parent
INPUT = Path("/scratch/chanj3/wangm10/HTAN/results/aligned_scdebussy_preprocessed_v2_ru942_deg_all.h5ad")
CANDIDATE = Path(__file__).resolve().parents[1] / "auto_gate" / "auto_gate.npz"


def label_grid(obs, position):
    target = obs[TYPE_KEY].astype(str).map({k: i / 2 for i, k in enumerate(TYPES)}).to_numpy(float)
    patients = obs[PATIENT_KEY].astype(str).to_numpy()
    grid = np.full((len(PATIENTS), N_BINS), np.nan)
    for i, patient in enumerate(PATIENTS):
        mask = patients == patient
        p = np.asarray(position)[mask]
        finite = np.isfinite(p)
        if not finite.any():
            continue
        values = np.clip(p[finite], 0, 1)
        bins = np.clip(np.digitize(values, np.linspace(0, 1, N_BINS + 1)) - 1, 0, N_BINS - 1)
        for b in range(N_BINS):
            y = target[mask][finite][bins == b]
            if len(y) >= MIN_CELLS:
                grid[i, b] = y.mean()
    return grid


def main():
    data = ad.read_h5ad(INPUT)
    cache = np.load(CANDIDATE, allow_pickle=False)
    if not np.array_equal(cache["cell_ids"], np.asarray(data.obs_names, str)):
        raise ValueError("cell identity/order mismatch")
    cfg = json.loads(str(cache["config"]))
    methods = [
        ("Raw (unaligned $s_{local}$)", data.obs["s_local"].to_numpy(float), np.full(data.n_obs, "input"), np.zeros(data.n_obs)),
        ("Final: 8-patient joint fit, RU1303-only gate", cache["position"], cache["status"], cache["unprojected"]),
    ]

    cmap = LinearSegmentedColormap.from_list("types", [COLORS[x] for x in TYPES])
    fig = plt.figure(figsize=(10.5, 4.7), dpi=180)
    gs = gridspec.GridSpec(len(methods), 2, width_ratios=[40, 1.65], hspace=0.45, wspace=0.04, left=0.10, right=0.77, top=0.88, bottom=0.15)
    for row, (name, position, status, unprojected) in enumerate(methods):
        ax = fig.add_subplot(gs[row, 0])
        strip = fig.add_subplot(gs[row, 1])
        grid = label_grid(data.obs, position)
        ax.imshow(np.ma.masked_invalid(grid), aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1, extent=[0, 1, len(PATIENTS) - 0.5, -0.5])
        ax.set_facecolor("#e8e8e8")
        ax.set_yticks(range(len(PATIENTS)))
        ax.set_yticklabels(PATIENTS, fontsize=9)
        ax.set_title(name, loc="left", fontsize=12, fontweight="bold", pad=6)
        for y in np.arange(0.5, len(PATIENTS) - 0.5):
            ax.axhline(y, color="white", lw=1)
        if row < len(methods) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("pseudotime", fontsize=10)
        patients = data.obs[PATIENT_KEY].astype(str).to_numpy()
        position = np.asarray(position)
        status = np.asarray(status, str)
        for i, patient in enumerate(PATIENTS):
            mask = patients == patient
            rejected = np.mean(~np.isfinite(position[mask]))
            unsupported_finite = np.mean((status[mask] == "unsupported") & np.isfinite(position[mask]))
            strip.barh(i, rejected, color="black", height=0.78, align="center")
            strip.barh(i, unsupported_finite, left=rejected, color="#f28e2b", height=0.78, align="center")
        strip.set_xlim(0, 1)
        strip.set_ylim(len(PATIENTS) - 0.5, -0.5)
        strip.set_yticks([])
        strip.set_xticks([])
        for spine in strip.spines.values():
            spine.set_visible(False)
        if row == 0:
            strip.set_title("rejected", fontsize=7.5, pad=5)
    fig.legend(
        handles=[mpatches.Patch(color=COLORS[x], label=x) for x in TYPES]
        + [mpatches.Patch(color="#e8e8e8", label="empty / <3 cells"), mpatches.Patch(color="black", label="nonfinite projection"), mpatches.Patch(color="#f28e2b", label="finite but unsupported")],
        loc="upper left", bbox_to_anchor=(0.79, 0.88), fontsize=8, frameon=False, title="Cell type / status", title_fontsize=8,
    )
    fig.suptitle("HTAN 8-patient alignment: raw vs. today's final result", x=0.10, y=0.98, ha="left", fontsize=13)
    fig.text(0.10, 0.015, "Gray = empty/<3 finite cells. Right bar: nonfinite (black), finite-but-unsupported (orange).", fontsize=7.5)
    fig.savefig(OUT / "raw_vs_final.png", bbox_inches="tight", facecolor="white")
    print(OUT / "raw_vs_final.png")


if __name__ == "__main__":
    main()
