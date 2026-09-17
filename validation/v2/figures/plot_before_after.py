"""Patient x pseudotime heatmap, before vs after alignment, with scDeBussy v2 added.

Adapted from figures/figureS7/plot_before_after_alignment_heatmap.py: same target
construction, same robust rescale, same cell-type palette. The per-bin consistency
strip and the label-free metric annotations are omitted.
"""

from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SCRATCH = Path("/tmp/claude-164119608/-data1-chanj3-wangm10-scDeBussy/c71890ea-55a9-455a-9c01-c8686eb93d37/scratchpad")
H5AD = "/scratch/chanj3/wangm10/HTAN/results/aligned_scdebussy_preprocessed_v3.h5ad"

GRAY = np.array([0.91, 0.91, 0.91])
PATIENT_KEY = "patient"
ORDINAL_KEY = "cell_type_final2"
ORDINAL_ORDER = ["LUAD", "SCLC-A", "SCLC-N"]
CELL_TYPE_COLORS = {"LUAD": "gold", "SCLC-A": "tab:red", "SCLC-N": "tab:cyan"}
N_BINS, MIN_CELLS = 100, 3


def build_target(obs, ordinal_key, ordinal_order):
    """Map the ordinal cell-type column onto evenly spaced values in [0, 1]."""
    order = {name: i for i, name in enumerate(ordinal_order)}
    missing = set(obs[ordinal_key].astype(str).unique()) - set(order)
    if missing:
        raise ValueError(f"{ordinal_key!r} has categories not in ordinal_order: {sorted(missing)}")
    return obs[ordinal_key].astype(str).map(order).to_numpy(dtype=float) / max(len(order) - 1, 1)


def robust_rescale(vals, q_low=0.01, q_high=0.99):
    """Rescale to [0, 1] against percentiles rather than min/max.

    One outlier cell can otherwise set a whole patient's scale (observed on this
    cohort).
    """
    lo, hi = np.nanquantile(vals, [q_low, q_high])
    return np.clip((vals - lo) / (hi - lo), 0.0, 1.0)


def per_patient_mean_target(obs, patients, axis_key, target, n_bins, min_cells, rescale):
    """Return an (n_patients, n_bins) grid of the mean target value per bin.

    Never rescale an already-aligned axis -- re-stretching each patient back to
    [0, 1] erases the coverage differences alignment is meant to show.
    """
    edges = np.linspace(0, 1, n_bins + 1)
    grid = np.full((len(patients), n_bins), np.nan)
    for i, p in enumerate(patients):
        m = (obs[PATIENT_KEY].astype(str) == p).to_numpy()
        raw = obs.loc[m, axis_key].astype(float).to_numpy()
        finite = np.isfinite(raw)  # digitize would sort NaN into the last bin
        vals = robust_rescale(raw[finite]) if rescale else np.clip(raw[finite], 0.0, 1.0)
        tgt = target[m][finite]
        b = np.clip(np.digitize(vals, edges) - 1, 0, n_bins - 1)
        for j in range(n_bins):
            sel = tgt[b == j]
            if sel.size >= min_cells:
                grid[i, j] = sel.mean()
    return grid


# ---- assemble the three axes ----------------------------------------------------------
adata = ad.read_h5ad(H5AD)
obs = adata.obs.copy()
cached = np.load(SCRATCH / "htan_barycenters.npz", allow_pickle=True)
assert list(cached["cell_patient"]) == list(obs[PATIENT_KEY].astype(str)), "cell order mismatch"
# Name must contain "aligned" so the rescale rule above treats it as a shared coordinate.
obs["aligned_pseudotime_v2"] = cached["v2_pos"]

PANELS = [
    ("BEFORE (local pseudotime)", "s_local"),
    ("scDeBussy v1", "aligned_pseudotime"),
    ("scDeBussy v2", "aligned_pseudotime_v2"),
]

patients = sorted(obs[PATIENT_KEY].astype(str).unique())
target = build_target(obs, ORDINAL_KEY, ORDINAL_ORDER)
cmap = LinearSegmentedColormap.from_list("cell_type", [CELL_TYPE_COLORS[c] for c in ORDINAL_ORDER])

fig = plt.figure(figsize=(7.5, 2.35 * len(PANELS)))
outer = gridspec.GridSpec(len(PANELS), 1, hspace=0.30, right=0.86, bottom=0.07, top=0.95)

for row, (label, col) in enumerate(PANELS):
    rescale = "aligned" not in col
    grid = per_patient_mean_target(obs, patients, col, target, N_BINS, MIN_CELLS, rescale)
    ax = fig.add_subplot(outer[row])
    ax.imshow(
        np.ma.masked_invalid(grid),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=1,
        extent=[0, 1, len(patients) - 0.5, -0.5],
    )
    ax.set_facecolor(GRAY)
    ax.set_yticks(range(len(patients)))
    ax.set_yticklabels(patients, fontsize=10)
    ax.set_title(label, fontsize=13, loc="left", pad=4, fontweight="bold")
    for y in np.arange(0.5, len(patients) - 0.5):
        ax.axhline(y, color="white", lw=1.2)
    if row == len(PANELS) - 1:
        ax.set_xlabel("pseudotime", fontsize=11)
    else:
        ax.set_xticklabels([])

fig.legend(
    handles=[mpatches.Patch(color=CELL_TYPE_COLORS[c], label=c) for c in ORDINAL_ORDER],
    loc="upper right",
    bbox_to_anchor=(0.995, 0.995),
    fontsize=9,
    title="Cell type",
    title_fontsize=9,
    frameon=False,
)
fig.suptitle("HTAN LUAD-to-SCLC alignment", fontsize=12, y=1.0)

out = SCRATCH / "htan_before_after_v2.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print("saved:", out)

for label, col in PANELS:
    rescale = "aligned" not in col
    g = per_patient_mean_target(obs, patients, col, target, N_BINS, MIN_CELLS, rescale)
    filled = np.isfinite(g).sum(1)
    print(
        f"{label:<28} bins with >={MIN_CELLS} cells: mean {filled.mean():.0f}/{N_BINS}"
        f"  (min {filled.min()}, max {filled.max()})"
    )
