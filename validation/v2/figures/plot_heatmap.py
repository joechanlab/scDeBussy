"""Barycenter heatmap after v2 alignment, with v1 under the same gene ordering."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.use("agg")
SCRATCH = Path("/tmp/claude-164119608/-data1-chanj3-wangm10-scDeBussy/c71890ea-55a9-455a-9c01-c8686eb93d37/scratchpad")
d = np.load(SCRATCH / "htan_barycenters.npz", allow_pickle=True)

v1, v2 = d["v1_bary"], d["v2_bary"]  # (100 landmarks, 2370 genes)
genes = d["v2_genes"]
assert list(genes) == list(d["v1_genes"]), "gene order differs between v1 and v2"

MARKERS = {  # validated phase markers for LUAD -> SCLC-A -> SCLC-N
    "TACSTD2": "LUAD",
    "MYC": "LUAD",
    "EZH2": "LUAD",
    "ASCL1": "SCLC-A",
    "CHGA": "SCLC-A",
    "NEUROD1": "SCLC-N",
    "DLL3": "SCLC-N",
    "INSM1": "SCLC-N",
}
PHASE_INK = {"LUAD": "#7B3294", "SCLC-A": "#008837", "SCLC-N": "#C2571A"}


def zscore(bary):
    """Z-score each gene across landmarks; drop genes with no variation."""
    x = bary.T.astype(float)  # (genes, landmarks)
    mu, sd = x.mean(1, keepdims=True), x.std(1, keepdims=True)
    ok = (sd[:, 0] > 0) & np.isfinite(x).all(1)
    return (x - mu) / np.where(sd > 0, sd, 1.0), ok


z2, ok2 = zscore(v2)
z1, ok1 = zscore(v1)
keep = ok1 & ok2
z1, z2, kept_genes = z1[keep], z2[keep], genes[keep]

# Order genes ONCE, by centre of mass of positive v2 signal (steadier than argmax).
w = np.clip(z2, 0, None)
com = (w * np.arange(z2.shape[1])).sum(1) / np.maximum(w.sum(1), 1e-12)
order = np.argsort(com)
z2, z1, kept_genes = z2[order], z1[order], kept_genes[order]

tau = np.linspace(0, 1, z2.shape[1])
VLIM = 2.5

fig = plt.figure(figsize=(11.2, 6.9), dpi=200)
gs = fig.add_gridspec(
    2,
    3,
    width_ratios=[1, 1, 0.045],
    height_ratios=[1, 0.13],
    hspace=0.12,
    wspace=0.16,
    left=0.13,
    right=0.92,
    top=0.86,
    bottom=0.10,
)

axes = []
for col, (z, name) in enumerate([(z2, "scDeBussy v2"), (z1, "v1 (same gene order)")]):
    ax = fig.add_subplot(gs[0, col])
    axes.append(ax)
    im = ax.imshow(
        z, aspect="auto", cmap="RdBu_r", vmin=-VLIM, vmax=VLIM, interpolation="nearest", extent=[0, 1, z.shape[0], 0]
    )
    ax.set_title(
        name,
        fontsize=11,
        pad=7,
        fontweight="bold" if col == 0 else "normal",
        color="#1A1A1A" if col == 0 else "#595959",
    )
    ax.tick_params(labelsize=8.5, colors="#595959", length=3, labelbottom=False)
    for s in ax.spines.values():
        s.set_color("#D4D4D4")
    if col == 0:
        ax.set_ylabel(f"{z.shape[0]:,} genes  (ordered by v2 peak)", fontsize=9.5, color="#404040")
        pos = {g: i for i, g in enumerate(kept_genes)}
        found = sorted([(g, pos[g]) for g in MARKERS if g in pos], key=lambda t: t[1])
        # Nudge labels apart so adjacent markers stay legible.
        rows, gap = [], z.shape[0] * 0.028
        for _, i in found:
            rows.append(i if not rows else max(i, rows[-1] + gap))
        ax.set_yticks(rows)
        ax.set_yticklabels([g for g, _ in found], fontsize=7.6)
        for lbl, (g, _) in zip(ax.get_yticklabels(), found, strict=True):
            lbl.set_color(PHASE_INK[MARKERS[g]])
        for (_, i), r in zip(found, rows, strict=True):
            ax.plot(
                [-0.035, -0.008], [r, i], color="#8C8C8C", lw=0.5, clip_on=False, transform=ax.get_yaxis_transform()
            )
            ax.axhline(i, color="#333333", lw=0.45, alpha=0.45)
    else:
        ax.set_yticks([])

# Roughness strip: mean |change| between adjacent landmarks — where the consensus jumps.
for col, (b, _ax) in enumerate(zip([v2, v1], axes, strict=True)):
    step = np.abs(np.diff(b.astype(float), axis=0)).mean(1)
    axs = fig.add_subplot(gs[1, col])
    axs.fill_between(tau[1:], step, color="#4C4C4C", lw=0)
    axs.set_ylim(0, 0.16)
    axs.set_xlim(0, 1)
    axs.set_yticks([])
    axs.tick_params(labelsize=8.5, colors="#595959", length=3)
    axs.set_xlabel("aligned pseudotime", fontsize=9.5, color="#404040")
    for side in ("top", "right", "left"):
        axs.spines[side].set_visible(False)
    axs.spines["bottom"].set_color("#D4D4D4")
    if col == 0:
        axs.set_ylabel(
            "consensus\nroughness", fontsize=7.5, rotation=0, ha="right", va="center", color="#595959", labelpad=10
        )
    axs.text(
        0.02,
        0.86,
        f"mean {step.mean():.3f} · max/median {step.max() / np.median(step):.1f}",
        transform=axs.transAxes,
        fontsize=7.4,
        color="#707070",
        va="top",
    )

cax = fig.add_subplot(gs[0, 2])
cb = fig.colorbar(im, cax=cax)
cb.set_label("expression z-score (per gene)", fontsize=8.8, color="#404040")
cb.ax.tick_params(labelsize=8, colors="#595959", length=3)
cb.outline.set_edgecolor("#D4D4D4")

fig.suptitle(
    "HTAN LUAD→SCLC consensus trajectory after alignment  ·  9 patients, 34,553 cells",
    fontsize=12.5,
    fontweight="bold",
    color="#1A1A1A",
    x=0.13,
    ha="left",
    y=0.965,
)
fig.text(
    0.13,
    0.915,
    "Gene order is defined by the v2 panel, so its diagonal is partly by construction — "
    "the v1 panel, under that same order, is the honest comparison.",
    fontsize=8.6,
    color="#707070",
    ha="left",
)
handles = [plt.Line2D([], [], marker="s", ls="", ms=6, color=c, label=k) for k, c in PHASE_INK.items()]
fig.legend(
    handles=handles,
    loc="lower left",
    bbox_to_anchor=(0.13, 0.008),
    ncol=3,
    frameon=False,
    fontsize=8.5,
    handletextpad=0.4,
    columnspacing=1.4,
)

out = SCRATCH / "htan_v2_heatmap.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("wrote", out)

# Where do the markers actually land, on each method's own ordering?
print(f"\n{'gene':<9}{'phase':<9}{'v2 peak tau':>12}{'v1 peak tau':>12}")
for g in MARKERS:
    if g in set(kept_genes):
        i = list(kept_genes).index(g)
        print(f"{g:<9}{MARKERS[g]:<9}{tau[np.argmax(z2[i])]:>12.2f}{tau[np.argmax(z1[i])]:>12.2f}")
    else:
        print(f"{g:<9}{MARKERS[g]:<9}{'dropped by HVG filter':>25}")
