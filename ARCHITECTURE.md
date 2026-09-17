# Architecture — scDeBussy

## Overview

**scDeBussy** aligns developmental pseudotime trajectories across patients or conditions in single-cell RNA-seq data. It uses Dynamic Time Warping (DTW)-based barycenter averaging to warp each patient's local pseudotime onto a shared consensus timeline, enabling cross-batch comparison of gene expression dynamics.

---

## Repository Layout

```
src/scdebussy/          # Installable package (current API)
  tl/                   # Tool layer: algorithms
    _alignment.py       # Core EM alignment (scDeBussy class)
    _metrics.py         # Cross-batch KNN purity evaluation
    _synthetic_dataset.py  # GP-based synthetic benchmark data
  pl/                   # Plot layer: visualizations
    _alignment.py       # Barycenter boundary plots
    _metrics.py         # Purity sweep curve

tests/                  # pytest suite
docs/                   # Sphinx + MyST documentation
```

The package follows the **scanpy/AnnData ecosystem convention**: a `tl` (tools) namespace for algorithms and a `pl` (plots) namespace for visualizations, operating on `AnnData` objects.

---

## Core Data Flow

```
AnnData  (cells × genes, per-patient local pseudotime in obs)
    │
    │  smooth_patient_trajectory()
    │  Gaussian kernel regression → uniform pseudotime grid τ ∈ [0,1]
    ▼
Smoothed matrices  [n_bins × n_genes]  per patient
    │
    │  tl.scDeBussy(adata, ...)  — fitting and projection in one function
    │  ┌─ E-step: asymmetric DTW (Sakoe-Chiba band, open ends)
    │  │          aligns each patient trajectory → barycenter
    │  └─ M-step: soft-DTW barycenter averaging (tslearn)
    │             updates consensus barycenter  B(τ)
    ▼
adata.uns["barycenter"]  +  per-patient warp paths
    │
    │  map_cells_to_aligned_pseudotime()
    │  soft-DTW alignment + isotonic regression (monotonicity)
    ▼
adata.obs["aligned_pseudotime"]
    │
    ├─►  cross_batch_knn_purity()  →  adata.uns["cross_batch_purity"]
    └─►  pl.plot_barycenter_boundaries() / pl.cross_batch_purity_sweep()
```

---

## Key Components

### `tl._alignment` — EM Barycenter Alignment

`smooth_patient_trajectory(adata, patient_key, pseudotime_key, n_bins)` interpolates each patient's cells onto a fixed grid using Gaussian kernel regression (PyTorch tensors for efficiency).

`scDeBussy(adata, ...)` is a function, not an estimator class. It smooths,
fits, and writes aligned pseudotime and barycenter outputs in one call. The
legacy DTW path and opt-in soft-EM path remain unchanged in v2 development.

The additive `scdebussy.align` package exposes separate `fit`, `project`,
`summarize`, and `compare` functions. Its immutable lattice graph is shared by
enumeration and Numba inference; fitting returns an `AlignmentResult` without
mutating AnnData. Package exports are lazy so v2 does not import Torch.
See [the v2 contract](docs/v2.md) for terminal rules, locality support, and
the distinction between posterior projection and MAP diagnostics.

### `tl._metrics` — Alignment Quality

`cross_batch_knn_purity(adata, k)` measures how well biological cell types are conserved across batches after alignment: for each cell, it finds *k* nearest neighbors in the 1-D aligned-pseudotime space and reports the fraction of cross-batch neighbors that share the same cell-type label. Cell types present in only one batch are excluded.

`cross_batch_purity_sweep(adata, k_values)` returns purity scores over a range of *k* to assess stability.

### `tl._synthetic_dataset` — Benchmarking

`simulate_LF_MOGP()` generates ground-truth data with known alignment:
- Latent factors (monotone up/down, Gaussian bump, flat) sampled from RBF-kernel GPs.
- Patient-specific GP deviations; Poisson-sampled cell counts.
- Configurable pseudotime distributions (`'early'`, `'late'`, `'transition'`, `'bimodal'`, `'full'`).

### `pl` — Visualizations

| Function | Output |
|---|---|
| `pl.plot_barycenter_boundaries()` | Transcriptomic velocity + detected segment boundaries |
| `pl.cross_batch_purity_sweep()` | Purity vs. *k* stability curve |

---

## Key Dependencies

| Package | Role |
|---|---|
| `anndata` / `scanpy` | Core data structure and ecosystem integration |
| `tslearn` | `softdtw_barycenter`, `soft_dtw_alignment` |
| `dtw` | Asymmetric DTW with Sakoe-Chiba window (EM E-step) |
| `torch` | Vectorised Gaussian kernel smoothing |
| `scikit-learn` | KNN (`NearestNeighbors`), isotonic regression, PCA |
| `numpy` / `pandas` / `matplotlib` | Numerics, DataFrames, plotting |
