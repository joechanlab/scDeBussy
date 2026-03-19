# Product — scDeBussy

## Vision

Single-cell RNA-seq studies frequently sample the same biological process (e.g., differentiation, disease progression) across multiple patients or experimental batches. Each patient's cells carry a **local pseudotime** that is independently estimated and has no shared scale with other patients. scDeBussy provides a principled, reference-free method to **align these local pseudotimes onto a common timeline**, making gene expression dynamics directly comparable across cohorts.

---

## Problem Statement

| Challenge | Impact without alignment |
|---|---|
| Each patient's pseudotime starts at 0 and ends at 1 but progresses at different rates | Cross-patient averaging of expression profiles is meaningless |
| Developmental transitions occur at different pseudotime positions per patient | Marker genes appear to activate at inconsistent timepoints |
| Batch effects compound with pseudotime offsets | Downstream analyses (clustering, trajectory inference) conflate technical variation with biology |

---

## Core Functionality

### 1. Pseudotime Alignment (`tl.scDeBussy`)

The primary workflow aligns a cohort of patients to a consensus **barycenter trajectory**:

1. **Smooth** each patient's sparse cell measurements onto a uniform pseudotime grid using Gaussian kernel regression.
2. **Fit** an expectation-maximization (EM) loop:
   - *E-step*: warp each patient's smoothed trajectory to the current barycenter via asymmetric DTW with open-end matching (handles partial developmental coverage per patient).
   - *M-step*: update the barycenter by soft-DTW barycenter averaging across all warped trajectories.
3. **Transform** raw cells to aligned pseudotime by mapping each cell through the learned warp path + isotonic regression to enforce monotonicity.

**Result**: `adata.obs["aligned_pseudotime"]` — a shared pseudotime coordinate valid across all patients.

### 2. Alignment Quality Evaluation (`tl.cross_batch_knn_purity`)

After alignment, quality is measured by **cross-batch KNN purity**:
- For each cell, find its *k* nearest neighbors in the 1-D aligned-pseudotime space.
- Compute the fraction of cross-batch neighbors that share the same annotated cell type.
- A purity of 1.0 means every cross-batch neighbor is biologically matched; 0.0 means random mixing.

`cross_batch_purity_sweep()` evaluates purity over a range of *k* values to ensure the result is not sensitive to the choice of *k*.

### 3. Synthetic Benchmarking (`tl.simulate_LF_MOGP`)

Ground-truth benchmark datasets are generated via a **latent factor multi-output GP** model:
- Structured latent factors (monotone trends, transient pulses, flat baselines) are sampled from RBF-kernel Gaussian Processes.
- Gene expression is a linear combination of factors with known loadings.
- Patient-specific GP deviations simulate inter-individual variability.
- Cell counts are Poisson-sampled; pseudotime distributions are configurable (`'early'`, `'late'`, `'transition'`, `'bimodal'`, `'full'`) to mimic incomplete developmental sampling per patient.

This enables controlled evaluation of alignment accuracy against known ground truth.

### 4. Visualizations (`pl`)

| Plot | Purpose |
|---|---|
| `pl.plot_barycenter_boundaries()` | Overlay transcriptomic velocity with detected segment boundaries on the barycenter |
| `pl.cross_batch_purity_sweep()` | Line plot of purity vs. *k* to visualise alignment stability |

---

## Typical User Workflow

```python
import scdebussy as sd

# 1. Smooth per-patient trajectories onto a shared grid
trajectories = sd.tl.smooth_patient_trajectory(
    adata, patient_key="patient_id", pseudotime_key="dpt_pseudotime", n_bins=100
)

# 2. Align to consensus barycenter
model = sd.tl.scDeBussy(n_bins=100, max_iter=20)
model.fit(trajectories)
model.transform(adata)          # writes adata.obs["aligned_pseudotime"]

# 3. Evaluate alignment quality
sd.tl.cross_batch_knn_purity(adata, k=50, label_key="cell_type",
                              batch_key="patient_id")
print(adata.uns["cross_batch_purity"])

# 4. Visualise
sd.pl.cross_batch_purity_sweep(adata, k_values=range(10, 100, 10))
```

---

## Design Goals

- **AnnData-native**: integrates directly with the scanpy ecosystem; results are stored in standard `obs` / `uns` slots.
- **Reference-free**: no single patient is chosen as a reference; the barycenter is a consensus learned from all patients jointly.
- **Partial coverage**: open-end DTW matching handles patients that only cover early or late portions of the trajectory without forcing artificial alignment at the boundaries.
- **Interpretable quality metric**: the KNN purity score provides a single number that can be tracked across parameter choices or compared between methods.
