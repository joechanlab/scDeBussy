"""Shared evaluation metrics for the scDeBussy benchmark suite.

All public functions accept an AnnData that has already been processed by
scDeBussy.  They are intentionally free of side-effects and produce plain
Python dicts so results can be serialised directly to JSON.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scdebussy.tl import smooth_patient_trajectory as _smooth_patient_trajectory

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _warp_patient_trajectory_to_barycenter_grid(
    patient_traj: np.ndarray,
    warp_path: list,
    n_bins: int,
) -> np.ndarray:
    """Project a per-patient smoothed trajectory onto the barycenter pseudotime grid.

    For each barycenter bin *b*, this function averages the patient trajectory values
    at all patient bins that the DTW warp path maps to *b*.  The result is an
    (n_bins, n_genes) matrix aligned to the barycenter coordinate system.

    Parameters
    ----------
    patient_traj:
        Smoothed patient expression matrix of shape ``(n_bins, n_genes)``.
    warp_path:
        DTW warp path—each element is a ``(barycenter_bin_idx, patient_bin_idx)`` pair
        as stored in ``adata.uns[barycenter_key]['warp_paths']``.
    n_bins:
        Number of barycenter bins (output rows).

    Returns
    -------
    warped : np.ndarray, shape (n_bins, n_genes)
    """
    if not warp_path or patient_traj.size == 0:
        n_genes = patient_traj.shape[1] if patient_traj.ndim > 1 else 1
        return np.zeros((n_bins, n_genes), dtype=float)

    n_genes = patient_traj.shape[1]
    warped = np.zeros((n_bins, n_genes), dtype=float)
    path = np.asarray(warp_path, dtype=int)
    bary_idx = path[:, 0]
    pat_idx = path[:, 1]

    for b in range(n_bins):
        sel = pat_idx[bary_idx == b]
        if sel.size > 0:
            warped[b] = patient_traj[sel].mean(axis=0)

    return warped


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


# ---------------------------------------------------------------------------
# Supervised evaluation (requires tau_global ground-truth)
# ---------------------------------------------------------------------------


def evaluate_run(
    adata,
    *,
    key_added: str = "aligned_pseudotime",
    baseline_key: str = "s_local",
    truth_key: str = "tau_global",
    patient_key: str = "patient",
) -> dict:
    """Compute supervised alignment metrics for one benchmark run.

    Parameters
    ----------
    adata : AnnData
        Dataset after ``scDeBussy`` has written ``key_added`` into ``adata.obs``.
    key_added : str
        Key of the aligned pseudotime column in ``adata.obs``.
    baseline_key : str
        Key of the observed (unaligned) pseudotime column in ``adata.obs``.
    truth_key : str
        Key of the ground-truth pseudotime column in ``adata.obs``.
    patient_key : str
        Key of the patient column in ``adata.obs``.

    Returns
    -------
    result : dict
        Nested dict with keys:

        - ``baseline``  : global + per-patient metrics for ``baseline_key``
        - ``aligned``   : global + per-patient metrics for ``key_added``; also
          contains ``n_patients_improved`` and ``worst_patient_rmse``
    """
    truth = adata.obs[truth_key].to_numpy(dtype=float)
    baseline_pred = adata.obs[baseline_key].to_numpy(dtype=float)
    aligned_pred = adata.obs[key_added].to_numpy(dtype=float)
    patients = adata.obs[patient_key].to_numpy()

    # --- global ---
    baseline_global = {
        "pearson_r": _safe_pearson(baseline_pred, truth),
        "rmse": _rmse(baseline_pred, truth),
    }
    aligned_global = {
        "pearson_r": _safe_pearson(aligned_pred, truth),
        "rmse": _rmse(aligned_pred, truth),
    }

    # --- per patient ---
    baseline_pp, aligned_pp = [], []
    for pid in sorted(pd.unique(patients)):
        mask = patients == pid
        bp = _run_patient_metrics(baseline_pred[mask], truth[mask], pid, mask.sum())
        ap = _run_patient_metrics(aligned_pred[mask], truth[mask], pid, mask.sum())
        baseline_pp.append(bp)
        aligned_pp.append(ap)

    # --- fairness stats ---
    n_patients_improved = sum(
        1
        for b, a in zip(baseline_pp, aligned_pp, strict=True)
        if np.isfinite(a["rmse"]) and np.isfinite(b["rmse"]) and a["rmse"] < b["rmse"]
    )
    worst_patient_rmse = float(max((a["rmse"] for a in aligned_pp if np.isfinite(a["rmse"])), default=float("nan")))

    return {
        "baseline": {
            "global_pearson_r": baseline_global["pearson_r"],
            "global_rmse": baseline_global["rmse"],
            "per_patient": baseline_pp,
        },
        "aligned": {
            "global_pearson_r": aligned_global["pearson_r"],
            "global_rmse": aligned_global["rmse"],
            "per_patient": aligned_pp,
            "n_patients_improved": n_patients_improved,
            "worst_patient_rmse": worst_patient_rmse,
        },
    }


def _run_patient_metrics(pred, truth, patient_id, n_cells) -> dict:
    return {
        "patient": str(patient_id),
        "n_cells": int(n_cells),
        "pearson_r": _safe_pearson(pred, truth),
        "rmse": _rmse(pred, truth),
    }


def compute_generic_unsupervised_metrics(
    adata,
    *,
    key_added: str = "aligned_pseudotime",
    patient_key: str = "patient",
    n_hist_bins: int = 40,
) -> dict:
    """Compute method-agnostic unsupervised metrics from aligned pseudotime.

    These metrics are intentionally generic so they can be compared across
    methods that do not expose DTW paths or barycenter internals.
    """
    if key_added not in adata.obs:
        raise ValueError(f"{key_added!r} is missing from adata.obs.")
    if patient_key not in adata.obs:
        raise ValueError(f"{patient_key!r} is missing from adata.obs.")

    aligned = adata.obs[key_added].to_numpy(dtype=float)
    patients = adata.obs[patient_key].to_numpy()

    if aligned.size == 0:
        raise ValueError("Aligned pseudotime vector is empty.")

    range_violation = float(np.mean((aligned < 0.0) | (aligned > 1.0)))

    bins = np.linspace(0.0, 1.0, n_hist_bins + 1)
    histograms = []
    roughness = []
    for pid in sorted(pd.unique(patients)):
        vals = aligned[patients == pid]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        clipped = np.clip(vals, 0.0, 1.0)
        h, _ = np.histogram(clipped, bins=bins, density=False)
        h = h.astype(float)
        h /= h.sum() + 1e-12
        histograms.append(h)

        svals = np.sort(clipped)
        if svals.size > 1:
            roughness.append(float(np.mean(np.abs(np.diff(svals)))))

    if len(histograms) > 1:
        pairwise = [
            float(np.mean(np.abs(histograms[a] - histograms[b])))
            for a in range(len(histograms))
            for b in range(a + 1, len(histograms))
        ]
        patient_hist_disagreement = float(np.mean(pairwise))
    else:
        patient_hist_disagreement = 0.0

    mean_roughness = float(np.mean(roughness)) if roughness else 0.0

    unsupervised_score_generic = 0.70 * patient_hist_disagreement + 0.20 * range_violation + 0.10 * mean_roughness

    return {
        "unsupervised_score_generic": float(unsupervised_score_generic),
        "range_violation": float(range_violation),
        "patient_hist_disagreement": float(patient_hist_disagreement),
        "mean_sorted_step": float(mean_roughness),
    }


# ---------------------------------------------------------------------------
# Unsupervised evaluation (no tau_global required)
# ---------------------------------------------------------------------------


def compute_unsupervised_metrics(
    adata,
    *,
    key_added: str = "aligned_pseudotime",
    barycenter_key: str = "barycenter",
    patient_key: str = "patient",
    n_hist_bins: int = 40,
) -> dict:
    """Compute unsupervised alignment quality metrics that do not require tau_global.

    The primary signal is **cross-patient gene-expression agreement**: after alignment,
    smoothed patient trajectories projected onto the shared barycenter grid are
    pairwise-correlated (Pearson r).  High mean correlation means different patients
    land at the same biological state at each pseudotime coordinate—the exact goal of
    trajectory alignment.  This is a statistically valid proxy for Pearson r with
    ground-truth pseudotime.

    Parameters
    ----------
    adata : AnnData
        Dataset after ``scDeBussy`` has been run.
    key_added : str
        Key of the aligned pseudotime column in ``adata.obs``.
    barycenter_key : str
        Key of the barycenter metadata dict in ``adata.uns``.
    patient_key : str
        Key of the patient column in ``adata.obs``.
    n_hist_bins : int
        Number of histogram bins used for patient-histogram disagreement.

    Returns
    -------
    result : dict
        Keys: ``unsupervised_score`` (lower = better),
        ``mean_cross_patient_trajectory_corr`` (higher = better),
        ``mean_barycenter_reconstruction_corr`` (higher = better),
        ``patient_hist_disagreement``, ``density_coverage_penalty``.
    """
    if barycenter_key not in adata.uns:
        raise ValueError(f"{barycenter_key!r} not found in adata.uns.")
    if key_added not in adata.obs:
        raise ValueError(f"{key_added!r} is missing from adata.obs.")
    if patient_key not in adata.obs:
        raise ValueError(f"{patient_key!r} is missing from adata.obs.")

    meta = adata.uns[barycenter_key]
    params = meta.get("params", {})
    bary_expr = np.asarray(meta.get("expression"), dtype=float)

    if bary_expr.ndim != 2:
        raise ValueError("Barycenter expression must be a 2D array of shape (n_bins, n_genes).")

    n_bins = int(bary_expr.shape[0])
    patient_ids = meta.get("patient_ids", [])
    warp_paths = meta.get("warp_paths", [])

    # -----------------------------------------------------------------------
    # Recompute per-patient smoothed trajectories then project onto barycenter
    # -----------------------------------------------------------------------
    n_bins_param = int(params.get("n_bins", n_bins))
    bandwidth = float(params.get("bandwidth", 0.1))
    pseudotime_key_param = str(params.get("pseudotime_key", "s_local"))
    patient_key_param = str(params.get("patient_key", patient_key))

    warped_trajectories: list[np.ndarray] = []
    for patient_id, warp_path in zip(patient_ids, warp_paths, strict=False):
        try:
            smoothed, _ = _smooth_patient_trajectory(
                adata,
                patient_id=patient_id,
                patient_key=patient_key_param,
                pseudotime_key=pseudotime_key_param,
                n_bins=n_bins_param,
                bandwidth=bandwidth,
            )
            smoothed_np = (
                np.asarray(smoothed.detach().cpu(), dtype=float)
                if hasattr(smoothed, "detach")
                else np.asarray(smoothed, dtype=float)
            )
        except (ValueError, RuntimeError, AttributeError):
            continue
        warped = _warp_patient_trajectory_to_barycenter_grid(smoothed_np, warp_path, n_bins)
        warped_trajectories.append(warped)

    # -----------------------------------------------------------------------
    # Per-gene z-score normalisation across time before correlation
    # (makes the metric scale-invariant across genes and hyperparameter runs)
    # -----------------------------------------------------------------------
    def _normalise_trajectory(mat: np.ndarray) -> np.ndarray:
        mu = mat.mean(axis=0, keepdims=True)
        sigma = mat.std(axis=0, keepdims=True)
        return (mat - mu) / (sigma + 1e-8)

    patient_vecs = [_normalise_trajectory(w).flatten() for w in warped_trajectories]
    bary_vec = _normalise_trajectory(bary_expr).flatten()

    pairwise_corrs: list[float] = []
    bary_recon_corrs: list[float] = []
    for idx, pv in enumerate(patient_vecs):
        bary_recon_corrs.append(_safe_pearson(bary_vec, pv))
        for jdx in range(idx + 1, len(patient_vecs)):
            pairwise_corrs.append(_safe_pearson(pv, patient_vecs[jdx]))

    mean_cross_patient_corr = float(np.nanmean(pairwise_corrs)) if pairwise_corrs else 0.0
    mean_bary_recon_corr = float(np.nanmean(bary_recon_corrs)) if bary_recon_corrs else 0.0

    # -----------------------------------------------------------------------
    # Patient-histogram disagreement on aligned pseudotime
    # -----------------------------------------------------------------------
    aligned = adata.obs[key_added].to_numpy(dtype=float)
    patients = adata.obs[patient_key].to_numpy()
    bins = np.linspace(0.0, 1.0, n_hist_bins + 1)

    histograms: list[np.ndarray] = []
    for pid in sorted(pd.unique(patients)):
        vals = aligned[patients == pid]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        h, _ = np.histogram(np.clip(vals, 0.0, 1.0), bins=bins, density=False)
        h = h.astype(float)
        h /= h.sum() + 1e-12
        histograms.append(h)

    if len(histograms) > 1:
        pairwise_hist = [
            float(np.mean(np.abs(histograms[a] - histograms[b])))
            for a in range(len(histograms))
            for b in range(a + 1, len(histograms))
        ]
        patient_hist_disagreement = float(np.mean(pairwise_hist))
    else:
        patient_hist_disagreement = 0.0

    # -----------------------------------------------------------------------
    # Density coverage penalty
    # -----------------------------------------------------------------------
    coverage_penalties = []
    for density in meta.get("patient_densities", {}).values():
        d = np.asarray(density, dtype=float)
        d = d / (np.max(d) + 1e-12)
        coverage_penalties.append(float(np.mean(d < 0.02)))
    density_coverage_penalty = float(np.mean(coverage_penalties)) if coverage_penalties else 0.0

    # -----------------------------------------------------------------------
    # Composite score — lower is better
    # Primary (55 %): cross-patient expression agreement (higher corr = lower score)
    # Secondary (25 %): barycenter reconstruction quality (higher corr = lower score)
    # Tertiary (15 %): histogram disagreement on aligned pseudotime
    # Minor (5 %): density coverage penalty
    # -----------------------------------------------------------------------
    unsupervised_score = (
        0.55 * (1.0 - mean_cross_patient_corr)
        + 0.25 * (1.0 - mean_bary_recon_corr)
        + 0.15 * patient_hist_disagreement
        + 0.05 * density_coverage_penalty
    )

    return {
        "unsupervised_score": float(unsupervised_score),
        "mean_cross_patient_trajectory_corr": float(mean_cross_patient_corr),
        "mean_barycenter_reconstruction_corr": float(mean_bary_recon_corr),
        "patient_hist_disagreement": float(patient_hist_disagreement),
        "density_coverage_penalty": float(density_coverage_penalty),
    }
