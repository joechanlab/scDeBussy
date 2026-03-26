"""Shared evaluation metrics for the scDeBussy benchmark suite.

All public functions accept an AnnData that has already been processed by
scDeBussy.  They are intentionally free of side-effects and produce plain
Python dicts so results can be serialised directly to JSON.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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
        Keys: ``unsupervised_score``, ``warp_diagonal_deviation``,
        ``warp_step_irregularity``, ``patient_hist_disagreement``,
        ``barycenter_curvature``, ``density_coverage_penalty``.
    """
    if barycenter_key not in adata.uns:
        raise ValueError(f"{barycenter_key!r} not found in adata.uns.")

    meta = adata.uns[barycenter_key]
    warp_paths = meta.get("warp_paths", [])
    bary_expr = np.asarray(meta.get("expression"), dtype=float)

    if bary_expr.ndim != 2:
        raise ValueError("Barycenter expression must be a 2D array of shape (n_bins, n_genes).")

    n_bins = int(bary_expr.shape[0])
    denom = max(1, n_bins - 1)

    diag_devs, step_irrs = [], []
    for path in warp_paths:
        if not path:
            continue
        arr = np.asarray(path, dtype=int)
        i_idx = arr[:, 0]
        j_idx = arr[:, 1]
        diag_devs.append(float(np.mean(np.abs(i_idx - j_idx)) / denom))
        if j_idx.size > 1:
            step_irrs.append(float(np.mean(np.abs(np.diff(j_idx) - 1))))
        else:
            step_irrs.append(0.0)

    warp_diagonal_deviation = float(np.mean(diag_devs)) if diag_devs else 1.0
    warp_step_irregularity = float(np.mean(step_irrs)) if step_irrs else 1.0

    d1 = np.diff(bary_expr, axis=0)
    d2 = np.diff(bary_expr, n=2, axis=0)
    barycenter_curvature = float(np.linalg.norm(d2) / (np.linalg.norm(d1) + 1e-8))

    aligned = adata.obs[key_added].to_numpy(dtype=float)
    patients = adata.obs[patient_key].to_numpy()
    bins = np.linspace(0.0, 1.0, n_hist_bins + 1)

    histograms = []
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
        pairwise = [
            float(np.mean(np.abs(histograms[a] - histograms[b])))
            for a in range(len(histograms))
            for b in range(a + 1, len(histograms))
        ]
        patient_hist_disagreement = float(np.mean(pairwise))
    else:
        patient_hist_disagreement = 0.0

    coverage_penalties = []
    for density in meta.get("patient_densities", {}).values():
        d = np.asarray(density, dtype=float)
        d = d / (np.max(d) + 1e-12)
        coverage_penalties.append(float(np.mean(d < 0.02)))
    density_coverage_penalty = float(np.mean(coverage_penalties)) if coverage_penalties else 0.0

    unsupervised_score = (
        0.40 * warp_diagonal_deviation
        + 0.20 * warp_step_irregularity
        + 0.20 * patient_hist_disagreement
        + 0.15 * barycenter_curvature
        + 0.05 * density_coverage_penalty
    )

    return {
        "unsupervised_score": float(unsupervised_score),
        "warp_diagonal_deviation": float(warp_diagonal_deviation),
        "warp_step_irregularity": float(warp_step_irregularity),
        "patient_hist_disagreement": float(patient_hist_disagreement),
        "barycenter_curvature": float(barycenter_curvature),
        "density_coverage_penalty": float(density_coverage_penalty),
    }
