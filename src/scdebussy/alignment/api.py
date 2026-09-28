"""Experimental, label-independent linear alignment fitting."""

import hashlib
import platform
import warnings
from dataclasses import replace
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from .consensus import costs, expected_match_energy, initialize, sufficient_statistics
from .landmarks import estimate
from .lattice import LatticeModel
from .posterior import infer
from .types import AlignmentResult, DegenerateCalibrationError


@lru_cache(maxsize=1)
def _source_fingerprint():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def fit(
    adata,
    *,
    patient_key,
    pseudotime_key,
    n_landmarks,
    locality_radius,
    beta,
    mode="open",
    n_min=5,
    epsilon=1e-6,
    eta_i=None,
    eta_d=None,
    kappa_i=3.0,
    kappa_d=3.0,
    calibration_quantile=0.90,
    calibration_band=0.25,
    max_iter=200,
    tol=1e-6,
    patience=3,
    d_min=1e-3,
    chunk_size=1024,
    backend="numba",
    landmark_method="fixed",
    consensus_resolution=None,
    discovery_resolution=None,
    precision_mode="estimated",
    contribution_cap_quantile=0.95,
) -> AlignmentResult:
    """Fit a fixed-temperature Gibbs alignment, without mutating ``adata``.

    ``locality_radius`` and ``beta`` are deliberately required. Expression is
    used as supplied. Gaps can represent temporal resampling or unmatched data;
    neither the objective nor its posteriors identify biological missingness.
    """
    if landmark_method not in {"fixed", "expression"}:
        raise ValueError("landmark_method must be fixed or expression")
    if precision_mode not in {"estimated", "unit", "capped"}:
        raise ValueError("precision_mode must be estimated, unit, or capped")
    if not 0 < contribution_cap_quantile <= 1:
        raise ValueError("contribution_cap_quantile must be in (0,1]")
    consensus_resolution = n_landmarks if consensus_resolution is None else consensus_resolution
    for name, value in (("consensus_resolution", consensus_resolution), ("discovery_resolution", discovery_resolution)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 2):
            raise ValueError(f"{name} must be an integer >= 2")
    if mode not in {"open", "global"} or backend not in {"numpy", "numba"}:
        raise ValueError("mode must be open/global and backend must be numpy/numba")
    for name, value in (
        ("n_landmarks", n_landmarks),
        ("n_min", n_min),
        ("chunk_size", chunk_size),
        ("max_iter", max_iter),
        ("patience", patience),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < (2 if name == "n_landmarks" else 1)
        ):
            raise ValueError(f"{name} must be an integer >= {2 if name == 'n_landmarks' else 1}")
    for name, value in (("locality_radius", locality_radius), ("beta", beta), ("epsilon", epsilon), ("d_min", d_min)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name, value in (("kappa_i", kappa_i), ("kappa_d", kappa_d), ("tol", tol)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not 0 <= calibration_quantile <= 1 or not 0 <= calibration_band <= 1:
        raise ValueError("calibration_quantile and calibration_band must be in [0,1]")
    if (eta_i is None) != (eta_d is None):
        raise ValueError("supply both eta_i and eta_d, or neither")
    landmarks, cell_patients, cell_time = estimate(
        adata,
        patient_key=patient_key,
        pseudotime_key=pseudotime_key,
        n_landmarks=n_landmarks,
        locality_radius=locality_radius,
        n_min=n_min,
        epsilon=epsilon,
        chunk_size=chunk_size,
        landmark_method=landmark_method,
        discovery_resolution=discovery_resolution,
    )
    consensus_coordinates = np.linspace(0.0, 1.0, consensus_resolution)
    barycenter = initialize(landmarks, consensus_coordinates, locality_radius)
    if precision_mode == "unit":
        landmarks = tuple(replace(lm, precision=np.ones_like(lm.precision)) for lm in landmarks)
    elif precision_mode == "capped":
        gene_cost = np.zeros(adata.n_vars)
        for lm in landmarks:
            nearest = np.argmin(np.abs(lm.coordinates[:, None] - consensus_coordinates[None, :]), axis=1)
            residual = lm.expression - barycenter[nearest]
            gene_cost += np.sum((residual[lm.matchable] ** 2) * lm.precision[lm.matchable], axis=0)
        positive = gene_cost[gene_cost > 0]
        cap = np.quantile(positive, contribution_cap_quantile) if len(positive) else 0.0
        weights = np.minimum(1.0, cap / np.maximum(gene_cost, 1e-300))
        landmarks = tuple(replace(lm, precision=lm.precision * weights) for lm in landmarks)
    calibration = "explicit"
    if eta_i is None:
        samples = []
        for lm in landmarks:
            delta = costs(lm, barycenter)
            for i in np.flatnonzero(lm.matchable):
                band = np.abs(lm.coordinates[i] - consensus_coordinates) <= calibration_band
                if band.any():
                    samples.append(delta[i, band].min())
        eta = float(np.quantile(samples, calibration_quantile, method="linear")) if samples else 0.0
        if not np.isfinite(eta) or eta <= 0:
            raise DegenerateCalibrationError("calibration is empty or zero; supply positive eta_i and eta_d explicitly")
        eta_i = eta_d = eta
        calibration = "band_minimum_quantile"
    for value in (eta_i, eta_d):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("fit requires positive finite extension costs")
    config = {
        "patient_key": patient_key,
        "pseudotime_key": pseudotime_key,
        "n_landmarks": n_landmarks,
        "landmark_method": landmark_method,
        "consensus_resolution": consensus_resolution,
        "discovery_resolution": discovery_resolution or max(201, 4 * n_landmarks + 1),
        "discovery_criterion": "density-balanced-standardized-slope-peaks-v1",
        "locality_radius": locality_radius,
        "beta": beta,
        "mode": mode,
        "n_min": n_min,
        "epsilon": epsilon,
        "eta_i": float(eta_i),
        "eta_d": float(eta_d),
        "kappa_i": kappa_i,
        "kappa_d": kappa_d,
        "lambda_i": kappa_i * eta_i,
        "lambda_d": kappa_d * eta_d,
        "calibration": calibration,
        "calibration_quantile": calibration_quantile,
        "calibration_band": calibration_band,
        "max_iter": max_iter,
        "tol": tol,
        "patience": patience,
        "d_min": d_min,
        "chunk_size": chunk_size,
        "backend": backend,
        "precision_mode": precision_mode,
        "contribution_cap_quantile": contribution_cap_quantile,
    }

    def e_step(b, counts=False):
        return tuple(
            infer(
                LatticeModel(costs(lm, b), beta, eta_i, eta_d, kappa_i * eta_i, kappa_d * eta_d, lm.matchable, mode),
                backend=backend,
                count_distribution=counts,
            )
            for lm in landmarks
        )

    posterior = e_step(barycenter)
    history, stable, converged = [], 0, False
    for iteration in range(1, max_iter + 1):
        before = -sum(p.log_z for p in posterior)
        statistics = sufficient_statistics(landmarks, posterior, barycenter)
        updated = statistics.mean
        conditional = before + beta * (
            expected_match_energy(landmarks, posterior, updated)
            - expected_match_energy(landmarks, posterior, barycenter)
        )
        following = e_step(updated)
        after = -sum(p.log_z for p in following)
        slack = 1e-8 * max(1.0, abs(before), abs(conditional), abs(after))
        if conditional > before + slack or after > conditional + slack:
            raise ArithmeticError("fixed-objective EM descent check failed")
        relative = (before - after) / max(1.0, abs(before))
        history.append(
            {
                "iteration": iteration,
                "free_energy_before": before,
                "free_energy_conditional": conditional,
                "free_energy": after,
                "relative_improvement": relative,
            }
        )
        stable = stable + 1 if relative <= tol else 0
        barycenter, posterior = updated, following
        if stable >= patience:
            converged = True
            break
    # Final counts and all diagnostics refer to exactly the returned barycenter.
    posterior = e_step(barycenter, counts=True)
    statistics = sufficient_statistics(landmarks, posterior, barycenter)
    history[-1]["free_energy"] = -sum(p.log_z for p in posterior)
    provenance = {
        "python": platform.python_version(),
        "implementation": "scdebussy.alignment",
        "experimental": True,
        "contract_version": "affine-gap-linear-v1",
        "coordinate_convention": "endpoint-inclusive-local-pseudotime",
        "projection_basis": "adjacent-linear-supported-v1",
        "source_sha256": _source_fingerprint(),
    }
    for package in ("scdebussy", "numpy", "scipy", "numba", "anndata"):
        try:
            provenance[package] = version(package)
        except PackageNotFoundError:
            provenance[package] = "source"
    diagnostic = {}
    for lm, p in zip(landmarks, posterior, strict=True):
        low = p.low_match_probability()
        diagnostic[lm.patient] = {
            "no_match_probability": p.no_match_probability,
            "low_match_probability": low,
            "unaligned": not np.isfinite(p.log_a).any(),
            "path_entropy": p.entropy,
            "numerical_zero_matches": int(np.sum(np.isfinite(p.log_a) & (p.a == 0))),
        }
        if low > 0.5:
            warnings.warn(
                f"{lm.patient}: posterior probability of <10% matched consensus exceeds 0.5",
                RuntimeWarning,
                stacklevel=2,
            )
    return AlignmentResult(
        tuple(lm.patient for lm in landmarks),
        tuple(map(str, adata.var_names)),
        barycenter,
        landmarks,
        posterior,
        statistics.s0,
        statistics.s1,
        statistics.log_s0,
        np.isfinite(statistics.log_s0),
        np.isfinite(statistics.log_s0) & (statistics.log_s0 < np.log(d_min)),
        config,
        history,
        converged,
        np.asarray(adata.obs_names, str),
        cell_patients.copy(),
        cell_time.copy(),
        provenance,
        diagnostic,
        consensus_coordinates,
    )
