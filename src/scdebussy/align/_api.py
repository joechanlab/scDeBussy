"""Experimental, label-independent linear alignment fitting."""

import hashlib
import platform
import warnings
from dataclasses import replace
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from ._barycenter import costs, expected_match_energy, initialize, sufficient_statistics
from ._dp import check_joint_invariants, infer, infer_joint
from ._landmarks import estimate
from ._model import LatticeModel
from ._types import AlignmentResult, DegenerateCalibrationError


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
    warp="off",
    omega_warp=0.0,
    lambda_terminal_gap=0.0,
) -> AlignmentResult:
    """Fit a fixed-temperature Gibbs alignment, without mutating ``adata``.

    ``locality_radius`` and ``beta`` are deliberately required. Expression is
    used as supplied. Gaps can represent temporal resampling or unmatched data;
    neither the objective nor its posteriors identify biological missingness.

    ``warp``/``omega_warp``/``lambda_terminal_gap`` are an opt-in, experimental
    joint posterior-warp mode (see ``validation/v21/experiment_round/
    JOINT_WARP_CONTRACT.md``). ``warp="off"`` (the default) is byte-identical
    to the pre-existing behavior. ``warp="on"`` requires ``mode="open"`` and a
    positive ``lambda_terminal_gap``, mirroring ``LatticeModel``'s own
    validation, and replaces the E-step's ``infer``/``Posterior`` with
    ``infer_joint``/``JointPosterior`` throughout the EM loop.
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
    if warp not in {"off", "on"}:
        raise ValueError("warp must be 'off' or 'on'")
    if not np.isfinite(omega_warp) or omega_warp < 0:
        raise ValueError("omega_warp must be finite and nonnegative")
    if not np.isfinite(lambda_terminal_gap) or lambda_terminal_gap < 0:
        raise ValueError("lambda_terminal_gap must be finite and nonnegative")
    if warp == "on":
        if mode != "open":
            raise ValueError("warp='on' requires mode='open'")
        if lambda_terminal_gap <= 0:
            raise ValueError("warp='on' requires lambda_terminal_gap > 0")
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
        "warp": warp,
        "omega_warp": omega_warp,
        "lambda_terminal_gap": lambda_terminal_gap,
    }

    def e_step(b, counts=False):
        if warp == "on":
            return tuple(
                infer_joint(
                    LatticeModel(
                        costs(lm, b),
                        beta,
                        eta_i,
                        eta_d,
                        kappa_i * eta_i,
                        kappa_d * eta_d,
                        lm.matchable,
                        mode,
                        warp="on",
                        omega_warp=omega_warp,
                        lambda_terminal_gap=lambda_terminal_gap,
                    ),
                    backend=backend,
                )
                for lm in landmarks
            )
        return tuple(
            infer(
                LatticeModel(costs(lm, b), beta, eta_i, eta_d, kappa_i * eta_i, kappa_d * eta_d, lm.matchable, mode),
                backend=backend,
                count_distribution=counts,
            )
            for lm in landmarks
        )

    def joint_safeguards(posteriors):
        """Raise on invariant violations; return coarse pathology diagnostics."""
        for lm, p in zip(landmarks, posteriors, strict=True):
            check_joint_invariants(p, len(lm.coordinates), consensus_resolution)
        total_occupancy = sum(p.consensus_occupancy for p in posteriors)
        shares = [float(np.mean(p.consensus_occupancy / np.maximum(total_occupancy, 1e-12))) for p in posteriors]
        dominance = max(shares) if shares else 0.0
        if len(posteriors) > 1 and dominance > 0.6:
            warnings.warn(
                "joint-mode consensus occupancy is dominated by a single patient (possible degenerate fit)",
                RuntimeWarning,
                stacklevel=3,
            )
        return {
            "e11_mass": sum(p.expected_match_11 for p in posteriors),
            "warp_mass": sum(p.expected_match_12 + p.expected_match_21 for p in posteriors),
            "internal_unmatched_mass": sum(float(p.patient_unmatched_internal.sum()) for p in posteriors),
            "terminal_unmatched_mass": sum(float(p.patient_unmatched_terminal.sum()) for p in posteriors),
            "consensus_dominance": dominance,
        }

    posterior = e_step(barycenter)
    if warp == "on":
        joint_safeguards(posterior)
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
        row = {
            "iteration": iteration,
            "free_energy_before": before,
            "free_energy_conditional": conditional,
            "free_energy": after,
            "relative_improvement": relative,
        }
        if warp == "on":
            row["barycenter_change"] = float(np.max(np.abs(updated - barycenter)))
            row.update(joint_safeguards(following))
        history.append(row)
        stable = stable + 1 if relative <= tol else 0
        barycenter, posterior = updated, following
        if stable >= patience:
            converged = True
            break
    # Final counts and all diagnostics refer to exactly the returned barycenter.
    posterior = e_step(barycenter, counts=True)
    if warp == "on":
        joint_safeguards(posterior)
    statistics = sufficient_statistics(landmarks, posterior, barycenter)
    history[-1]["free_energy"] = -sum(p.log_z for p in posterior)
    provenance = {
        "python": platform.python_version(),
        "implementation": "scdebussy.align.v2",
        "experimental": True,
        "contract_version": "affine-gap-linear-v2.1",
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
    if warp == "on":
        # low_match_probability/no_match_probability are old-mode-only diagnostics
        # (JOINT_WARP_CONTRACT.md Q5): not extended here, deliberately, because
        # what "low support" should mean (patient vs consensus occupancy) is left
        # open by the plan. Joint mode instead reports the occupancy/mass fields
        # the contract does define, plus pathology flags.
        total_occupancy = sum(p.consensus_occupancy for p in posterior)
        for lm, p in zip(landmarks, posterior, strict=True):
            terminal_mass = float(p.patient_unmatched_terminal.sum())
            all_terminal_dominant = terminal_mass > 0.9 * max(len(lm.coordinates), 1)
            endpoint_occupancy = (
                float(p.consensus_occupancy[0] + p.consensus_occupancy[-1]) if len(p.consensus_occupancy) else 0.0
            )
            share = float(np.mean(p.consensus_occupancy / np.maximum(total_occupancy, 1e-12)))
            diagnostic[lm.patient] = {
                "unaligned": not np.isfinite(p.log_a).any(),
                "path_entropy": p.entropy,
                "numerical_zero_matches": int(np.sum(np.isfinite(p.log_a) & (p.a == 0))),
                "expected_match_11": p.expected_match_11,
                "expected_match_12": p.expected_match_12,
                "expected_match_21": p.expected_match_21,
                "patient_unmatched_internal_mass": float(p.patient_unmatched_internal.sum()),
                "patient_unmatched_terminal_mass": terminal_mass,
                "consensus_unmatched_internal_mass": float(p.consensus_unmatched_internal.sum()),
                "consensus_unmatched_terminal_mass": float(p.consensus_unmatched_terminal.sum()),
                "endpoint_occupancy": endpoint_occupancy,
                "consensus_share": share,
                "degenerate_all_gap_risk": bool(all_terminal_dominant),
            }
            if all_terminal_dominant:
                warnings.warn(
                    f"{lm.patient}: joint posterior mass is dominated by terminal unmatched "
                    "material (possible degenerate all-gap collapse)",
                    RuntimeWarning,
                    stacklevel=2,
                )
    else:
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
    result = AlignmentResult(
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
    if warp == "on":
        # Per-patient mean/max/p95 displacement, computed once against the
        # returned barycenter via the same conditional-position machinery
        # project()/summarize() already use (occupancy-weighted A[i,:]/M_i),
        # rather than duplicating that logic here.
        from ._project import summarize

        quantities = summarize(result)
        for k, patient in enumerate(result.patients):
            displacement = quantities.kinetic_curve[k]
            finite = np.isfinite(displacement)
            magnitude = np.abs(displacement[finite])
            result.diagnostics[patient]["mean_displacement"] = (
                float(np.mean(magnitude)) if finite.any() else float("nan")
            )
            result.diagnostics[patient]["max_displacement"] = float(np.max(magnitude)) if finite.any() else float("nan")
            result.diagnostics[patient]["p95_displacement"] = (
                float(np.percentile(magnitude, 95)) if finite.any() else float("nan")
            )
    return result
