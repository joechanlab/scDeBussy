from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TuningResult:
    """Container for tuning outputs across all supported methods."""

    method: str
    objective: str
    n_trials: int
    n_successful_trials: int
    best_score: float
    best_params: dict[str, Any]
    sweep_df: pd.DataFrame
    best_method_result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary payload."""
        return {
            "method": self.method,
            "objective": self.objective,
            "n_trials": int(self.n_trials),
            "n_successful_trials": int(self.n_successful_trials),
            "best_score": float(self.best_score),
            "best_params": dict(self.best_params),
        }


_TUNABLE_METHODS = {
    "scdebussy",
    "cellalign_consensus",
    "cellalign_fixed_reference",
    "genes2genes_consensus",
    "genes2genes_fixed_reference",
}


def tunable_methods() -> list[str]:
    """Return methods that currently expose a built-in tuning search space."""
    return sorted(_TUNABLE_METHODS)


def is_method_tunable(method: str) -> bool:
    """Return whether a benchmark method has a formal tuning configuration."""
    return method in _TUNABLE_METHODS


def _import_optuna():
    try:
        import optuna
        from optuna.pruners import MedianPruner
        from optuna.samplers import TPESampler
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Optuna is required for tuning APIs. Install with: pip install optuna") from exc
    return optuna, TPESampler, MedianPruner


def _import_benchmark_runtime():
    try:
        from scripts.benchmark.methods import run_method
        from scripts.benchmark.metrics import (
            compute_generic_unsupervised_metrics,
            compute_unsupervised_metrics,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Benchmark tuning requires repository benchmark modules under scripts/benchmark.") from exc
    return run_method, compute_generic_unsupervised_metrics, compute_unsupervised_metrics


def _preflight_method_dependencies(method: str) -> None:
    """Fail fast when a method's external runtime dependency is missing."""
    if method.startswith("genes2genes"):
        if importlib.util.find_spec("genes2genes") is None:
            raise ImportError(
                "genes2genes is required for genes2genes tuning and execution. "
                "Install it in the active Python environment before running benchmark tuning."
            )


def _sample_scdebussy_params(trial) -> dict[str, Any]:
    return {
        "n_bins": trial.suggest_int("n_bins", 50, 300),
        "bandwidth": trial.suggest_float("bandwidth", 0.01, 0.25),
        "gamma": trial.suggest_float("gamma", 0.05, 0.3),
        "max_iter": trial.suggest_int("max_iter", 10, 20),
        "dtw_dist_method": trial.suggest_categorical("dtw_dist_method", ["cosine", "euclidean"]),
        "dtw_step_pattern": trial.suggest_categorical(
            "dtw_step_pattern",
            ["asymmetric", "asymmetricP05", "asymmetricP1", "asymmetricP2"],
        ),
        "dtw_open_begin": trial.suggest_categorical("dtw_open_begin", [True, False]),
        "dtw_open_end": trial.suggest_categorical("dtw_open_end", [True, False]),
        "dtw_window_fraction": trial.suggest_float("dtw_window_fraction", 0.1, 0.5),
        "barycenter_init_iter": trial.suggest_int("barycenter_init_iter", 3, 8),
        "barycenter_update_iter": trial.suggest_int("barycenter_update_iter", 1, 3),
    }


def _sample_cellalign_params(trial) -> dict[str, Any]:
    return {
        "num_pts": trial.suggest_int("num_pts", 50, 300),
        "win_sz": trial.suggest_float("win_sz", 0.03, 0.25),
        "dist_method": trial.suggest_categorical("dist_method", ["Euclidean", "Correlation"]),
    }


def _sample_genes2genes_params(trial, *, include_reference: bool) -> dict[str, Any]:
    params: dict[str, Any] = {
        "n_bins": trial.suggest_int("n_bins", 30, 150),
        "gene_list_mode": trial.suggest_categorical("gene_list_mode", ["all_genes", "hvg"]),
    }
    if include_reference:
        params["reference_policy"] = trial.suggest_categorical("reference_policy", ["medoid"])
    return params


def _build_sampler(method: str):
    if method == "scdebussy":
        return _sample_scdebussy_params
    if method in {"cellalign_consensus", "cellalign_fixed_reference"}:
        return _sample_cellalign_params
    if method == "genes2genes_consensus":
        return lambda trial: _sample_genes2genes_params(trial, include_reference=False)
    if method == "genes2genes_fixed_reference":
        return lambda trial: _sample_genes2genes_params(trial, include_reference=True)
    raise ValueError(f"No tuning sampler defined for method={method!r}.")


def _unsupervised_objective_for_method(
    method: str,
    adata_run,
    *,
    aligned_key: str,
    method_result: dict[str, Any],
    compute_generic_unsupervised_metrics,
    compute_unsupervised_metrics,
    patient_key: str,
) -> tuple[float, dict[str, Any]]:
    generic = compute_generic_unsupervised_metrics(
        adata_run,
        key_added=aligned_key,
        patient_key=patient_key,
    )
    method_specific = dict(method_result.get("unsupervised_method", {}))

    score = float(generic.get("unsupervised_score_generic", np.inf))
    objective_name = "generic_unsupervised_score"

    if method == "scdebussy":
        barycenter_key = method_result.get("method_meta", {}).get("barycenter_key", "barycenter")
        method_specific = {
            **method_specific,
            **compute_unsupervised_metrics(
                adata_run,
                key_added=aligned_key,
                barycenter_key=barycenter_key,
                patient_key=patient_key,
            ),
        }
        score = float(method_specific.get("unsupervised_score", np.inf))
        objective_name = "scdebussy_unsupervised_score"

    metrics = {
        "objective_name": objective_name,
        "objective_score": score,
        **{f"generic_{k}": v for k, v in generic.items()},
        **{f"method_{k}": v for k, v in method_specific.items() if isinstance(v, (int, float, str, bool)) or v is None},
    }
    return score, metrics


def tune_method(
    method: str,
    adata,
    *,
    method_params: dict[str, Any] | None = None,
    n_trials: int = 30,
    seed: int = 42,
    objective: str = "unsupervised",
    verbose: bool = False,
) -> TuningResult:
    """Tune one supported method and return best parameters and sweep history.

    Parameters
    ----------
    method : str
        Method name as registered in benchmark adapters.
    adata : AnnData
        Input dataset.
    method_params : dict, optional
        Base kwargs for method execution. Search-space suggestions are merged on top.
    n_trials : int
        Optuna trial budget.
    seed : int
        Optuna sampler random seed.
    objective : str
        Currently supports only ``"unsupervised"``.
    verbose : bool
        If True, prints trial failures from Optuna objective.
    """
    if method not in _TUNABLE_METHODS:
        raise ValueError(f"Method {method!r} is not tunable. Supported: {sorted(_TUNABLE_METHODS)}")
    if objective != "unsupervised":
        raise ValueError("Only objective='unsupervised' is currently supported.")

    optuna, TPESampler, MedianPruner = _import_optuna()
    run_method, compute_generic_unsupervised_metrics, compute_unsupervised_metrics = _import_benchmark_runtime()
    _preflight_method_dependencies(method)

    base_params = dict(method_params or {})
    patient_key = str(base_params.get("patient_key", "patient"))
    sampler_fn = _build_sampler(method)

    sweep_rows: list[dict[str, Any]] = []
    best_score = float("inf")
    best_params: dict[str, Any] | None = None
    best_method_result: dict[str, Any] | None = None

    def _objective(trial):
        nonlocal best_score, best_params, best_method_result

        sampled = sampler_fn(trial)
        params = {**base_params, **sampled}

        try:
            adata_run = adata.copy()
            method_result = run_method(method, adata_run, params)
            if method_result.get("evaluation_mode") != "single_axis":
                raise ValueError("Only single_axis methods can be tuned with this objective.")

            aligned_key = method_result.get("aligned_key")
            if not aligned_key:
                raise ValueError("Adapter did not return a valid aligned_key for tuning objective.")

            score, metric_row = _unsupervised_objective_for_method(
                method,
                adata_run,
                aligned_key=aligned_key,
                method_result=method_result,
                compute_generic_unsupervised_metrics=compute_generic_unsupervised_metrics,
                compute_unsupervised_metrics=compute_unsupervised_metrics,
                patient_key=patient_key,
            )

            row = {
                "trial": int(trial.number + 1),
                **sampled,
                **metric_row,
                "status": "ok",
            }
            sweep_rows.append(row)

            if np.isfinite(score) and score < best_score:
                best_score = float(score)
                best_params = dict(params)
                best_method_result = {
                    "aligned_key": method_result.get("aligned_key"),
                    "method_meta": dict(method_result.get("method_meta", {})),
                    "method_params": dict(method_result.get("method_params", params)),
                    "objective_name": metric_row["objective_name"],
                }

            return float(score)
        except Exception as exc:  # noqa: BLE001
            sweep_rows.append(
                {
                    "trial": int(trial.number + 1),
                    **sampled,
                    "status": "error",
                    "error": str(exc),
                }
            )
            if verbose:
                print(f"{method} trial {trial.number + 1} failed: {exc}")
            return float("inf")

    sampler = TPESampler(seed=seed, n_startup_trials=min(10, max(1, n_trials // 3)))
    pruner = MedianPruner(n_startup_trials=min(5, max(1, n_trials // 6)), n_warmup_steps=0)
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="minimize")
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)

    sweep_df = pd.DataFrame(sweep_rows)
    if not sweep_df.empty and "objective_score" in sweep_df.columns:
        sweep_df = sweep_df.sort_values(["objective_score"], ascending=[True], na_position="last").reset_index(
            drop=True
        )

    successful = int(np.sum(np.isfinite(sweep_df.get("objective_score", pd.Series(dtype=float)).to_numpy(dtype=float))))
    if best_params is None or best_method_result is None or not np.isfinite(best_score):
        error_rows = [row for row in sweep_rows if row.get("status") == "error"]
        if error_rows:
            seen_errors: list[str] = []
            for row in error_rows:
                msg = str(row.get("error", "unknown error"))
                if msg not in seen_errors:
                    seen_errors.append(msg)
                if len(seen_errors) >= 3:
                    break
            raise RuntimeError(
                f"No successful tuning trial for method={method!r} after {n_trials} trials. "
                f"Example trial errors: {' | '.join(seen_errors)}"
            )

        raise RuntimeError(
            f"No successful tuning trial for method={method!r} after {n_trials} trials. "
            "All trials returned non-finite objective scores."
        )

    return TuningResult(
        method=method,
        objective=objective,
        n_trials=int(n_trials),
        n_successful_trials=successful,
        best_score=float(best_score),
        best_params=best_params,
        sweep_df=sweep_df,
        best_method_result=best_method_result,
    )


def tune_scdebussy(
    adata,
    *,
    method_params: dict[str, Any] | None = None,
    n_trials: int = 30,
    seed: int = 42,
    objective: str = "unsupervised",
    verbose: bool = False,
) -> TuningResult:
    """Tune scDeBussy with package-level defaults."""
    return tune_method(
        "scdebussy",
        adata,
        method_params=method_params,
        n_trials=n_trials,
        seed=seed,
        objective=objective,
        verbose=verbose,
    )


def tune_cellalign_consensus(
    adata,
    *,
    method_params: dict[str, Any] | None = None,
    n_trials: int = 30,
    seed: int = 42,
    objective: str = "unsupervised",
    verbose: bool = False,
) -> TuningResult:
    """Tune CellAlign consensus adapter parameters."""
    return tune_method(
        "cellalign_consensus",
        adata,
        method_params=method_params,
        n_trials=n_trials,
        seed=seed,
        objective=objective,
        verbose=verbose,
    )


def tune_cellalign_fixed_reference(
    adata,
    *,
    method_params: dict[str, Any] | None = None,
    n_trials: int = 30,
    seed: int = 42,
    objective: str = "unsupervised",
    verbose: bool = False,
) -> TuningResult:
    """Tune CellAlign fixed-reference adapter parameters."""
    return tune_method(
        "cellalign_fixed_reference",
        adata,
        method_params=method_params,
        n_trials=n_trials,
        seed=seed,
        objective=objective,
        verbose=verbose,
    )


def tune_genes2genes_consensus(
    adata,
    *,
    method_params: dict[str, Any] | None = None,
    n_trials: int = 30,
    seed: int = 42,
    objective: str = "unsupervised",
    verbose: bool = False,
) -> TuningResult:
    """Tune Genes2Genes consensus adapter parameters."""
    return tune_method(
        "genes2genes_consensus",
        adata,
        method_params=method_params,
        n_trials=n_trials,
        seed=seed,
        objective=objective,
        verbose=verbose,
    )


def tune_genes2genes_fixed_reference(
    adata,
    *,
    method_params: dict[str, Any] | None = None,
    n_trials: int = 30,
    seed: int = 42,
    objective: str = "unsupervised",
    verbose: bool = False,
) -> TuningResult:
    """Tune Genes2Genes fixed-reference adapter parameters."""
    return tune_method(
        "genes2genes_fixed_reference",
        adata,
        method_params=method_params,
        n_trials=n_trials,
        seed=seed,
        objective=objective,
        verbose=verbose,
    )


__all__ = [
    "TuningResult",
    "is_method_tunable",
    "tunable_methods",
    "tune_method",
    "tune_scdebussy",
    "tune_cellalign_consensus",
    "tune_cellalign_fixed_reference",
    "tune_genes2genes_consensus",
    "tune_genes2genes_fixed_reference",
]
