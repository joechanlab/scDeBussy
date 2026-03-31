"""Standalone benchmark runner for real-world datasets (no ground-truth pseudotime).

Usage
-----
    python scripts/benchmark/run_real_data.py \
        --adata /path/to/adata.h5ad \
        --output-dir results/real_data/ \
        --method scdebussy \
        --patient-key patient \
        --pseudotime-key s_local \
        --cell-type-key cell_type \
        [--method-params '{"gamma": 0.05}'] \
        [--seed 1] \
        [--tune auto] \
        [--tune-trials 30] \
        [--purity-n-neighbors 50] \
        [--purity-sweep]

The script writes one JSON file per invocation:
    <output_dir>/<method>/real_data/seed_<seed>.json

The JSON schema is compatible with ``aggregate_results.py`` and
``compare_methods.py``.  Supervised fields (``aligned.global_pearson_r``,
``baseline.global_pearson_r``, etc.) are absent; a ``scenario: "real_data"``
marker and a ``real_data_metrics`` block with purity results are added instead.

python scripts/benchmark/run_real_data.py \
  --adata /data/nsclc_pt1.h5ad /data/sclc_pt2.h5ad \
  --cell-type-pattern 'NSCLC|SCLC' \
  --cell-type-key cell_type_final2 \
  --output-dir /scratch/chanj3/wangm10/HTAN \
  --method scdebussy
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time

import numpy as np

# ---------------------------------------------------------------------------
# Make the package importable from the repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")

for _p in (_REPO_ROOT, _SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_and_concat(
    adata_paths: list[str],
    patient_key: str,
    patient_labels: list[str] | None = None,
):
    """Load one or more h5ad files and return a single concatenated AnnData.

    Parameters
    ----------
    adata_paths:
        List of absolute paths to ``.h5ad`` files.
    patient_key:
        Name of the ``.obs`` column that encodes the patient / batch label.
        If the column is absent in an individual file the patient label is
        derived from ``patient_labels`` (if provided) or from the filename
        stem, then written into the concatenated ``.obs``.
    patient_labels:
        Explicit labels to assign to each file, in the same order as
        ``adata_paths``.  Required length must match ``adata_paths``.
        Overrides filename-stem inference.

    Returns
    -------
    anndata.AnnData
        Single concatenated AnnData with ``patient_key`` present in ``.obs``.
    """
    import anndata

    if len(adata_paths) == 1:
        adata = anndata.read_h5ad(adata_paths[0])
        if patient_key not in adata.obs.columns:
            label = (
                patient_labels[0]
                if patient_labels is not None
                else os.path.splitext(os.path.basename(adata_paths[0]))[0]
            )
            adata.obs[patient_key] = label
        return adata

    if patient_labels is not None and len(patient_labels) != len(adata_paths):
        raise ValueError(
            f"--patient-labels has {len(patient_labels)} entries but --adata has {len(adata_paths)} paths."
        )

    adatas = []
    for i, path in enumerate(adata_paths):
        adata_i = anndata.read_h5ad(path)
        if patient_labels is not None:
            label = patient_labels[i]
        elif patient_key in adata_i.obs.columns:
            label = None  # keep existing column values
        else:
            label = os.path.splitext(os.path.basename(path))[0]

        if label is not None:
            adata_i.obs[patient_key] = label

        adatas.append(adata_i)

    combined = anndata.concat(adatas, join="inner", merge="same")
    combined.obs_names_make_unique()
    return combined


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def _run_one(
    adata_paths: list[str],
    seed: int,
    *,
    method: str,
    patient_key: str,
    pseudotime_key: str,
    cell_type_key: str,
    patient_labels: list[str] | None = None,
    cell_type_pattern: str | None = None,
    method_overrides: dict | None = None,
    tune: str = "auto",
    tune_trials: int = 30,
    tune_seed: int = 42,
    purity_n_neighbors: int = 50,
    purity_sweep: bool = False,
) -> dict:
    """Execute one method on a real-data AnnData and return a serialisable result dict."""
    from scripts.benchmark.methods import run_method
    from scripts.benchmark.metrics import (
        compute_generic_unsupervised_metrics,
        compute_real_data_label_metrics,
        compute_unsupervised_metrics,
    )

    from scdebussy import is_method_tunable, tune_method

    # ------------------------------------------------------------------ #
    # 1.  Load and (optionally) concatenate data                           #
    # ------------------------------------------------------------------ #
    adata = _load_and_concat(adata_paths, patient_key, patient_labels)

    # Validate required obs columns (patient_key is guaranteed by _load_and_concat)
    required_obs = {pseudotime_key, cell_type_key}
    missing = required_obs - set(adata.obs.columns)
    if missing:
        raise ValueError(
            f"AnnData is missing required obs columns: {sorted(missing)}. "
            f"Available columns: {sorted(adata.obs.columns)}"
        )

    # ------------------------------------------------------------------ #
    # 1b. Optional cell-type filter                                        #
    # ------------------------------------------------------------------ #
    n_cells_before_filter = int(adata.n_obs)
    if cell_type_pattern is not None:
        try:
            compiled = re.compile(cell_type_pattern)
        except re.error as exc:
            raise ValueError(f"--cell-type-pattern {cell_type_pattern!r} is not a valid regex: {exc}") from exc
        mask = adata.obs[cell_type_key].str.contains(compiled, regex=True, na=False)
        n_kept = int(mask.sum())
        if n_kept == 0:
            raise ValueError(
                f"Cell-type filter {cell_type_pattern!r} removed all {n_cells_before_filter} "
                f"cells.  Check the pattern against unique values: "
                f"{sorted(adata.obs[cell_type_key].unique())[:20]}"
            )
        adata = adata[mask].copy()

    # ------------------------------------------------------------------ #
    # 2.  Build method kwargs                                              #
    # ------------------------------------------------------------------ #
    method_kwargs: dict = {}
    if method_overrides:
        method_kwargs.update(method_overrides)

    # Inject obs key names so adapters can pick them up
    method_kwargs.setdefault("patient_key", patient_key)
    method_kwargs.setdefault("pseudotime_key", pseudotime_key)

    # ------------------------------------------------------------------ #
    # 3.  Optional tuning                                                  #
    # ------------------------------------------------------------------ #
    tuning_info: dict = {"enabled": False, "mode": tune}

    if tune not in {"off", "auto", "force"}:
        raise ValueError("tune must be one of {'off', 'auto', 'force'}.")

    should_tune = tune in {"auto", "force"} and is_method_tunable(method)
    if tune == "force" and not is_method_tunable(method):
        raise ValueError(f"Method {method!r} does not expose a formal package tuning API.")

    if should_tune:
        try:
            tuning_result = tune_method(
                method,
                adata,
                method_params=method_kwargs,
                n_trials=tune_trials,
                seed=tune_seed,
                objective="unsupervised",
                verbose=False,
            )
            method_kwargs = dict(tuning_result.best_params)
            tuning_info = {
                "enabled": True,
                "mode": tune,
                "objective": tuning_result.objective,
                "n_trials": tuning_result.n_trials,
                "n_successful_trials": tuning_result.n_successful_trials,
                "best_score": tuning_result.best_score,
                "best_params": dict(tuning_result.best_params),
            }
        except Exception as exc:
            if tune == "force":
                raise
            tuning_info = {
                "enabled": False,
                "mode": tune,
                "error": str(exc),
                "fallback": "untuned_method_params",
            }

    # ------------------------------------------------------------------ #
    # 4.  Run method                                                       #
    # ------------------------------------------------------------------ #
    t0 = time.perf_counter()
    method_result = run_method(method, adata, method_kwargs)
    runtime_s = time.perf_counter() - t0

    key_added = method_result["aligned_key"]
    evaluation_mode = method_result.get("evaluation_mode", "single_axis")
    method_meta = dict(method_result.get("method_meta", {}))
    unsupervised_method = dict(method_result.get("unsupervised_method", {}))
    method_params_out = dict(method_result.get("method_params", method_kwargs))

    # ------------------------------------------------------------------ #
    # 5.  Compute metrics                                                  #
    # ------------------------------------------------------------------ #
    generic_unsupervised: dict = {}
    if evaluation_mode == "single_axis" and key_added is not None:
        generic_unsupervised = compute_generic_unsupervised_metrics(
            adata,
            key_added=key_added,
            patient_key=patient_key,
        )

        # scDeBussy-specific barycenter metrics when available
        if method == "scdebussy":
            barycenter_key = method_meta.get("barycenter_key", method_kwargs.get("barycenter_key", "barycenter"))
            try:
                unsupervised_method = {
                    **unsupervised_method,
                    **compute_unsupervised_metrics(
                        adata,
                        key_added=key_added,
                        barycenter_key=barycenter_key,
                        patient_key=patient_key,
                    ),
                }
            except Exception as exc:  # noqa: BLE001
                unsupervised_method = {**unsupervised_method, "error": str(exc)}

    # Real-data label-informed purity metrics
    real_data_metrics: dict = {}
    if evaluation_mode == "single_axis" and key_added is not None:
        try:
            real_data_metrics = compute_real_data_label_metrics(
                adata,
                key_added=key_added,
                patient_key=patient_key,
                label_key=cell_type_key,
                purity_n_neighbors=purity_n_neighbors,
                compute_purity_sweep=purity_sweep,
            )
        except Exception as exc:  # noqa: BLE001
            real_data_metrics = {"error": str(exc)}

    # Aggregate unsupervised block (same shape as run_scenario.py)
    unsupervised_combined = {
        **generic_unsupervised,
        **{f"method_{k}": v for k, v in unsupervised_method.items()},
    }

    # ------------------------------------------------------------------ #
    # 6.  Sanitise for JSON                                                #
    # ------------------------------------------------------------------ #
    method_params_out = {
        k: v for k, v in method_params_out.items() if isinstance(v, (int, float, str, bool, list, dict, type(None)))
    }

    return {
        "method": method,
        "evaluation_mode": evaluation_mode,
        "scenario": "real_data",
        "adata_paths": adata_paths,
        "seed": seed,
        "runtime_s": runtime_s,
        "method_params": method_params_out,
        "method_meta": method_meta,
        # No supervised "baseline"/"aligned" blocks — real data has no tau_global.
        "unsupervised": unsupervised_combined,
        "unsupervised_generic": generic_unsupervised,
        "unsupervised_method": unsupervised_method,
        "real_data_metrics": real_data_metrics,
        "tuning": tuning_info,
        "obs_keys": {
            "patient_key": patient_key,
            "pseudotime_key": pseudotime_key,
            "cell_type_key": cell_type_key,
        },
        "cell_type_filter": {
            "pattern": cell_type_pattern,
            "n_cells_before": n_cells_before_filter,
            "n_cells_after": int(adata.n_obs),
        },
    }


def _write_result(result: dict, output_dir: str) -> str:
    """Write result JSON atomically; return the final path."""
    method = result.get("method", "unknown")
    scenario = result.get("scenario", "real_data")
    seed = result["seed"]
    dest_dir = os.path.join(output_dir, str(method), scenario)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, f"seed_{seed}.json")

    fd, tmp_path = tempfile.mkstemp(dir=dest_dir, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(result, fh, indent=2, default=_json_default)
        os.replace(tmp_path, dest_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return dest_path


def _json_default(obj):
    """Fallback JSON serialiser for numpy scalars."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def main():
    """Run a single real-data benchmark for a specified method and write results to JSON."""
    from scripts.benchmark.methods import available_methods

    parser = argparse.ArgumentParser(
        description=("Run one real-data benchmark for a selected alignment method and write a JSON result file."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--adata",
        required=True,
        nargs="+",
        metavar="ADATA",
        help=(
            "Path(s) to preprocessed AnnData .h5ad file(s).  Supply a single "
            "pre-combined file or one file per patient — they will be "
            "concatenated automatically via anndata.concat()."
        ),
    )
    parser.add_argument(
        "--patient-labels",
        nargs="+",
        default=None,
        dest="patient_labels",
        metavar="LABEL",
        help=(
            "Explicit patient labels assigned to each file in --adata order.  "
            "Only used when individual files lack a --patient-key column.  "
            "Must have the same number of entries as --adata paths."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        dest="output_dir",
        help="Root directory where results are written.",
    )
    parser.add_argument(
        "--method",
        default="scdebussy",
        choices=available_methods(),
        help="Registered alignment method adapter to run.",
    )
    parser.add_argument(
        "--seed",
        default=1,
        type=int,
        help="Integer seed for reproducibility (used for stochastic methods).",
    )
    parser.add_argument(
        "--patient-key",
        default="patient",
        dest="patient_key",
        help="adata.obs column for patient / batch labels.",
    )
    parser.add_argument(
        "--pseudotime-key",
        default="s_local",
        dest="pseudotime_key",
        help="adata.obs column for per-patient observed pseudotime.",
    )
    parser.add_argument(
        "--cell-type-key",
        default="cell_type",
        dest="cell_type_key",
        help="adata.obs column for cell-type labels (used for purity metrics).",
    )
    parser.add_argument(
        "--cell-type-pattern",
        default=None,
        dest="cell_type_pattern",
        help=(
            "Python regex pattern (re.search) applied to the cell-type column "
            "to keep only matching cells before alignment and metric computation.  "
            "E.g. 'NSCLC|SCLC' or '^AT2' or '(?i)epithelial'."
        ),
    )
    parser.add_argument(
        "--method-params",
        default=None,
        dest="method_params",
        help="Optional JSON string of method parameter overrides.",
    )
    parser.add_argument(
        "--tune",
        default="auto",
        choices=["off", "auto", "force"],
        help="Tuning mode: off=never tune, auto=tune if supported, force=error if not tunable.",
    )
    parser.add_argument(
        "--tune-trials",
        default=30,
        type=int,
        dest="tune_trials",
        help="Number of Optuna trials per tuned run.",
    )
    parser.add_argument(
        "--tune-seed",
        default=42,
        type=int,
        dest="tune_seed",
        help="Optuna sampler random seed.",
    )
    parser.add_argument(
        "--purity-n-neighbors",
        default=50,
        type=int,
        dest="purity_n_neighbors",
        help="k for cross-batch KNN purity computation.",
    )
    parser.add_argument(
        "--purity-sweep",
        action="store_true",
        dest="purity_sweep",
        help="Also compute purity at multiple k values and report sweep summary statistics.",
    )

    args = parser.parse_args()

    overrides = None
    if args.method_params is not None:
        try:
            overrides = json.loads(args.method_params)
        except json.JSONDecodeError as exc:
            parser.error(f"--method-params contains invalid JSON: {exc}")

    result = _run_one(
        args.adata,
        args.seed,
        method=args.method,
        patient_key=args.patient_key,
        pseudotime_key=args.pseudotime_key,
        cell_type_key=args.cell_type_key,
        patient_labels=args.patient_labels,
        cell_type_pattern=args.cell_type_pattern,
        method_overrides=overrides,
        tune=args.tune,
        tune_trials=args.tune_trials,
        tune_seed=args.tune_seed,
        purity_n_neighbors=args.purity_n_neighbors,
        purity_sweep=args.purity_sweep,
    )

    dest = _write_result(result, args.output_dir)
    print(f"Result written to: {dest}")

    # Print a brief human-readable summary
    rd = result.get("real_data_metrics", {})
    unsup = result.get("unsupervised_generic", {})
    print(
        f"  method={result['method']}  seed={result['seed']}  "
        f"runtime={result['runtime_s']:.1f}s\n"
        f"  unsupervised_score_generic={unsup.get('unsupervised_score_generic', float('nan')):.4f}  "
        f"global_cross_batch_purity={rd.get('global_cross_batch_purity', float('nan')):.4f}"
    )


if __name__ == "__main__":
    main()
