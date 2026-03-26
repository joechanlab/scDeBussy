"""CLI runner for a single benchmark (scenario, seed) pair.

Usage
-----
    python scripts/benchmark/run_scenario.py \
        --scenario warp_medium \
        --method scdebussy \
        --seed 3 \
        --output_dir results/ \
        [--method_params '{"gamma": 0.05}']

The script writes one JSON file per invocation:
    <output_dir>/<scenario_name>/seed_<seed>.json

Results include supervised metrics (requires tau_global ground-truth),
unsupervised metrics (no ground-truth needed), and runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np

import scdebussy.tl as tl

# ---------------------------------------------------------------------------
# Ensure the package is importable when running from the repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

# ---------------------------------------------------------------------------
# Pseudotime grid shared across all scenarios
# ---------------------------------------------------------------------------
_T = 50
_TIME_GRID = np.linspace(0.0, 1.0, _T)


def _run_one(
    scenario_name: str,
    seed: int,
    *,
    method: str,
    method_overrides: dict | None = None,
) -> dict:
    from scripts.benchmark.methods import run_method
    from scripts.benchmark.metrics import (
        compute_generic_unsupervised_metrics,
        compute_unsupervised_metrics,
        evaluate_run,
    )
    from scripts.benchmark.scenarios import SCENARIO_REGISTRY

    sc = SCENARIO_REGISTRY[scenario_name]
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # 1.  Build simulation kwargs                                          #
    # ------------------------------------------------------------------ #
    sim_kwargs = dict(sc["sim_kwargs"])
    sim_kwargs["time_grid"] = _TIME_GRID  # fill the None placeholder

    # Initialise gene loadings with this run's seed
    K = sim_kwargs["K"]
    M = sim_kwargs["M"]
    W, gene_categories = tl.initialize_structured_loadings(K=K, M=M, rng=rng)
    sim_kwargs["W"] = W
    sim_kwargs["gene_categories"] = gene_categories
    sim_kwargs["rng"] = rng

    # ------------------------------------------------------------------ #
    # 2.  Simulate dataset                                                 #
    # ------------------------------------------------------------------ #
    adata = tl.simulate_LF_MOGP(**sim_kwargs)

    # ------------------------------------------------------------------ #
    # 3.  Build method kwargs                                              #
    # ------------------------------------------------------------------ #
    if method == "scdebussy":
        method_kwargs = dict(sc.get("scdebussy_params", {}))
    else:
        method_kwargs = dict(sc.get("method_params", {}).get(method, {}))

    if method_overrides:
        method_kwargs.update(method_overrides)

    patient_key = method_kwargs.get("patient_key", "patient")
    baseline_key = method_kwargs.get("pseudotime_key", "s_local")

    # ------------------------------------------------------------------ #
    # 4.  Run selected method and measure wall-clock time                  #
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
    # 5.  Evaluate                                                         #
    # ------------------------------------------------------------------ #
    baseline_only = evaluate_run(
        adata,
        key_added=baseline_key,
        baseline_key=baseline_key,
        truth_key="tau_global",
        patient_key=patient_key,
    )

    if evaluation_mode == "single_axis":
        supervised = evaluate_run(
            adata,
            key_added=key_added,
            baseline_key=baseline_key,
            truth_key="tau_global",
            patient_key=patient_key,
        )

        generic_unsupervised = compute_generic_unsupervised_metrics(
            adata,
            key_added=key_added,
            patient_key=patient_key,
        )
    else:
        supervised = {
            "baseline": baseline_only["baseline"],
            "aligned": {},
        }
        generic_unsupervised = {}

    # Keep scDeBussy-specific diagnostics when available.
    if method == "scdebussy" and evaluation_mode == "single_axis":
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

    # Backward-compatible aggregate field used by existing aggregator.
    unsupervised_combined = {
        **generic_unsupervised,
        **{f"method_{k}": v for k, v in unsupervised_method.items()},
    }

    # ------------------------------------------------------------------ #
    # 6.  Serialise sim_params (drop non-JSON-serialisable objects)        #
    # ------------------------------------------------------------------ #
    sim_params_out = {
        k: v
        for k, v in sim_kwargs.items()
        if k not in {"W", "gene_categories", "rng", "time_grid", "factor_kernels", "deviation_kernel"}
    }
    # warp_types is a list of str — fine; noise_settings is a dict — fine
    # lambda_cells may be a list or scalar — both fine
    for k, v in sim_params_out.items():
        if not isinstance(v, (int, float, str, bool, list, dict, type(None))):
            sim_params_out[k] = str(v)

    method_params_out = {
        k: v for k, v in method_params_out.items() if isinstance(v, (int, float, str, bool, list, dict, type(None)))
    }

    result = {
        "method": method,
        "evaluation_mode": evaluation_mode,
        "scenario": scenario_name,
        "seed": seed,
        "runtime_s": runtime_s,
        "sim_params": sim_params_out,
        "method_params": method_params_out,
        "method_meta": method_meta,
        "baseline": baseline_only["baseline"],
        "aligned": supervised["aligned"],
        "unsupervised": unsupervised_combined,
        "unsupervised_generic": generic_unsupervised,
        "unsupervised_method": unsupervised_method,
    }
    if method == "scdebussy":
        result["scdebussy_params"] = dict(method_params_out)
    return result


def _write_result(result: dict, output_dir: str) -> str:
    """Write result JSON atomically; return the final path."""
    method = result.get("method", "unknown")
    scenario = result["scenario"]
    seed = result["seed"]
    dest_dir = os.path.join(output_dir, str(method), scenario)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, f"seed_{seed}.json")

    # Atomic write via temp file + rename
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
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def main():
    """
    Run a single benchmark scenario for a specified alignment method and output results.

    This function parses command-line arguments to configure and execute a benchmark scenario
    using a selected alignment method. It supports method parameter overrides via JSON and
    handles deprecated parameter aliases. Results are written to a JSON file and a summary
    is printed to stdout.

    Command-line Arguments:
        --method (str): Registered alignment method adapter to run.
            Default: "scdebussy". Must be one of available_methods().
        --scenario (str): Name of the scenario to run. Required.
            Must be one of the registered scenarios in SCENARIO_REGISTRY.
        --seed (int): Integer random seed for reproducibility. Required.
        --output_dir (str): Root directory where results are written. Required.
        --method_params (str): Optional JSON string of method parameter overrides.
        --scdebussy_params (str): Deprecated alias for --method_params when --method=scdebussy.
            Cannot be used simultaneously with --method_params.

    Raises
    ------
        SystemExit: If required arguments are missing, invalid choices are provided,
            or incompatible argument combinations are used.
        json.JSONDecodeError: If --method_params or --scdebussy_params contains invalid JSON.

    Returns
    -------
        None. Writes result JSON file to output_dir and prints summary to stdout.
    """
    from scripts.benchmark.methods import available_methods
    from scripts.benchmark.scenarios import SCENARIO_REGISTRY

    parser = argparse.ArgumentParser(
        description="Run one benchmark scenario for a selected method and write a JSON result file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--method", default="scdebussy", choices=available_methods(), help="Registered alignment method adapter to run."
    )
    parser.add_argument(
        "--scenario", required=True, choices=sorted(SCENARIO_REGISTRY), help="Name of the scenario to run."
    )
    parser.add_argument("--seed", required=True, type=int, help="Integer random seed for reproducibility.")
    parser.add_argument("--output_dir", required=True, help="Root directory where results are written.")
    parser.add_argument("--method_params", default=None, help="Optional JSON string of method parameter overrides.")
    parser.add_argument(
        "--scdebussy_params", default=None, help="Deprecated alias for --method_params when --method=scdebussy."
    )
    args = parser.parse_args()

    overrides = None
    raw_override_json = args.method_params
    if args.scdebussy_params:
        if args.method != "scdebussy":
            parser.error("--scdebussy_params can only be used when --method=scdebussy")
        if args.method_params:
            parser.error("Use either --method_params or --scdebussy_params, not both")
        raw_override_json = args.scdebussy_params

    if raw_override_json:
        try:
            overrides = json.loads(raw_override_json)
        except json.JSONDecodeError as exc:
            parser.error(f"Method params JSON is not valid: {exc}")

    result = _run_one(
        args.scenario,
        args.seed,
        method=args.method,
        method_overrides=overrides,
    )
    out_path = _write_result(result, args.output_dir)

    if result.get("evaluation_mode") == "single_axis":
        n_imp = result["aligned"].get("n_patients_improved", "?")
        g_r = result["aligned"].get("global_pearson_r", float("nan"))
        print(
            f"[OK] {args.method}/{args.scenario}/seed_{args.seed}  "
            f"Pearson={g_r:.4f}  n_improved={n_imp}  "
            f"runtime={result['runtime_s']:.1f}s  → {out_path}"
        )
    else:
        pair_success_rate = 1.0 - float(result["unsupervised_method"].get("fraction_failed_pairs", 0.0))
        print(
            f"[OK] {args.method}/{args.scenario}/seed_{args.seed}  "
            f"pairwise_only=1  pair_success_rate={pair_success_rate:.4f}  "
            f"runtime={result['runtime_s']:.1f}s  → {out_path}"
        )


if __name__ == "__main__":
    main()
