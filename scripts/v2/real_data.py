"""Evaluate the frozen v1 and experimental v2 on the same preprocessed AnnData.

No filtering, scaling, gene selection, or subsampling is performed here. Both
methods receive exactly the same X and cells. Baseline reproduction is a gate,
not an assumption inferred from a filename or a previously saved alignment.
"""

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import anndata as ad
import numpy as np

from scdebussy import align, tl


def metrics(data, key, n_bins):
    """Use the same legacy coverage and kNN purity functions for both methods."""
    data.uns["v2_evaluation_grid"] = {"expression": np.zeros((n_bins, 1))}
    coverage = tl.compute_patient_coverage(
        data, patient_key="patient", aligned_pseudotime_key=key, barycenter_key="v2_evaluation_grid"
    )
    purity = tl.cross_batch_knn_purity(
        data, pseudotime_key=key, label_key="cell_type_final2", batch_key="patient", n_neighbors=50, inplace=False
    )
    return {
        "coverage": float(coverage.mean()),
        "purity": purity["global_cross_batch_purity"],
        "finite_projection": float(np.isfinite(data.obs[key]).mean()),
        "mean_span": float(data.obs.groupby("patient", observed=True)[key].agg(lambda s: s.max() - s.min()).mean()),
    }


def evaluate(path, fit_config, contract):
    """Run both algorithms and report scientific gates without promoting v2."""
    started = time.monotonic()
    data = ad.read_h5ad(path)
    if data.obs.patient.nunique() != 9:
        raise ValueError("the replacement gate requires the specified nine-patient cohort")
    v1_options = {
        "patient_key": "patient",
        "pseudotime_key": "s_local",
        "n_bins": fit_config["n_landmarks"],
        "bandwidth": 0.1,
        "gamma": 0.1,
        "max_iter": 20,
        "tol": 1e-4,
        "use_soft_em": True,
        "strict_em": False,
        "verbose": False,
        "key_added": "v2_evaluation_v1",
        "barycenter_key": "v2_evaluation_v1_barycenter",
    }
    # Only a private in-memory object is mutated by the legacy API.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.scDeBussy(data, **v1_options)
        baseline = metrics(data, "v2_evaluation_v1", fit_config["n_landmarks"])
        print(f"v1 baseline: {baseline}", flush=True)
        result = align.fit(data, **fit_config)
        projection = align.project(result)
        data.obs["v2_evaluation_v2"] = projection.position
        candidate = metrics(data, "v2_evaluation_v2", fit_config["n_landmarks"])
        supported = data.copy()
        supported.obs.loc[projection.status == "unsupported", "v2_evaluation_v2"] = np.nan
        candidate_supported = metrics(supported, "v2_evaluation_v2", fit_config["n_landmarks"])
    limits = contract["thresholds"]
    reproduced = (
        abs(baseline["coverage"] - limits["baseline_coverage_reference"]) <= limits["real_coverage_tolerance"]
        and abs(baseline["purity"] - limits["baseline_purity_reference"]) <= 0.02
    )
    passed = (
        reproduced
        and result.converged
        and abs(candidate["coverage"] - baseline["coverage"]) <= limits["real_coverage_tolerance"]
        and candidate["purity"] >= limits["real_purity_min"]
    )
    return {
        "status": "completed",
        "input": str(path),
        "n_cells": data.n_obs,
        "n_genes": data.n_vars,
        "patients": list(result.patients),
        "preprocessing": "unchanged input X, cells, and gene order",
        "v1_config": v1_options,
        "v2_config": result.config,
        "baseline": baseline,
        "candidate": candidate,
        "candidate_supported_only": candidate_supported,
        "baseline_reproduced": reproduced,
        "gate_d_numeric_pass": passed,
        "replacement_recommended": False,
        "replacement_note": "Gates A-C and baseline reproduction are also required",
        "v2_converged": result.converged,
        "v2_iterations": len(result.history),
        "v2_coverage_mass": float(align.summarize(result, projection).coverage.mean()),
        "v2_diagnostics": result.diagnostics,
        "provenance": result.provenance,
        "warnings": [str(item.message) for item in caught],
        "seconds": time.monotonic() - started,
    }


def main():
    """Run the real-data gate with explicit scientific configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract_path = Path(__file__).with_name("contract.json")
    contract = json.loads(contract_path.read_text())
    config = json.loads(args.config.read_text())
    if args.output.exists():
        raise ValueError("choose a new output path; previous evaluations are not overwritten")
    report = evaluate(args.adata, config, contract)
    report["contract_sha256"] = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
