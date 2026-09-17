"""Reproducible, resumable validation of the experimental linear model.

Run with PYTHONPATH=src python scripts/v2/validate.py --suite smoke --output DIR.
The full suites are deliberately separate from unit tests and never tune on
held-out seeds. Reports retain failures and cannot silently promote v2.
"""

import argparse
import hashlib
import json
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

from scdebussy import align

SCENARIOS = {
    "null": {},
    "delay": {"warp": 0.6},
    "occupancy": {"occupancy": 2.0},
    "expression": {"expression": 0.8},
    "censoring": {"gap_depth": 1.0},
    "delay_censoring": {"warp": 0.6, "gap_depth": 1.0},
    "combined": {"warp": 0.6, "occupancy": 2.0, "expression": 0.8, "gap_depth": 1.0},
}
ROBUSTNESS = {
    "nonlinear_warp": {"warp": 1.5},
    "heavy_tailed": {"noise_model": "heavy_tailed"},
    "correlated": {"noise_model": "correlated"},
    "heteroskedastic": {"noise_model": "heteroskedastic"},
    "multimodal": {"multimodal": True},
    "reversal": {"reversal": True},
    **{
        f"panel_{panel}": {"gene_panel": panel}
        for panel in ("random", "high_variance", "low_variance", "duplicated", "nuisance", "informative")
    },
}


def run_case(name, cfg, fit_config, contract):
    """Run one independently seeded fixture and retain all diagnostics."""
    start = time.monotonic()
    row = {"case": name, "seed": cfg.seed, "config": cfg.__dict__, "fit_config": fit_config}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            data = align.simulate(cfg)
            result = align.fit(data, **fit_config)
            projection = align.project(result)
            summary = align.summarize(result, projection)
            error = np.abs(projection.position - data.obs.z_true.to_numpy())
            thresholds = contract["thresholds"]
            confident = (projection.unprojected <= thresholds["confidence_unprojected_max"]) & (
                projection.entropy <= thresholds["confidence_entropy_max"]
            )
            labels = data.obs.groupby("patient", sort=False)["condition"].first().to_dict()
            metadata = {}
            if cfg.design != "exchangeable":
                key = "block" if cfg.design == "blocked" else "pair"
                metadata["blocks" if cfg.design == "blocked" else "pairs"] = (
                    data.obs.groupby("patient")[key].first().to_dict()
                )
            comparison = align.compare(
                result, labels, scheme=cfg.design, seed=cfg.seed, n_perm=contract["n_perm"], **metadata
            )
            finite = np.isfinite(error)
            row.update(
                status="completed",
                converged=result.converged,
                iterations=len(result.history),
                rmse=float(np.sqrt(np.mean(error[finite] ** 2))) if finite.any() else None,
                finite_projection=float(finite.mean()),
                coverage=float(summary.coverage.mean()),
                confident_bad_fraction=float(np.mean(confident & (error > thresholds["bad_projection_error"]))),
                kinetic_difference=float(
                    np.nanmean(summary.kinetic_mean[1::2]) - np.nanmean(summary.kinetic_mean[::2])
                ),
                mean_gap=float(np.mean([p.deletion.mean() for p in result.posteriors])),
                mean_entropy=float(np.mean([p.entropy for p in result.posteriors])),
                max_no_match=float(max(p.no_match_probability for p in result.posteriors)),
                kinetic_p=float(comparison.kinetic.pvalue.iloc[0]),
                occupancy_p=float(comparison.occupancy.pvalue.iloc[0]),
                expression_q=comparison.expression.qvalue.to_dict(),
                significant_genes=comparison.expression.index[comparison.expression.qvalue < 0.05].tolist(),
                free_energy=result.history[-1]["free_energy"],
                provenance=result.provenance,
            )
        except (ValueError, ArithmeticError) as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        row["warnings"] = [str(item.message) for item in caught]
    row["seconds"] = time.monotonic() - start
    return row


def cases(suite, contract):
    """Yield prespecified cases without adapting to evaluation outcomes."""
    base = align.SimulationConfig()
    fit = contract["fit"]
    seeds = contract["heldout_seeds"]
    if suite == "smoke":
        for name, changes in {**SCENARIOS, **ROBUSTNESS}.items():
            yield name, replace(base, seed=0, cells_per_patient=60, **changes), dict(fit, n_landmarks=12, max_iter=30)
    elif suite in {"recovery", "robustness"}:
        scenarios = SCENARIOS if suite == "recovery" else ROBUSTNESS
        for seed in seeds:
            for name, changes in scenarios.items():
                for noise in (0.04, 0.12) if suite == "recovery" else (0.08,):
                    # Coverage is factored separately from the intended effect.
                    for gap in (0.0, 1.0) if suite == "recovery" else (0.0,):
                        settings = dict(changes, noise=noise)
                        settings["gap_depth"] = max(settings.get("gap_depth", 0.0), gap)
                        yield f"{name}_noise{noise}_gap{gap}", replace(base, seed=seed, **settings), dict(fit)
    elif suite == "null":
        for seed in range(contract["null_seed_start"], contract["null_seed_start"] + contract["null_replicates"]):
            for design in ("exchangeable", "blocked", "paired"):
                yield f"null_{design}", replace(base, seed=seed, design=design), dict(fit)
    elif suite == "calibration":
        for seed in contract["calibration_seeds"]:
            for noise in (0.04, 0.12):
                yield f"null_noise{noise}", replace(base, seed=seed, noise=noise), dict(fit)
    elif suite == "sensitivity":
        for seed in seeds[:3]:
            for parameter, values in contract["sensitivity"].items():
                for value in values:
                    config = dict(fit)
                    if parameter == "kappa":
                        config.update(kappa_i=value, kappa_d=value)
                    else:
                        config[parameter] = value
                    if parameter == "calibration_band":
                        config.pop("eta_i", None)
                        config.pop("eta_d", None)
                    yield f"{parameter}_{value}", replace(base, seed=seed, warp=0.6), config


def evaluate(rows, suite, contract):
    """Evaluate frozen thresholds; smoke success does not open a release gate."""
    thresholds = contract["thresholds"]
    failures = [f"{r['case']}/{r['seed']}: {r.get('error', 'failed')}" for r in rows if r["status"] != "completed"]
    if suite != "smoke":
        failures += [f"{r['case']}/{r['seed']}: did not converge" for r in rows if r.get("converged") is False]
    for row in rows:
        if row["status"] != "completed":
            continue
        if suite == "recovery":
            if row["rmse"] is None or row["rmse"] > thresholds["recovery_rmse_max"]:
                failures.append(f"{row['case']}/{row['seed']}: RMSE exceeds recovery threshold")
            if row["finite_projection"] < thresholds["recovery_finite_projection_min"]:
                failures.append(f"{row['case']}/{row['seed']}: inadequate finite projection")
        if (
            suite == "robustness"
            and row["confident_bad_fraction"] > thresholds["confident_bad_projection_fraction_max"]
        ):
            failures.append(f"{row['case']}/{row['seed']}: confidently incorrect projection")
    if suite == "null":
        for design in ("exchangeable", "blocked", "paired"):
            group = [r for r in rows if r["case"] == f"null_{design}" and r["status"] == "completed"]
            for family in ("kinetic_p", "occupancy_p", "expression_q"):
                values = [(min(r[family].values()) if family == "expression_q" else r[family]) for r in group]
                if (
                    not values
                    or not np.isfinite(values).all()
                    or np.mean(np.asarray(values) < 0.05) > thresholds["null_rejection_rate_max"]
                ):
                    failures.append(f"{design}/{family}: null calibration failed")
    if suite == "sensitivity":
        for seed in contract["heldout_seeds"][:3]:
            group = [
                r
                for r in rows
                if r["seed"] == seed and r["case"].startswith("n_landmarks_") and r.get("rmse") is not None
            ]
            if len(group) != 3 or np.ptp([r["rmse"] for r in group]) > thresholds["resolution_rmse_range_max"]:
                failures.append(f"{seed}: resolution sensitivity failed")
    return {
        "suite": suite,
        "cases": len(rows),
        "passed": not failures,
        "failures": failures,
        "release_gate": suite != "smoke",
        "replacement_recommended": False,
    }


def main():
    """Run or resume a suite under an unchanged contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=["smoke", "calibration", "recovery", "robustness", "null", "sensitivity", "all"],
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=Path(__file__).with_name("contract.json"))
    args = parser.parse_args()
    contract_text = args.contract.read_text()
    contract = json.loads(contract_text)
    fingerprint = hashlib.sha256(contract_text.encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    existing = args.output / "contract.json"
    if existing.exists() and existing.read_text() != contract_text:
        raise ValueError("output directory contains a different frozen contract")
    existing.write_text(contract_text)
    suites = ["calibration", "recovery", "robustness", "null", "sensitivity"] if args.suite == "all" else [args.suite]
    for suite in suites:
        rows = []
        for index, (name, cfg, fit_config) in enumerate(cases(suite, contract)):
            path = args.output / f"{suite}_{index:04d}.json"
            if path.exists():
                row = json.loads(path.read_text())
                if row.get("contract_sha256") != fingerprint:
                    raise ValueError("cached case belongs to a different contract")
            else:
                row = run_case(name, cfg, fit_config, contract)
                row["contract_sha256"] = fingerprint
                path.write_text(json.dumps(row, indent=2))
            rows.append(row)
            print(f"{suite} {index + 1}: {name} seed={cfg.seed} {row['status']} ({row['seconds']:.1f}s)", flush=True)
        report = evaluate(rows, suite, contract)
        (args.output / f"{suite}_summary.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
