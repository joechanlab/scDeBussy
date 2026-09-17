"""Aggregate raw validation records without changing frozen acceptance gates."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def aggregate(directory):
    """Produce compact review tables and descriptive recovery diagnostics."""
    contract = json.loads((directory / "contract.json").read_text())
    fingerprint = hashlib.sha256((directory / "contract.json").read_bytes()).hexdigest()
    records = []
    for path in sorted(directory.glob("*_[0-9][0-9][0-9][0-9].json")):
        row = json.loads(path.read_text())
        if row["contract_sha256"] != fingerprint:
            raise ValueError(f"contract mismatch: {path}")
        row["suite"] = path.stem.rsplit("_", 1)[0]
        records.append(row)
    columns = [
        "suite",
        "case",
        "seed",
        "status",
        "converged",
        "iterations",
        "seconds",
        "rmse",
        "finite_projection",
        "coverage",
        "kinetic_difference",
        "mean_gap",
        "mean_entropy",
        "max_no_match",
        "kinetic_p",
        "occupancy_p",
        "confident_bad_fraction",
    ]
    table = pd.DataFrame([{key: row.get(key) for key in columns} for row in records], columns=columns)
    table.to_csv(directory / "metrics.csv", index=False)
    calibration = [r for r in records if r["suite"] == "calibration" and r["status"] == "completed"]
    thresholds = {}
    for noise in sorted({r["config"]["noise"] for r in calibration}):
        values = [abs(r["kinetic_difference"]) for r in calibration if r["config"]["noise"] == noise]
        thresholds[str(noise)] = float(np.quantile(values, 0.95, method="linear"))
    recovery = []
    groups = {}
    for row in records:
        if row["suite"] == "recovery" and row["status"] == "completed":
            groups.setdefault(row["case"], []).append(row)
    for name, rows in groups.items():
        config = rows[0]["config"]
        threshold = thresholds.get(str(config["noise"]))
        changed_genes = [f"g{i}" for i in range(max(1, config["n_genes"] // 3))]
        actual_effect = bool(config["expression"])
        expected = set(changed_genes) if actual_effect else set()
        discovered = [set(r["significant_genes"]) for r in rows]
        recovery.append(
            {
                "case": name,
                "replicates": len(rows),
                "warp_truth": config["warp"],
                "occupancy_truth": config["occupancy"],
                "expression_truth": config["expression"],
                "censoring_truth": config["gap_depth"],
                "kinetic_null_q95": threshold,
                "kinetic_exceeds_matched_noise_null": float(
                    np.mean([abs(r["kinetic_difference"]) > threshold for r in rows])
                )
                if threshold is not None
                else None,
                "kinetic_direction_positive_fraction": float(np.mean([r["kinetic_difference"] > 0 for r in rows])),
                "kinetic_detection_rate": float(np.mean([r["kinetic_p"] < 0.05 for r in rows])),
                "occupancy_detection_rate": float(np.mean([r["occupancy_p"] < 0.05 for r in rows])),
                "expression_gene_recall": float(np.mean([len(s & expected) / len(expected) for s in discovered]))
                if expected
                else None,
                "expression_mean_false_discovery_proportion": float(
                    np.mean([len(s - expected) / max(1, len(s)) for s in discovered])
                ),
                "mean_rmse": float(np.mean([r["rmse"] for r in rows])),
                "mean_gap": float(np.mean([r["mean_gap"] for r in rows])),
            }
        )
    resolution = []
    for seed in contract["heldout_seeds"][:3]:
        rows = [
            r
            for r in records
            if r["suite"] == "sensitivity"
            and r["seed"] == seed
            and r["case"].startswith("n_landmarks_")
            and r["status"] == "completed"
        ]
        if not rows:
            continue
        signs = [int(np.sign(r["kinetic_difference"])) for r in rows]
        sets = [set(r["significant_genes"]) for r in rows]
        union = set.union(*sets)
        resolution.append(
            {
                "seed": seed,
                "resolutions": [r["fit_config"]["n_landmarks"] for r in rows],
                "kinetic_signs_agree": len(set(signs)) == 1,
                "significant_gene_intersection_over_union": len(set.intersection(*sets)) / len(union)
                if union
                else None,
                "no_significant_genes_at_any_resolution": not union,
                "rmse_range": float(np.ptp([r["rmse"] for r in rows])),
                "mean_gap_range": float(np.ptp([r["mean_gap"] for r in rows])),
            }
        )
    report = {
        "contract_sha256": fingerprint,
        "descriptive_only": True,
        "note": "No post-hoc gate thresholds were introduced. Gap mass does not identify censoring.",
        "kinetic_null_thresholds": thresholds,
        "recovery": recovery,
        "resolution": resolution,
    }
    (directory / "decomposition.json").write_text(json.dumps(report, indent=2))
    return len(records)


def main():
    """Aggregate a directory after its suites have finished."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(f"Aggregated {aggregate(args.directory)} cases")


if __name__ == "__main__":
    main()
