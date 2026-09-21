"""Build the machine-readable run table for the joint-warp Phase 4 matrix.

Joins the six joint-warp candidates against four comparison baselines (raw
s_local, original v2, trajectory-250 restricted, moderate DTW prototype),
reusing summarize_and_plot.score() for the standard metric set and adding the
posterior-mass diagnostics that only joint-mode candidates carry.
"""

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from summarize_and_plot import score


def joint_candidate_row(name, cache_path, report_path):
    """Summarize one joint-warp candidate's posterior mass and diagnostics."""
    report = json.loads(report_path.read_text())
    diag = report["diagnostics"]
    e11 = sum(v["expected_match_11"] for v in diag.values())
    e12 = sum(v["expected_match_12"] for v in diag.values())
    e21 = sum(v["expected_match_21"] for v in diag.values())
    internal_u = sum(v["patient_unmatched_internal_mass"] for v in diag.values())
    terminal_u = sum(v["patient_unmatched_terminal_mass"] for v in diag.values())
    total_states = e11 + e12 + 2 * e21 + internal_u + terminal_u
    return {
        "method": name,
        "iterations": report["iterations"],
        "converged": report["converged"],
        "seconds": report["seconds"],
        "e11_mass": e11,
        "warp_mass": e12 + e21,
        "internal_unmatched_mass": internal_u,
        "terminal_unmatched_mass": terminal_u,
        "e11_frac": e11 / total_states if total_states else float("nan"),
        "warp_frac": (e12 + e21) / total_states if total_states else float("nan"),
        "internal_unmatched_frac": internal_u / total_states if total_states else float("nan"),
        "terminal_unmatched_frac": terminal_u / total_states if total_states else float("nan"),
        "omega_warp": report["resolved_fit_config"]["omega_warp"],
        "lambda_terminal_gap": report["resolved_fit_config"]["lambda_terminal_gap"],
        "max_patient_displacement": max(v["max_displacement"] for v in diag.values()),
        "max_patient_p95_displacement": max(v["p95_displacement"] for v in diag.values()),
        "ru831_mean_displacement": diag.get("RU831", {}).get("mean_displacement"),
        "ru831_max_displacement": diag.get("RU831", {}).get("max_displacement"),
        "consensus_dominance_max": max(v["consensus_share"] for v in diag.values()),
    }


def main():
    """Build and write the joint-warp run table (csv + json)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", type=Path, required=True)
    parser.add_argument("--joint-dir", type=Path, required=True)
    parser.add_argument("--baseline", action="append", nargs=2, metavar=("NAME", "CACHE"), default=[])
    parser.add_argument("--heldout-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = ad.read_h5ad(args.adata)
    heldout = json.loads(args.heldout_summary.read_text())
    rows = []

    raw = score(
        data.obs, data.obs.s_local.to_numpy(float), np.full(data.n_obs, "input"), np.zeros(data.n_obs), "s_local"
    )[-1]
    raw.update(
        {
            "run_type": "control",
            **dict.fromkeys(
                (
                    "iterations",
                    "converged",
                    "seconds",
                    "e11_mass",
                    "warp_mass",
                    "internal_unmatched_mass",
                    "terminal_unmatched_mass",
                    "e11_frac",
                    "warp_frac",
                    "internal_unmatched_frac",
                    "terminal_unmatched_frac",
                    "omega_warp",
                    "lambda_terminal_gap",
                    "max_patient_displacement",
                    "max_patient_p95_displacement",
                    "ru831_mean_displacement",
                    "ru831_max_displacement",
                    "consensus_dominance_max",
                )
            ),
        }
    )
    raw["held_out_marker_concordance"] = heldout.get("s_local", {}).get("mean_gene_concordance")
    raw["held_out_peak_spread"] = heldout.get("s_local", {}).get("mean_peak_spread")
    rows.append(raw)

    for name, cache_path in args.baseline:
        cache_path = Path(cache_path)
        d = np.load(cache_path, allow_pickle=False)
        if not np.array_equal(d["cell_ids"], np.asarray(data.obs_names, str)):
            raise ValueError(f"cell identity/order mismatch in {cache_path}")
        cfg = json.loads(str(d["config"]))
        row = score(data.obs, d["position"], d["status"], d["unprojected"], name)[-1]
        row["run_type"] = "baseline"
        for key in (
            "iterations",
            "converged",
            "seconds",
            "e11_mass",
            "warp_mass",
            "internal_unmatched_mass",
            "terminal_unmatched_mass",
            "e11_frac",
            "warp_frac",
            "internal_unmatched_frac",
            "terminal_unmatched_frac",
            "omega_warp",
            "lambda_terminal_gap",
            "max_patient_displacement",
            "max_patient_p95_displacement",
            "ru831_mean_displacement",
            "ru831_max_displacement",
            "consensus_dominance_max",
        ):
            row.setdefault(key, None)
        row["held_out_marker_concordance"] = heldout.get(cfg["name"], {}).get("mean_gene_concordance")
        row["held_out_peak_spread"] = heldout.get(cfg["name"], {}).get("mean_peak_spread")
        rows.append(row)

    for cache_path in sorted(args.joint_dir.glob("joint_warp_*.npz")):
        name = cache_path.stem
        report_path = cache_path.with_suffix(".json")
        d = np.load(cache_path, allow_pickle=False)
        if not np.array_equal(d["cell_ids"], np.asarray(data.obs_names, str)):
            raise ValueError(f"cell identity/order mismatch in {cache_path}")
        row = score(data.obs, d["position"], d["status"], d["unprojected"], name)[-1]
        row["run_type"] = "joint_posterior_warp"
        row.update(joint_candidate_row(name, cache_path, report_path))
        row["held_out_marker_concordance"] = heldout.get(name, {}).get("mean_gene_concordance")
        row["held_out_peak_spread"] = heldout.get(name, {}).get("mean_peak_spread")
        rows.append(row)

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    df.to_json(args.output.with_suffix(".json"), orient="records", indent=2)
    print(f"wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
