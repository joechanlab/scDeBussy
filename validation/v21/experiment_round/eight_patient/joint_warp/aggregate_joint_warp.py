"""Build the machine-readable run table for the joint-warp Phase 4 matrix.

Joins the six joint-warp candidates against five comparison baselines (raw
s_local, original v2, both trajectory-250 restricted fits, moderate DTW),
reusing summarize_and_plot.score() for the standard metric set and adding the
posterior-mass diagnostics that only joint-mode candidates carry.
"""

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from summarize_and_plot import score

MASS_FIELDS = (
    "e11_mass", "e12_mass", "e21_mass", "warp_edge_mass",
    "internal_unmatched_mass", "terminal_unmatched_mass",
    "warp_edge_fraction_among_match_edges", "e11_patient_state_fraction",
    "warp_patient_state_fraction", "internal_unmatched_patient_state_fraction",
    "terminal_unmatched_patient_state_fraction",
)


def posterior_fractions(e11, e12, e21, internal_u, terminal_u):
    """Separate match-edge usage from the partition of patient-state mass."""
    match_edges = e11 + e12 + e21
    patient_states = e11 + e12 + 2 * e21 + internal_u + terminal_u
    masses = [e11, e12, e21, internal_u, terminal_u]
    if not np.isfinite(masses).all() or min(masses) < 0:
        raise ValueError("posterior masses must be finite and nonnegative")
    if patient_states <= 0:
        raise ValueError("posterior must include patient-state mass")
    fractions = {
        "warp_edge_fraction_among_match_edges": (e12 + e21) / match_edges if match_edges else float("nan"),
        "e11_patient_state_fraction": e11 / patient_states,
        "warp_patient_state_fraction": (e12 + 2 * e21) / patient_states,
        "internal_unmatched_patient_state_fraction": internal_u / patient_states,
        "terminal_unmatched_patient_state_fraction": terminal_u / patient_states,
    }
    partition = sum(value for key, value in fractions.items() if key.endswith("patient_state_fraction"))
    if not np.isclose(partition, 1.0, rtol=0, atol=1e-8):
        raise ValueError(f"patient-state fractions do not sum to one: {partition}")
    return fractions


def joint_candidate_row(name, cache_path, report_path):
    """Summarize one joint-warp candidate's posterior mass and diagnostics."""
    report = json.loads(report_path.read_text())
    diag = report["diagnostics"]
    e11 = sum(v["expected_match_11"] for v in diag.values())
    e12 = sum(v["expected_match_12"] for v in diag.values())
    e21 = sum(v["expected_match_21"] for v in diag.values())
    internal_u = sum(v["patient_unmatched_internal_mass"] for v in diag.values())
    terminal_u = sum(v["patient_unmatched_terminal_mass"] for v in diag.values())
    return {
        "method": name,
        "iterations": report["iterations"],
        "converged": report["converged"],
        "seconds": report["seconds"],
        "e11_mass": e11,
        "e12_mass": e12,
        "e21_mass": e21,
        "warp_edge_mass": e12 + e21,
        "internal_unmatched_mass": internal_u,
        "terminal_unmatched_mass": terminal_u,
        **posterior_fractions(e11, e12, e21, internal_u, terminal_u),
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
    parser.add_argument("--report-dir", type=Path,
                        help="directory containing committed candidate JSON reports (defaults to --joint-dir)")
    parser.add_argument("--baseline", action="append", nargs=2, metavar=("NAME", "CACHE"), default=[])
    parser.add_argument("--paired-unit-cache", type=Path, required=True,
                        help="frozen trajectory_250_unit restricted fit; required paired control")
    parser.add_argument("--heldout-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report_dir = args.report_dir or args.joint_dir
    if any(name == "trajectory_250_unit" for name, _ in args.baseline):
        parser.error("trajectory_250_unit is supplied by --paired-unit-cache")
    baselines = [*args.baseline, ("trajectory_250_unit", args.paired_unit_cache)]
    data = ad.read_h5ad(args.adata)
    heldout = json.loads(args.heldout_summary.read_text())
    rows = []

    raw = score(
        data.obs, data.obs.s_local.to_numpy(float), np.full(data.n_obs, "input"), np.zeros(data.n_obs), "s_local"
    )[-1]
    raw.update(
        {
            "run_type": "control",
            **dict.fromkeys((*MASS_FIELDS, "iterations", "converged", "seconds", "omega_warp",
                                  "lambda_terminal_gap", "max_patient_displacement",
                                  "max_patient_p95_displacement", "ru831_mean_displacement",
                                  "ru831_max_displacement", "consensus_dominance_max")),
        }
    )
    raw["held_out_marker_concordance"] = heldout.get("s_local", {}).get("mean_gene_concordance")
    raw["held_out_peak_spread"] = heldout.get("s_local", {}).get("mean_peak_spread")
    rows.append(raw)

    for name, cache_path in baselines:
        cache_path = Path(cache_path)
        d = np.load(cache_path, allow_pickle=False)
        if not np.array_equal(d["cell_ids"], np.asarray(data.obs_names, str)):
            raise ValueError(f"cell identity/order mismatch in {cache_path}")
        cfg = json.loads(str(d["config"]))
        row = score(data.obs, d["position"], d["status"], d["unprojected"], name)[-1]
        row["run_type"] = "baseline"
        for key in (*MASS_FIELDS, "iterations", "converged", "seconds", "omega_warp",
                    "lambda_terminal_gap", "max_patient_displacement", "max_patient_p95_displacement",
                    "ru831_mean_displacement", "ru831_max_displacement", "consensus_dominance_max"):
            row.setdefault(key, None)
        marker = heldout.get(name, heldout.get(cfg["name"], {}))
        row["held_out_marker_concordance"] = marker.get("mean_gene_concordance")
        row["held_out_peak_spread"] = marker.get("mean_peak_spread")
        rows.append(row)

    for cache_path in sorted(args.joint_dir.glob("joint_warp_*.npz")):
        name = cache_path.stem
        report_path = report_dir / f"{name}.json"
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
