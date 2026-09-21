"""RU831 diagnostics, part 2b/3b: audit-driven follow-ups.

Written in response to a coordinator review of the initial Part 2 draft,
which flagged that the landmark-level ``mean_displacement`` gap (0.115
moderate vs 0.246 severe for RU831) traces to a numerical-underflow artifact
in ``_project.py:summarize()`` at near-zero-occupancy landmarks, not a real
shift in posterior mass. This script:

1. Recomputes landmark-level mean displacement restricted to
   well-supported landmarks (``patient_occupancy`` above a threshold),
   for RU831/RU1303/RU1444/RU581 in both regimes, to test whether that
   restriction removes the moderate-vs-severe gap.
2. Refits excluding RU831 (leave-one-out) to see how much the consensus
   barycenter near RU831's unmatchable block (local pseudotime
   0.707-0.889) shifts when RU831 no longer contributes to it -- a direct
   test of the barycenter-compromise hypothesis.
3. Refits with a small deterministic perturbation to the initial
   barycenter (monkeypatching ``scdebussy.align._api.initialize`` at the
   script level only -- no edits to ``_barycenter.py`` or ``_api.py``) to
   probe, in a limited way, whether the moderate-regime fit is sensitive
   to initialization near a second low-cost path family for RU831.

No architecture changes to any library module. No grid widening beyond
what report/JOINT_WARP scope already permits.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from posterior_diagnostics import refit, transformed_data

from scdebussy.align._project import summarize

ADATA = Path("/scratch/chanj3/wangm10/HTAN/results/aligned_scdebussy_preprocessed_v2_ru942_deg_all.h5ad")
GENES = Path("validation/v21/experiment_round/eight_patient/feature_selection/top_250.json")
MODERATE_REPORT = Path("validation/v21/experiment_round/eight_patient/joint_warp/joint_warp_o0.1_teq.json")
SEVERE_REPORT = Path("validation/v21/experiment_round/eight_patient/joint_warp/joint_warp_o0.02_teq.json")
OUT = Path("validation/v21/experiment_round/eight_patient/joint_warp/ru831_diagnostics")
PATIENTS = ["RU831", "RU1303", "RU1444", "RU581"]
OCC_THRESHOLD = 0.1


def occupancy_thresholded_displacement(result, patient, threshold):
    """Recompute mean |displacement| for one patient restricted to landmarks with patient_occupancy above threshold."""
    q = summarize(result)
    ix = list(result.patients).index(patient)
    post = result.posteriors[ix]
    kin = q.kinetic_curve[ix]
    occ = post.patient_occupancy
    finite = np.isfinite(kin)
    well_supported = finite & (occ > threshold)
    full_mean = float(np.mean(np.abs(kin[finite]))) if finite.any() else float("nan")
    restricted_mean = float(np.mean(np.abs(kin[well_supported]))) if well_supported.any() else float("nan")
    return {
        "n_finite_unrestricted": int(finite.sum()),
        "n_finite_restricted": int(well_supported.sum()),
        "mean_abs_displacement_unrestricted": full_mean,
        f"mean_abs_displacement_restricted_occ_gt_{threshold:.2f}": restricted_mean,
        "n_dropped_by_threshold": int(finite.sum() - well_supported.sum()),
        "occupancy_at_dropped_rows": occ[finite & ~well_supported].tolist(),
    }


def part1_occupancy_threshold(adata):
    """For each regime, refit and compute occupancy-thresholded displacement for every patient."""
    results = {}
    for label, report_path in (("moderate_o0.1_teq", MODERATE_REPORT), ("severe_o0.02_teq", SEVERE_REPORT)):
        cached = json.loads(report_path.read_text())
        result = refit(adata, cached["resolved_fit_config"])
        results[label] = {p: occupancy_thresholded_displacement(result, p, OCC_THRESHOLD) for p in PATIENTS}
    return results


def part2_leave_one_out(adata):
    """Refit excluding RU831 and compare the shared barycenter to the full 8-patient fit."""
    cached = json.loads(MODERATE_REPORT.read_text())
    resolved = cached["resolved_fit_config"]

    full_result = refit(adata, resolved)
    full_consensus = full_result.consensus_coordinates
    full_barycenter = full_result.barycenter

    loo_adata = adata[adata.obs[resolved["patient_key"]] != "RU831"].copy()
    loo_result = refit(loo_adata, resolved)
    loo_consensus = loo_result.consensus_coordinates
    loo_barycenter = loo_result.barycenter

    # Compare barycenters at the region of RU831's unmatchable block
    # (local pseudotime ~0.707-0.889 maps to a similar range of consensus
    # coordinates since coordinates are unit [0,1] and roughly monotonic).
    region_mask_full = (full_consensus >= 0.65) & (full_consensus <= 0.92)
    region_mask_loo = (loo_consensus >= 0.65) & (loo_consensus <= 0.92)

    # barycenters may differ in length/consensus_resolution between the two
    # fits only if consensus_resolution is unset in resolved -- check shape.
    same_shape = full_barycenter.shape == loo_barycenter.shape
    out = {
        "full_n_patients": len(full_result.patients),
        "loo_n_patients": len(loo_result.patients),
        "full_barycenter_shape": list(full_barycenter.shape),
        "loo_barycenter_shape": list(loo_barycenter.shape),
        "same_shape": same_shape,
        "region_n_full": int(region_mask_full.sum()),
        "region_n_loo": int(region_mask_loo.sum()),
    }
    if same_shape:
        diff = full_barycenter - loo_barycenter
        region_diff = diff[region_mask_full]
        overall_rel = float(np.linalg.norm(diff) / (np.linalg.norm(full_barycenter) + 1e-12))
        region_rel = (
            float(np.linalg.norm(region_diff) / (np.linalg.norm(full_barycenter[region_mask_full]) + 1e-12))
            if region_mask_full.any()
            else float("nan")
        )
        out["overall_relative_barycenter_shift"] = overall_rel
        out["region_0.65_0.92_relative_barycenter_shift"] = region_rel
        out["region_vs_overall_ratio"] = region_rel / overall_rel if overall_rel > 0 else float("nan")
    # Other patients' displacement, with and without RU831 present
    out["other_patient_mean_displacement"] = {}
    for p in ("RU1303", "RU1444", "RU581"):
        if p in full_result.patients and p in loo_result.patients:
            out["other_patient_mean_displacement"][p] = {
                "with_RU831": full_result.diagnostics[p]["mean_displacement"],
                "without_RU831": loo_result.diagnostics[p]["mean_displacement"],
            }
    return out


def make_perturbed_initialize(offset_scale=0.05):
    """Return a drop-in replacement for align._api.initialize with a fixed offset.

    Applies a deterministic sinusoidal offset to the true equal-patient-mean
    initial barycenter. offset_scale is a fraction of the initial
    barycenter's per-gene std. This is a script-level monkeypatch only --
    it does not alter any library file, and is restored after use.
    """
    from scdebussy.align._barycenter import initialize as real_initialize

    def perturbed(landmarks, coordinates=None, locality_radius=None):
        base = real_initialize(landmarks, coordinates, locality_radius)
        size, genes = base.shape
        phase = np.linspace(0, 2 * np.pi, size)[:, None]
        wobble = np.sin(phase + np.arange(genes)[None, :] * 0.37)
        std = base.std(axis=0, keepdims=True)
        return base + offset_scale * std * wobble

    return perturbed


def part3_perturbed_init(adata):
    """Refit the moderate regime from a perturbed initial barycenter and compare to the baseline fit."""
    cached = json.loads(MODERATE_REPORT.read_text())
    resolved = cached["resolved_fit_config"]

    baseline = refit(adata, resolved)

    import scdebussy.align._api as api_mod

    original_initialize = api_mod.initialize
    api_mod.initialize = make_perturbed_initialize(offset_scale=0.05)
    try:
        perturbed_result = refit(adata, resolved)
    finally:
        api_mod.initialize = original_initialize

    out = {"patients": {}}
    for p in PATIENTS:
        base_diag = baseline.diagnostics[p]
        pert_diag = perturbed_result.diagnostics[p]
        out["patients"][p] = {
            "baseline_mean_displacement": base_diag["mean_displacement"],
            "perturbed_mean_displacement": pert_diag["mean_displacement"],
            "baseline_path_entropy": base_diag["path_entropy"],
            "perturbed_path_entropy": pert_diag["path_entropy"],
            "baseline_expected_match_11": base_diag["expected_match_11"],
            "perturbed_expected_match_11": pert_diag["expected_match_11"],
        }
    out["baseline_converged"] = baseline.converged
    out["perturbed_converged"] = perturbed_result.converged
    out["baseline_iterations"] = len(baseline.history)
    out["perturbed_iterations"] = len(perturbed_result.history)
    # objective/history final value if available
    if baseline.history and perturbed_result.history:
        out["baseline_final_history_entry"] = baseline.history[-1]
        out["perturbed_final_history_entry"] = perturbed_result.history[-1]
    return out


def main():
    """Run all three audit-driven follow-ups and write their JSON outputs."""
    adata = transformed_data(ADATA, GENES)

    print("[extended] part1: occupancy-thresholded displacement", flush=True)
    p1 = part1_occupancy_threshold(adata)
    (OUT / "occupancy_thresholded_displacement.json").write_text(json.dumps(p1, indent=2) + "\n")

    print("[extended] part2: leave-one-out (exclude RU831)", flush=True)
    p2 = part2_leave_one_out(adata)
    (OUT / "leave_one_out_RU831.json").write_text(json.dumps(p2, indent=2) + "\n")

    print("[extended] part3: perturbed-initialization probe", flush=True)
    p3 = part3_perturbed_init(adata)
    (OUT / "perturbed_init_probe.json").write_text(json.dumps(p3, indent=2) + "\n")

    print(json.dumps({"occupancy_threshold": p1, "leave_one_out": p2, "perturbed_init": p3}, indent=2))


if __name__ == "__main__":
    main()
