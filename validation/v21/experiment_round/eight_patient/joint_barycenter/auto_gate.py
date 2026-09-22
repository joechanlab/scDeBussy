"""8-patient joint fit with a label-free, data-driven coverage gate.

Tests whether the manually-selected RU1303-only gate (validated in
GATE_ABLATION_POSTFIX.md) can be recovered from pre-fit trajectory
information alone, without naming RU1303. Every other mechanic is frozen and
reused unchanged from eight_patient_joint.py: trajectory-250 genes, unit
precision, matchability handling, DTW-style warp/gap graph, gamma=1, and
terminal price values. Only the gate *decision rule* changes -- the
underlying endpoint/coverage score (`frozen_coverage`, already label-free and
pre-fit) is unchanged; what's new is a data-driven threshold instead of the
fixed `score <= max - 0.10` margin. No cell-type label, held-out marker, or
final alignment performance is used to pick the gate.
"""

import csv
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eight_patient_joint import (  # noqa: E402
    CACHE,
    CHEAP_PRICE,
    ETA,
    INPUT,
    MARKERS,
    PATIENTS,
    ROOT,
    cell_positions_and_status,
    compile_graph,
    frozen_coverage,
    marker_landmarks,
    marker_mse,
    match_cost,
    quality,
    ru1303_overlap,
    ru831_plateau_diagnostic,
    run_em,
    save_contribution_heatmap,
    smoothness,
)

OUT = Path(__file__).resolve().parent / "auto_gate"
GAMMA = 1.0
TERMINAL_K = 10  # same window frozen_coverage already uses for the endpoint centroid


def largest_gap_gate(scores, patients):
    """Simplest label-free split of the coverage-score distribution: sort the
    scores, find the single largest gap between consecutive values, and gate
    on (cheap terminal price) everything below the midpoint of that gap. No
    patient identity, cell-type label, held-out marker, or fit-quality metric
    is used -- only the pre-fit endpoint/coverage score itself. This is an
    unsupervised 1D two-cluster split (a simple "largest gap" / Otsu-style
    threshold for k=2), not a fixed margin."""
    ordered = sorted(patients, key=lambda p: scores[p])
    values = [scores[p] for p in ordered]
    gaps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    split = int(np.argmax(gaps))  # index of the larger-of-pair patient at the widest gap
    threshold = (values[split] + values[split + 1]) / 2
    cheap = {p: scores[p] < threshold for p in patients}
    diagnostics = {
        "sorted_scores": list(zip(ordered, values, strict=True)),
        "gaps": gaps,
        "largest_gap": gaps[split],
        "largest_gap_between": (ordered[split], ordered[split + 1]),
        "second_largest_gap": sorted(gaps, reverse=True)[1] if len(gaps) > 1 else None,
        "threshold": threshold,
    }
    return cheap, diagnostics


def late_matchable_fraction(matchable, window=10):
    return {p: float(matchable[p][-window:].mean()) for p in matchable}


def main():
    OUT.mkdir(exist_ok=True)
    fit = np.load(CACHE, allow_pickle=False)
    panel = json.loads(
        (ROOT / "validation/v21/experiment_round/eight_patient/feature_selection/top_250.json").read_text()
    )
    assert fit["genes"].tolist() == panel
    names = fit["patient_order"].tolist()
    all_expression = fit["landmark_expression"].reshape(8, 100, 250)
    all_matchable = fit["landmark_matchable"].reshape(8, 100)
    expression = {patient: all_expression[names.index(patient)] for patient in PATIENTS}
    matchable = {patient: all_matchable[names.index(patient)] for patient in PATIENTS}

    # --- pre-fit, label-free signals ---
    scores, peer_positions, _fixed_margin_cheap = frozen_coverage(expression, PATIENTS)
    cheap, gate_diag = largest_gap_gate(scores, PATIENTS)
    late10 = late_matchable_fraction(matchable, window=10)
    late20 = late_matchable_fraction(matchable, window=20)

    print("coverage scores:", json.dumps(scores, indent=2))
    print("gate diagnostics:", json.dumps(gate_diag, indent=2, default=str))
    print("late-10 matchable fraction:", json.dumps(late10, indent=2))
    print("automatic gate_on:", json.dumps(cheap, indent=2))

    prices = {patient: CHEAP_PRICE if cheap[patient] else ETA for patient in PATIENTS}
    with (OUT / "frozen_gate.json").open("w") as handle:
        json.dump(
            {
                "scores": scores,
                "peer_positions": peer_positions,
                "gate_on": cheap,
                "tail_prices": prices,
                "rule": "largest_gap_in_sorted_coverage_score (label-free, pre-fit, unsupervised 1D split)",
                "gate_diagnostics": gate_diag,
                "late10_matchable_fraction": late10,
                "late20_matchable_fraction": late20,
                "note": "late-matchable fractions are reported as a candidate signal but were NOT used in "
                "the gate decision: every patient (including RU1303) has late10=10/10, so this signal is "
                "uninformative for distinguishing RU1303 in this cohort; the decision is driven entirely "
                "by the largest-gap split of the endpoint/coverage score.",
            },
            handle,
            indent=2,
            default=str,
        )
        handle.write("\n")
    full_patients = tuple(p for p in PATIENTS if not cheap[p])
    print(f"full_patients={full_patients}")

    seed = np.mean([expression[p] for p in full_patients], axis=0)
    graphs = {
        p: compile_graph(match_cost(expression[p], seed), prices[p], matchable=matchable[p]) for p in PATIENTS
    }

    adata = ad.read_h5ad(INPUT, backed="r")
    assert np.array_equal(fit["cell_ids"], np.asarray(adata.obs_names, str))
    all_patients = adata.obs["patient"].astype(str).to_numpy()
    selected = np.isin(all_patients, PATIENTS)
    cell_ids = np.asarray(adata.obs_names, str)[selected]
    cell_patients = all_patients[selected]
    labels = adata.obs["cell_type_final2"].astype(str).to_numpy()[selected]
    raw = adata.obs["s_local"].to_numpy(float)[selected]
    markers = {patient: marker_landmarks(adata, patient, MARKERS) for patient in PATIENTS}
    marker_scale = np.std(np.vstack([markers[p] for p in full_patients]), axis=0)
    marker_scale = np.where(marker_scale > 1e-8, marker_scale, 1)

    run = run_em(expression, graphs, seed, PATIENTS, GAMMA)
    posterior, initial_post = run["posterior"], run["initial_post"]
    final_position, status, unprojected = cell_positions_and_status(posterior, raw, cell_patients, PATIENTS)
    initial_position, _, _ = cell_positions_and_status(initial_post, raw, cell_patients, PATIENTS)
    quality_by_phase = {
        "initial": quality(initial_position, raw, cell_patients, labels, PATIENTS),
        "final": quality(final_position, raw, cell_patients, labels, PATIENTS),
    }
    rows = []
    for phase, state in (("initial", initial_post), ("final", posterior)):
        for patient in PATIENTS:
            p = state[patient]
            rows.append(
                {
                    "phase": phase,
                    "patient": patient,
                    "gate_on": cheap[patient],
                    "coverage_score": scores[patient],
                    "terminal_skipped_fraction": p["tail_skipped_fraction"],
                    "internal_patient_gaps": p["internal_patient_gaps"],
                    "internal_reference_gaps": p["internal_reference_gaps"],
                    "map_posterior_mean_abs_difference": p["map_posterior_mean_abs_difference"],
                    "posterior_mean_match_mass": p["posterior_mean_match_mass"],
                    "heldout_marker_mse": marker_mse(patient, state, markers, marker_scale, full_patients)
                    if patient in full_patients
                    else float("nan"),
                    **quality_by_phase[phase][patient],
                }
            )
        rows.append(
            {
                "phase": phase,
                "patient": "all",
                "gate_on": "",
                "coverage_score": float("nan"),
                "terminal_skipped_fraction": float("nan"),
                "internal_patient_gaps": "",
                "internal_reference_gaps": "",
                "map_posterior_mean_abs_difference": float("nan"),
                "posterior_mean_match_mass": float("nan"),
                "heldout_marker_mse": float("nan"),
                **quality_by_phase[phase]["all"],
            }
        )
    with (OUT / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    convergence = {
        "converged": run["converged"],
        "iterations": len(run["history"]),
        "initial_smoothness_rms_second_difference": smoothness(run["initial_barycenter"]),
        "final_smoothness_rms_second_difference": smoothness(run["final_barycenter"]),
        "ru1303_sclc_a_in_other_sclc_n_q10_q90_initial": ru1303_overlap(initial_position, cell_patients, labels),
        "ru1303_sclc_a_in_other_sclc_n_q10_q90_final": ru1303_overlap(final_position, cell_patients, labels),
        "ru831_plateau_diagnostic": ru831_plateau_diagnostic(posterior),
        "history": run["history"],
    }
    with (OUT / "convergence.json").open("w") as handle:
        json.dump(convergence, handle, indent=2)
        handle.write("\n")

    save_contribution_heatmap(posterior, PATIENTS, GAMMA, OUT / "contribution_weight_auto_gate.png")

    np.savez_compressed(
        OUT / "auto_gate.npz",
        cell_ids=cell_ids,
        position=final_position,
        status=status,
        unprojected=unprojected,
        genes=np.asarray(panel, str),
        config=json.dumps(
            {
                "name": "eight_patient_joint_auto_gate",
                "gamma": GAMMA,
                "patients": list(PATIENTS),
                "full_patients": list(full_patients),
                "gate_rule": "largest_gap_in_sorted_coverage_score",
            }
        ),
    )
    np.savez_compressed(OUT / "pilot_state.npz", final_barycenter=run["final_barycenter"])

    report = {
        "gate_on": cheap,
        "converged": run["converged"],
        "iterations": len(run["history"]),
        "final_smoothness": convergence["final_smoothness_rms_second_difference"],
        "ru1303_overlap_final": convergence["ru1303_sclc_a_in_other_sclc_n_q10_q90_final"],
        "overall": quality_by_phase["final"]["all"],
    }
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
