"""8-patient joint fit with the coverage gate manually frozen to RU1303-only.

Isolates whether the coverage regression found in eight_patient_joint.py
(automatic gate: 5/8 patients gated cheap) comes from the gate rule firing on
patients beyond RU1303, or from something else about scaling to 8 patients.
Every mechanic is otherwise identical and reused unchanged from
eight_patient_joint.py (frozen trajectory-250 genes, unit precision, DTW-style
warp/gap graph, gamma=1 curvature regularization, terminal pricing formula) --
only which patients receive the cheap vs full terminal price is overridden,
to match the gate outcome validated in the 3-/4-patient pilots (RU1303 on,
everyone else off). No parameter (gamma, graph, genes, precision, terminal
price values) is retuned; gamma=1 only (the validated main setting), no
gamma=0 control needed for this ablation.
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
    INPUT,
    MARKERS,
    PATIENTS,
    ROOT,
    TYPES,
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
from coverage_gate import CHEAP_PRICE  # noqa: E402
from focused_followup import ETA  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ru1303_ru581_pairwise"))

OUT = Path(__file__).resolve().parent / "gate_ablation"
GAMMA = 1.0
MANUAL_GATE = {"RU1303": True}  # every other patient defaults to False below


def main():
    OUT.mkdir(exist_ok=True)
    fit = np.load(CACHE, allow_pickle=False)
    panel = json.loads(
        (ROOT / "validation/v21/experiment_round/eight_patient/feature_selection/top_250.json").read_text()
    )
    assert fit["genes"].tolist() == panel
    assert not set(MARKERS).intersection(panel)
    names = fit["patient_order"].tolist()
    all_expression = fit["landmark_expression"].reshape(8, 100, 250)
    expression = {patient: all_expression[names.index(patient)] for patient in PATIENTS}

    # Record the automatic gate's scores for reference, but override the gate decision.
    scores, peer_positions, auto_cheap = frozen_coverage(expression, PATIENTS)
    cheap = {patient: MANUAL_GATE.get(patient, False) for patient in PATIENTS}
    prices = {patient: CHEAP_PRICE if cheap[patient] else ETA for patient in PATIENTS}
    with (OUT / "frozen_gate.json").open("w") as handle:
        json.dump(
            {
                "scores": scores,
                "peer_positions": peer_positions,
                "automatic_gate_on": auto_cheap,
                "manual_gate_on": cheap,
                "tail_prices": prices,
                "note": "gate manually frozen to RU1303-only per the ablation request; scores/prices formula unchanged",
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    full_patients = tuple(p for p in PATIENTS if not cheap[p])
    print(f"manual gate_on={cheap}")
    print(f"full_patients={full_patients}")

    seed = np.mean([expression[p] for p in full_patients], axis=0)
    graphs = {p: compile_graph(match_cost(expression[p], seed), prices[p]) for p in PATIENTS}

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

    print(f"running gamma={GAMMA}", flush=True)
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

    save_contribution_heatmap(posterior, PATIENTS, GAMMA, OUT / "contribution_weight_ru1303_only_gate.png")

    np.savez_compressed(
        OUT / "ru1303_only_gate.npz",
        cell_ids=cell_ids,
        position=final_position,
        status=status,
        unprojected=unprojected,
        genes=np.asarray(panel, str),
        config=json.dumps(
            {
                "name": "eight_patient_joint_ru1303_only_gate",
                "gamma": GAMMA,
                "patients": list(PATIENTS),
                "full_patients": list(full_patients),
                "gate": "manually frozen to RU1303-only",
            }
        ),
    )
    np.savez_compressed(OUT / "pilot_state.npz", final_barycenter=run["final_barycenter"])

    report = {
        "manual_gate_on": cheap,
        "converged": run["converged"],
        "iterations": len(run["history"]),
        "final_smoothness": convergence["final_smoothness_rms_second_difference"],
        "ru1303_overlap_final": convergence["ru1303_sclc_a_in_other_sclc_n_q10_q90_final"],
        "overall": quality_by_phase["final"]["all"],
    }
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
