"""Reconstruct the pre-matchability-fix 8-patient gamma=1 result for comparison.

The original pre-fix gamma1.npz (from commit 3455669) was gitignored (*.npz) and
got overwritten in place when eight_patient_joint.py was rerun with the fix. The
model is fully deterministic (no RNG anywhere in this pipeline), and
compile_graph(..., matchable=None) is a verified no-op reproducing prior
behavior exactly (see test_matchable_fix.py), so this reruns the identical
pre-fix computation to regenerate that one missing artifact -- not a new
experiment, just recovering a comparison baseline. gamma=1 only (the main
setting needed for the comparison); gamma=0 is not regenerated here.
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
    run_em,
    smoothness,
)

OUT = Path(__file__).resolve().parent / "eight_patient_joint"
GAMMA = 1.0


def main():
    fit = np.load(CACHE, allow_pickle=False)
    panel = json.loads(
        (ROOT / "validation/v21/experiment_round/eight_patient/feature_selection/top_250.json").read_text()
    )
    assert fit["genes"].tolist() == panel
    names = fit["patient_order"].tolist()
    all_expression = fit["landmark_expression"].reshape(8, 100, 250)
    expression = {patient: all_expression[names.index(patient)] for patient in PATIENTS}
    scores, peer_positions, cheap = frozen_coverage(expression, PATIENTS)
    prices = {patient: CHEAP_PRICE if cheap[patient] else ETA for patient in PATIENTS}
    full_patients = tuple(p for p in PATIENTS if not cheap[p])

    seed = np.mean([expression[p] for p in full_patients], axis=0)
    graphs = {p: compile_graph(match_cost(expression[p], seed), prices[p]) for p in PATIENTS}  # matchable=None: pre-fix

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
    posterior = run["posterior"]
    final_position, status, unprojected = cell_positions_and_status(posterior, raw, cell_patients, PATIENTS)

    # Sanity check against the numbers committed in 3455669 / EIGHT_PATIENT_JOINT.md
    print("final_smoothness:", smoothness(run["final_barycenter"]), "(expect 0.15857114222246593)")
    rows = []
    for patient in PATIENTS:
        p = posterior[patient]
        rows.append((patient, p["tail_skipped_fraction"], p["internal_patient_gaps"]))
    print(rows)

    np.savez_compressed(
        OUT / "gamma1_prefix.npz",
        cell_ids=cell_ids,
        position=final_position,
        status=status,
        unprojected=unprojected,
        genes=np.asarray(panel, str),
        config=json.dumps(
            {"name": "eight_patient_joint_gamma1_prefix", "gamma": GAMMA, "patients": list(PATIENTS)}
        ),
    )
    print("wrote", OUT / "gamma1_prefix.npz")


if __name__ == "__main__":
    main()
