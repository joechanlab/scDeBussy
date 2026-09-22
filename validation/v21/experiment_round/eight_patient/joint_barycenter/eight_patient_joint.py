"""Full 8-patient gated posterior barycenter fit, gamma=1 (main) vs gamma=0 (diagnostic).

Extends the validated 3-/4-patient pilots
(../ru1303_ru581_pairwise/{three,four}_patient_joint.py,
../ru1303_ru581_pairwise/barycenter_smoothing.py) to all 8 frozen HTAN patients.
Every frozen mechanic is reused unchanged and called directly from those
modules' PATIENTS-independent building blocks (`compile_graph`, `infer_graph`,
`match_cost`, `smoothness`, `second_diff_operator`): frozen trajectory-250
genes, unit precision, the DTW-style warp/gap graph, the coverage-gate rule,
and terminal pricing. Only the patient loop is generalized to an explicit
list. No parameter is retuned; only gamma in {0, 1} is compared, with gamma=0
run once as a diagnostic control (not swept).

Emits, per gamma, a candidate cache (`cell_ids`, `position`, `status`,
`unprojected`, `config`, `genes`) compatible with the frozen scoring/plotting
conventions in ../../summarize_and_plot.py and ../../heldout_markers.py, so
those unmodified scripts can be invoked on the new fits for a like-for-like
comparison against the existing raw/original_v2/trajectory_250 controls.

`status`/`unprojected` here approximate the v2 API's per-cell projection
confidence (`scdebussy.align._project`, unprojected = 1 - retained
probability mass, unsupported iff unprojected > 0.5) using the graph
posterior's own per-patient-landmark retained match mass
(`matrix.sum(axis=1)`, the same quantity already used to decide whether a
landmark's curve value is finite). It is NOT computed via `align.project`
and is only an analogous, not identical, diagnostic.
"""

import csv
import json
import sys
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PAIRWISE_DIR = Path(__file__).resolve().parents[1] / "ru1303_ru581_pairwise"
sys.path.insert(0, str(PAIRWISE_DIR))

from run_pairwise import CACHE, GRID, INPUT, MARKERS, ROOT, interpolate_curve, marker_landmarks  # noqa: E402
from three_patient_joint import MAX_ITER, PATIENCE, TOL, compile_graph, match_cost, smoothness  # noqa: E402
from three_patient_joint import infer_graph as _infer_graph  # noqa: E402
from coverage_gate import CHEAP_PRICE, COVERAGE_MARGIN, TERMINAL_K  # noqa: E402
from focused_followup import ETA  # noqa: E402
from barycenter_smoothing import second_diff_operator  # noqa: E402

OUT = Path(__file__).resolve().parent / "eight_patient_joint"
PATIENTS = ("RU1083", "RU1303", "RU1444", "RU151", "RU263", "RU581", "RU831", "RU942")
TYPES = ("LUAD", "SCLC-A", "SCLC-N")
COLORS = {"LUAD": "gold", "SCLC-A": "tab:red", "SCLC-N": "tab:cyan"}
GAMMAS = (1.0, 0.0)  # gamma=1 is the main run; gamma=0 is a single diagnostic control


def frozen_coverage(x, patients):
    """Identical terminal-vs-peer-position coverage score used by the 3-/4-patient pilots."""
    scores, detail = {}, {}
    for query in patients:
        centroid = x[query][-TERMINAL_K:].mean(axis=0)
        positions = {}
        for reference in patients:
            if reference == query:
                continue
            distance = np.mean((x[reference] - centroid) ** 2, axis=1)
            positions[reference] = float(GRID[np.argmin(distance)])
        scores[query] = float(np.median(list(positions.values())))
        detail[query] = positions
    maximum = max(scores.values())
    cheap = {patient: scores[patient] <= maximum - COVERAGE_MARGIN for patient in patients}
    return scores, detail, cheap


def expectation(x, barycenter, graphs, patients):
    return {patient: _infer_graph(*graphs[patient], match_cost(x[patient], barycenter)) for patient in patients}


def barycenter_update(x, posterior, previous, patients):
    numerator = sum(posterior[patient]["matrix"].T @ x[patient] for patient in patients)
    denominator = sum(posterior[patient]["matrix"].sum(axis=0) for patient in patients)
    updated = previous.copy()
    supported = denominator > 1e-10
    updated[supported] = numerator[supported] / denominator[supported, None]
    return updated, denominator


def barycenter_update_smooth(x, posterior, previous, patients, gamma):
    numerator = sum(posterior[patient]["matrix"].T @ x[patient] for patient in patients)
    denominator = sum(posterior[patient]["matrix"].sum(axis=0) for patient in patients)
    if gamma == 0.0:
        return barycenter_update(x, posterior, previous, patients)
    d = second_diff_operator()
    a = np.diag(denominator) + gamma * (d.T @ d)
    updated = np.linalg.solve(a, numerator)
    return updated, denominator


def run_em(expression, graphs, seed, patients, gamma):
    seed_post = expectation(expression, seed, graphs, patients)
    initial, _ = barycenter_update_smooth(expression, seed_post, seed, patients, gamma)
    posterior = expectation(expression, initial, graphs, patients)
    initial_post = posterior
    barycenter = initial
    history = []
    stable = 0
    converged = False
    for iteration in range(1, MAX_ITER + 1):
        before = -sum(posterior[patient]["log_z"] for patient in patients)
        updated, support = barycenter_update_smooth(expression, posterior, barycenter, patients, gamma)
        following = expectation(expression, updated, graphs, patients)
        after = -sum(following[patient]["log_z"] for patient in patients)
        relative = (before - after) / max(1.0, abs(before))
        history.append(
            {
                "iteration": iteration,
                "negative_log_partition": after,
                "relative_improvement": relative,
                "barycenter_max_change": float(np.max(np.abs(updated - barycenter))),
                "min_consensus_match_support": float(support.min()),
            }
        )
        barycenter, posterior = updated, following
        stable = stable + 1 if relative <= TOL else 0
        if stable >= PATIENCE:
            converged = True
            break
    return {
        "gamma": gamma,
        "initial_barycenter": initial,
        "final_barycenter": barycenter,
        "initial_post": initial_post,
        "posterior": posterior,
        "history": history,
        "converged": converged,
    }


def marker_mse(patient, posterior, markers, marker_scale, full_patients):
    peers = [peer for peer in full_patients if peer != patient]
    numerator = sum(posterior[peer]["matrix"].T @ markers[peer] for peer in peers)
    denominator = sum(posterior[peer]["matrix"].sum(axis=0) for peer in peers)
    supported = denominator > 1e-10
    template = np.full_like(numerator, np.nan)
    template[supported] = numerator[supported] / denominator[supported, None]
    for gene in range(template.shape[1]):
        template[:, gene] = np.interp(GRID, GRID[supported], template[supported, gene])
    curve = posterior[patient]["curve"]
    valid = np.isfinite(curve)
    predicted = np.column_stack([np.interp(curve[valid], GRID, template[:, gene]) for gene in range(len(MARKERS))])
    return float(np.mean(((markers[patient][valid] - predicted) / marker_scale) ** 2))


def cell_positions_and_status(posterior, local, cell_patients, patients):
    """Map cells through each patient's posterior curve; approximate per-cell status/unprojected
    from the same per-patient-landmark retained match mass used to decide curve finiteness."""
    mapped = local.copy()
    status = np.full(len(local), "unsupported", dtype="<U11")
    unprojected = np.ones(len(local), float)
    for patient in patients:
        selected = cell_patients == patient
        mapped[selected] = interpolate_curve(posterior[patient]["curve"], local[selected])
        mass = posterior[patient]["matrix"].sum(axis=1)  # retained probability per own-landmark bin, in [0,1]
        own_unprojected = np.clip(1.0 - mass, 0.0, 1.0)
        bins = np.clip((local[selected] * 99).astype(int), 0, 99)
        u = own_unprojected[bins]
        unprojected[selected] = u
        status[selected] = np.where(u > 0.5, "unsupported", "supported")
    return mapped, status, unprojected


def quality(position, raw, cell_patients, labels, patients):
    from sklearn.neighbors import NearestNeighbors

    finite = np.isfinite(position)
    indices = np.flatnonzero(finite)
    nearest = NearestNeighbors(n_neighbors=50, algorithm="kd_tree").fit(position[indices, None])
    neighbors = nearest.kneighbors(position[indices, None], return_distance=False)
    scores = np.full(len(position), np.nan)
    for local_index, original in enumerate(indices):
        candidate = indices[neighbors[local_index]]
        cross = cell_patients[candidate] != cell_patients[original]
        scores[original] = float(np.mean(labels[candidate][cross] == labels[original])) if cross.any() else 0.0
    bins = np.zeros(len(position), int)
    bins[finite] = np.clip((position[finite] * 99).astype(int), 0, 99)
    coverage = np.mean([len(np.unique(cell_patients[finite & (bins == b)])) / len(patients) for b in range(100)])
    result = {}
    for patient in (*patients, "all"):
        selected = cell_patients == patient if patient != "all" else np.ones(len(position), bool)
        valid = selected & finite
        occupied = np.zeros(100, bool)
        occupied[bins[valid]] = True
        displacement = np.abs(position[valid] - raw[valid])
        result[patient] = {
            "coverage": float(occupied.mean()) if patient != "all" else float(coverage),
            "purity": float(np.nanmean(scores[selected])),
            "mean_displacement": float(np.mean(displacement)) if valid.any() else float("nan"),
            "p95_displacement": float(np.quantile(displacement, 0.95)) if valid.any() else float("nan"),
            "finite_cell_fraction": float(finite[selected].mean()),
        }
    return result


def ru1303_overlap(position, cell_patients, labels):
    ru = cell_patients == "RU1303"
    finite = np.isfinite(position)
    other_n = position[(~ru) & (labels == "SCLC-N") & finite]
    ru_a = position[ru & (labels == "SCLC-A") & finite]
    if not len(other_n) or not len(ru_a):
        return float("nan")
    low, high = np.quantile(other_n, [0.1, 0.9])
    return float(np.mean((ru_a >= low) & (ru_a <= high)))


def ru831_plateau_diagnostic(posterior):
    """Longest run of consecutive identical MAP positions ('plateau') and matched span for RU831."""
    curve = posterior["RU831"]["map_curve"]
    finite = np.isfinite(curve)
    values = curve[finite]
    if len(values) == 0:
        return {"longest_plateau_landmarks": 0, "n_distinct_positions": 0, "matched_landmarks": 0}
    longest, current = 1, 1
    for a, b in zip(values[:-1], values[1:]):
        if a == b:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return {
        "longest_plateau_landmarks": int(longest),
        "n_distinct_positions": int(len(np.unique(values))),
        "matched_landmarks": int(finite.sum()),
    }


def local_curvature(barycenter):
    d2 = barycenter[2:] - 2 * barycenter[1:-1] + barycenter[:-2]
    return np.sqrt(np.mean(d2**2, axis=1))


def save_contribution_heatmap(posterior, patients, gamma, path):
    mass = np.stack([posterior[patient]["matrix"].sum(axis=0) for patient in patients])
    total = mass.sum(axis=0)
    share = np.divide(mass, total, out=np.zeros_like(mass), where=total > 1e-12)
    fig, ax = plt.subplots(figsize=(11, 3.0), dpi=170)
    im = ax.imshow(
        share, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=1,
        extent=[0, 1, len(patients) - 0.5, -0.5],
    )
    ax.set(
        yticks=np.arange(len(patients)), yticklabels=patients, xlabel="shared position [0,1]",
        title=f"Per-patient share of consensus match mass (γ={gamma:g}, 8 patients)",
    )
    fig.colorbar(im, ax=ax, label="fraction of total mass")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return share


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
    all_matchable = fit["landmark_matchable"].reshape(8, 100)
    expression = {patient: all_expression[names.index(patient)] for patient in PATIENTS}
    matchable = {patient: all_matchable[names.index(patient)] for patient in PATIENTS}
    print(f"matchable_counts={{{', '.join(f'{p}: {int(matchable[p].sum())}/100' for p in PATIENTS)}}}")
    scores, peer_positions, cheap = frozen_coverage(expression, PATIENTS)
    prices = {patient: CHEAP_PRICE if cheap[patient] else ETA for patient in PATIENTS}
    with (OUT / "frozen_gate.json").open("w") as handle:
        json.dump(
            {"scores": scores, "peer_positions": peer_positions, "gate_on": cheap, "tail_prices": prices},
            handle,
            indent=2,
        )
        handle.write("\n")
    full_patients = tuple(p for p in PATIENTS if not cheap[p])
    print(f"gate_on={cheap}")
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

    summaries = {}
    for gamma in GAMMAS:
        print(f"running gamma={gamma}", flush=True)
        run = run_em(expression, graphs, seed, PATIENTS, gamma)
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
                        "gamma": gamma,
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
                    "gamma": gamma,
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
        summaries[gamma] = {
            "converged": run["converged"],
            "iterations": len(run["history"]),
            "history": run["history"],
            "initial_smoothness": smoothness(run["initial_barycenter"]),
            "final_smoothness": smoothness(run["final_barycenter"]),
            "rows": rows,
            "final_position": final_position,
            "final_barycenter": run["final_barycenter"],
            "posterior": posterior,
            "status": status,
            "unprojected": unprojected,
            "ru1303_overlap_initial": ru1303_overlap(initial_position, cell_patients, labels),
            "ru1303_overlap_final": ru1303_overlap(final_position, cell_patients, labels),
            "ru831": ru831_plateau_diagnostic(posterior),
        }
        # candidate cache compatible with ../../summarize_and_plot.py and ../../heldout_markers.py
        np.savez_compressed(
            OUT / f"gamma{gamma:g}.npz",
            cell_ids=cell_ids,
            position=final_position,
            status=status,
            unprojected=unprojected,
            genes=np.asarray(panel, str),
            config=json.dumps(
                {
                    "name": f"eight_patient_joint_gamma{gamma:g}",
                    "gamma": gamma,
                    "patients": list(PATIENTS),
                    "full_patients": list(full_patients),
                }
            ),
        )

    with (OUT / "metrics.csv").open("w", newline="") as handle:
        all_rows = [row for gamma in GAMMAS for row in summaries[gamma]["rows"]]
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)

    convergence = {
        str(gamma): {
            "converged": summaries[gamma]["converged"],
            "iterations": summaries[gamma]["iterations"],
            "initial_smoothness_rms_second_difference": summaries[gamma]["initial_smoothness"],
            "final_smoothness_rms_second_difference": summaries[gamma]["final_smoothness"],
            "ru1303_sclc_a_in_other_sclc_n_q10_q90_initial": summaries[gamma]["ru1303_overlap_initial"],
            "ru1303_sclc_a_in_other_sclc_n_q10_q90_final": summaries[gamma]["ru1303_overlap_final"],
            "ru831_plateau_diagnostic": summaries[gamma]["ru831"],
            "history": summaries[gamma]["history"],
        }
        for gamma in GAMMAS
    }
    with (OUT / "convergence.json").open("w") as handle:
        json.dump(convergence, handle, indent=2)
        handle.write("\n")

    shares = {}
    for gamma in GAMMAS:
        shares[gamma] = save_contribution_heatmap(
            summaries[gamma]["posterior"], PATIENTS, gamma, OUT / f"contribution_weight_gamma{gamma:g}.png"
        )
    np.savez_compressed(
        OUT / "pilot_state.npz",
        patient_order=np.asarray(PATIENTS),
        full_patients=np.asarray(full_patients),
        **{f"final_barycenter_gamma{gamma:g}": summaries[gamma]["final_barycenter"] for gamma in GAMMAS},
        **{f"contribution_share_gamma{gamma:g}": shares[gamma] for gamma in GAMMAS},
    )

    report = {"patients": PATIENTS, "full_patients": full_patients, "gate_on": cheap, "coverage_scores": scores}
    for gamma in GAMMAS:
        report[f"gamma={gamma:g}"] = {
            "converged": summaries[gamma]["converged"],
            "iterations": summaries[gamma]["iterations"],
            "final_smoothness": summaries[gamma]["final_smoothness"],
            "ru1303_overlap_final": summaries[gamma]["ru1303_overlap_final"],
        }
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
