"""4-patient gated posterior barycenter pilot, comparing gamma=0 vs gamma=1.

Adds RU1444 to the frozen 3-patient pilot (RU1303, RU581, RU1083). Every
mechanic is reused unchanged from `three_patient_joint.py` / `barycenter_smoothing.py`
(frozen trajectory-250 genes, unit precision, DTW-style warp/gap graph,
coverage gate, terminal pricing) via direct calls into those modules'
PATIENTS-independent building blocks (`compile_graph`, `infer_graph`,
`match_cost`, `smoothness`). Only the patient loop is generalized to an
explicit list so a 4th patient can be added without touching any frozen
constant. No parameter is retuned; only gamma in {0, 1} is compared.
"""

import csv
import json
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from run_pairwise import CACHE, COLORS, GRID, INPUT, MARKERS, ROOT, TYPES, interpolate_curve, marker_landmarks
from three_patient_joint import MAX_ITER, PATIENCE, TOL, compile_graph, match_cost, smoothness
from three_patient_joint import infer_graph as _infer_graph  # noqa: F401 (used inside expectation via compile_graph's graphs)
from coverage_gate import CHEAP_PRICE, COVERAGE_MARGIN, TERMINAL_K
from focused_followup import ETA
from barycenter_smoothing import second_diff_operator

OUT = Path(__file__).resolve().parent / "four_patient_joint"
PATIENTS = ("RU1303", "RU581", "RU1083", "RU1444")
GAMMAS = (0.0, 1.0)


def frozen_coverage(x, patients):
    """Same terminal-vs-peer-position coverage score as the 3-patient pilot, now over 4 patients."""
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


def cell_positions(posterior, local, cell_patients, patients):
    mapped = local.copy()
    for patient in patients:
        selected = cell_patients == patient
        mapped[selected] = interpolate_curve(posterior[patient]["curve"], local[selected])
    return mapped


def quality(position, raw, cell_patients, labels, patients):
    finite = np.isfinite(position)
    indices = np.flatnonzero(finite)
    from sklearn.neighbors import NearestNeighbors

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
        result[patient] = {
            "coverage": float(occupied.mean()) if patient != "all" else float(coverage),
            "purity": float(np.nanmean(scores[selected])),
            "mean_displacement": float(np.mean(np.abs(position[valid] - raw[valid]))),
            "max_displacement": float(np.max(np.abs(position[valid] - raw[valid]))),
            "finite_cell_fraction": float(finite[selected].mean()),
        }
    return result


def heatmap_grid(position, cell_patients, labels, patients):
    grid = np.full((len(patients), 100), np.nan)
    for row, patient in enumerate(patients):
        selected = cell_patients == patient
        values = position[selected]
        types = labels[selected]
        finite = np.isfinite(values)
        bins = np.clip(np.digitize(values[finite], np.linspace(0, 1, 101)) - 1, 0, 99)
        encoded = np.array([TYPES.index(value) / 2 if value in TYPES else np.nan for value in types[finite]])
        for b in range(100):
            contents = encoded[bins == b]
            contents = contents[np.isfinite(contents)]
            if len(contents) >= 3:
                grid[row, b] = contents.mean()
    return grid


def save_heatmap(raw, gamma0_position, gamma1_position, cell_patients, labels, patients):
    cmap = LinearSegmentedColormap.from_list("types", [COLORS[t] for t in TYPES])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), dpi=170, sharey=True)
    for ax, (title, position) in zip(
        axes,
        (("Raw", raw), ("Unregularized joint fit (γ=0)", gamma0_position), ("Smoothed joint fit (γ=1)", gamma1_position)),
        strict=True,
    ):
        ax.imshow(
            np.ma.masked_invalid(heatmap_grid(position, cell_patients, labels, patients)),
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=1,
            extent=[0, 1, len(patients) - 0.5, -0.5],
        )
        ax.set(
            yticks=np.arange(len(patients)),
            yticklabels=patients,
            xlim=(0, 1),
            title=title,
            xlabel="shared position [0,1]",
        )
        ax.set_facecolor("#e8e8e8")
        ru_a = (cell_patients == "RU1303") & (labels == "SCLC-A")
        ax.axvline(np.nanmedian(position[ru_a]), color="black", ls="--", lw=0.8)
    fig.legend(
        handles=[mpatches.Patch(color=COLORS[t], label=t) for t in TYPES]
        + [mpatches.Patch(color="#e8e8e8", label="empty / <3 finite cells")],
        ncol=4,
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    fig.savefig(OUT / "four_patient_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def save_contribution_heatmap(posterior, patients, gamma):
    """Per-patient share of total posterior match mass at each of the 100 consensus landmarks."""
    mass = np.stack([posterior[patient]["matrix"].sum(axis=0) for patient in patients])  # (P,100)
    total = mass.sum(axis=0)
    share = np.divide(mass, total, out=np.zeros_like(mass), where=total > 1e-12)
    fig, ax = plt.subplots(figsize=(11, 2.4), dpi=170)
    im = ax.imshow(share, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=1,
                    extent=[0, 1, len(patients) - 0.5, -0.5])
    ax.set(yticks=np.arange(len(patients)), yticklabels=patients, xlabel="shared position [0,1]",
           title=f"Per-patient share of consensus match mass (γ={gamma:g})")
    fig.colorbar(im, ax=ax, label="fraction of total mass")
    fig.tight_layout()
    fig.savefig(OUT / f"contribution_weight_gamma{gamma:g}.png", bbox_inches="tight")
    plt.close(fig)
    return share


def local_curvature(barycenter):
    d2 = barycenter[2:] - 2 * barycenter[1:-1] + barycenter[:-2]
    return np.sqrt(np.mean(d2**2, axis=1))


def summarize(run, patients, markers, marker_scale, raw, cell_patients, labels, full_patients):
    initial_post, posterior = run["initial_post"], run["posterior"]
    initial_position = cell_positions(initial_post, raw, cell_patients, patients)
    final_position = cell_positions(posterior, raw, cell_patients, patients)
    quality_by_phase = {
        "initial": quality(initial_position, raw, cell_patients, labels, patients),
        "final": quality(final_position, raw, cell_patients, labels, patients),
    }
    rows = []
    for phase, state in (("initial", initial_post), ("final", posterior)):
        for patient in patients:
            p = state[patient]
            rows.append(
                {
                    "gamma": run["gamma"],
                    "phase": phase,
                    "patient": patient,
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
    return {
        "gamma": run["gamma"],
        "converged": run["converged"],
        "iterations": len(run["history"]),
        "history": run["history"],
        "initial_smoothness": smoothness(run["initial_barycenter"]),
        "final_smoothness": smoothness(run["final_barycenter"]),
        "rows": rows,
        "final_position": final_position,
        "final_barycenter": run["final_barycenter"],
        "posterior": posterior,
    }


def main():
    OUT.mkdir(exist_ok=True)
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
    with (OUT / "frozen_gate.json").open("w") as handle:
        json.dump(
            {"scores": scores, "peer_positions": peer_positions, "gate_on": cheap, "tail_prices": prices},
            handle,
            indent=2,
        )
        handle.write("\n")
    full_patients = tuple(p for p in PATIENTS if not cheap[p])

    seed = np.mean([expression[p] for p in full_patients], axis=0)
    graphs = {p: compile_graph(match_cost(expression[p], seed), prices[p]) for p in PATIENTS}

    adata = ad.read_h5ad(INPUT, backed="r")
    assert np.array_equal(fit["cell_ids"], np.asarray(adata.obs_names, str))
    all_patients = adata.obs["patient"].astype(str).to_numpy()
    selected = np.isin(all_patients, PATIENTS)
    cell_patients = all_patients[selected]
    labels = adata.obs["cell_type_final2"].astype(str).to_numpy()[selected]
    raw = adata.obs["s_local"].to_numpy(float)[selected]
    markers = {patient: marker_landmarks(adata, patient, MARKERS) for patient in PATIENTS}
    marker_scale = np.std(np.vstack([markers[p] for p in full_patients]), axis=0)
    marker_scale = np.where(marker_scale > 1e-8, marker_scale, 1)

    summaries = {}
    for gamma in GAMMAS:
        run = run_em(expression, graphs, seed, PATIENTS, gamma)
        summaries[gamma] = summarize(run, PATIENTS, markers, marker_scale, raw, cell_patients, labels, full_patients)

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
            "history": summaries[gamma]["history"],
        }
        for gamma in GAMMAS
    }
    with (OUT / "convergence.json").open("w") as handle:
        json.dump(convergence, handle, indent=2)
        handle.write("\n")

    save_heatmap(raw, summaries[0.0]["final_position"], summaries[1.0]["final_position"], cell_patients, labels, PATIENTS)
    shares = {}
    for gamma in GAMMAS:
        shares[gamma] = save_contribution_heatmap(summaries[gamma]["posterior"], PATIENTS, gamma)
    np.savez_compressed(
        OUT / "pilot_state.npz",
        patient_order=np.asarray(PATIENTS),
        full_patients=np.asarray(full_patients),
        **{f"final_barycenter_gamma{gamma:g}": summaries[gamma]["final_barycenter"] for gamma in GAMMAS},
        **{f"contribution_share_gamma{gamma:g}": shares[gamma] for gamma in GAMMAS},
    )
    report = {
        "patients": PATIENTS,
        "full_patients": full_patients,
        "gate_on": cheap,
        "coverage_scores": scores,
    }
    for gamma in GAMMAS:
        report[f"gamma={gamma:g}"] = {
            "converged": summaries[gamma]["converged"],
            "iterations": summaries[gamma]["iterations"],
            "final_smoothness": summaries[gamma]["final_smoothness"],
        }
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
