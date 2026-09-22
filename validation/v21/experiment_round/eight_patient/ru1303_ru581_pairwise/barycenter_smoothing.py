"""Diagnose the jagged 3-patient barycenter and test curvature-regularized M-steps.

Reuses the frozen coverage gate, pairwise graph, and EM loop from
`three_patient_joint.py` unchanged. Only the M-step is replaced by

    min_B  sum_j w_j || B_j - Bhat_j ||^2 + gamma * sum_j || B_{j+1} - 2 B_j + B_{j-1} ||^2

for a small sweep of gamma. No alignment/gating parameter is retuned.
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
from run_pairwise import COLORS, INPUT, MARKERS, TYPES, marker_landmarks
from three_patient_joint import (
    FULL_PATIENTS,
    MAX_ITER,
    PATIENCE,
    PATIENTS,
    TOL,
    cell_positions,
    compile_graph,
    expectation,
    frozen_coverage,
    heatmap_grid,
    marker_mse,
    match_cost,
    quality,
    smoothness,
)
from three_patient_joint import CACHE, ROOT
from three_patient_joint import barycenter_update as plain_barycenter_update
from coverage_gate import CHEAP_PRICE
from focused_followup import ETA

OUT = Path(__file__).resolve().parent / "barycenter_smoothing"
GAMMAS = (0.0, 0.1, 1.0)

_SECOND_DIFF = None


def second_diff_operator():
    """Build the fixed (98,100) second-difference operator once."""
    global _SECOND_DIFF
    if _SECOND_DIFF is None:
        d = np.zeros((98, 100))
        idx = np.arange(98)
        d[idx, idx] = 1.0
        d[idx, idx + 1] = -2.0
        d[idx, idx + 2] = 1.0
        _SECOND_DIFF = d
    return _SECOND_DIFF


def barycenter_update_smooth(x, posterior, previous, gamma):
    """Curvature-regularized correspondence-weighted M-step.

    numerator already equals w_j * Bhat_j, so solving
    (diag(w) + gamma * D^T D) B = numerator gives the smoothed update directly,
    without materializing the unconstrained Bhat.
    """
    numerator = sum(posterior[patient]["matrix"].T @ x[patient] for patient in PATIENTS)
    denominator = sum(posterior[patient]["matrix"].sum(axis=0) for patient in PATIENTS)
    if gamma == 0.0:
        return plain_barycenter_update(x, posterior, previous)
    d = second_diff_operator()
    a = np.diag(denominator) + gamma * (d.T @ d)
    updated = np.linalg.solve(a, numerator)
    return updated, denominator


def run_em(expression, graphs, seed, gamma):
    """Run the frozen EM loop with a curvature-regularized M-step at strength gamma."""
    seed_post = expectation(expression, seed, graphs)
    initial, _ = barycenter_update_smooth(expression, seed_post, seed, gamma)
    posterior = expectation(expression, initial, graphs)
    initial_post = posterior
    barycenter = initial
    history = []
    stable = 0
    converged = False
    for iteration in range(1, MAX_ITER + 1):
        before = -sum(posterior[patient]["log_z"] for patient in PATIENTS)
        updated, support = barycenter_update_smooth(expression, posterior, barycenter, gamma)
        following = expectation(expression, updated, graphs)
        after = -sum(following[patient]["log_z"] for patient in PATIENTS)
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


def summarize(run, expression, markers, marker_scale, raw, patients, labels):
    """Collect the same diagnostics reported for the unregularized pilot."""
    initial_post, posterior = run["initial_post"], run["posterior"]
    initial_position = cell_positions(initial_post, raw, patients)
    final_position = cell_positions(posterior, raw, patients)
    quality_by_phase = {
        "initial": quality(initial_position, raw, patients, labels),
        "final": quality(final_position, raw, patients, labels),
    }
    rows = []
    for phase, state in (("initial", initial_post), ("final", posterior)):
        for patient in PATIENTS:
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
                    "heldout_marker_mse": marker_mse(patient, state, markers, marker_scale),
                    **quality_by_phase[phase][patient],
                }
            )
    return {
        "gamma": run["gamma"],
        "converged": run["converged"],
        "iterations": len(run["history"]),
        "initial_smoothness": smoothness(run["initial_barycenter"]),
        "final_smoothness": smoothness(run["final_barycenter"]),
        "rows": rows,
        "final_position": final_position,
    }


def save_figure(raw, patients, labels, unreg_position, best_position, best_gamma, gammas, roughness):
    """Raw | unregularized joint fit | best smoothed joint fit, plus a roughness plot."""
    cmap = LinearSegmentedColormap.from_list("types", [COLORS[t] for t in TYPES])
    fig = plt.figure(figsize=(14, 5.6), dpi=170)
    grid = fig.add_gridspec(2, 3, height_ratios=(2.4, 1.6))
    panels = (
        ("Raw", raw),
        ("Unregularized joint fit (γ=0)", unreg_position),
        (f"Smoothed joint fit (γ={best_gamma:g})", best_position),
    )
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    for ax, (title, position) in zip(axes, panels, strict=True):
        ax.imshow(
            np.ma.masked_invalid(heatmap_grid(position, patients, labels)),
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=1,
            extent=[0, 1, len(PATIENTS) - 0.5, -0.5],
        )
        ax.set(
            yticks=np.arange(len(PATIENTS)),
            yticklabels=PATIENTS,
            xlim=(0, 1),
            title=title,
            xlabel="shared position [0,1]",
        )
        ax.set_facecolor("#e8e8e8")
        ru_a = (patients == "RU1303") & (labels == "SCLC-A")
        ax.axvline(np.nanmedian(position[ru_a]), color="black", ls="--", lw=0.8)
    fig.legend(
        handles=[mpatches.Patch(color=COLORS[t], label=t) for t in TYPES]
        + [mpatches.Patch(color="#e8e8e8", label="empty / <3 finite cells")],
        ncol=4,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.55),
    )
    ax = fig.add_subplot(grid[1, :])
    for gamma in gammas:
        ax.plot(np.arange(1, 99), roughness[gamma], label=f"γ={gamma:g}", lw=1.2)
    ax.set(xlabel="landmark index", ylabel="local RMS 2nd difference", yscale="log")
    ax.legend(fontsize=8, ncol=len(gammas))
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "barycenter_smoothing_figure.png", bbox_inches="tight")
    plt.close(fig)


def local_curvature(barycenter):
    d2 = barycenter[2:] - 2 * barycenter[1:-1] + barycenter[:-2]
    return np.sqrt(np.mean(d2**2, axis=1))


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
    scores, _peer_positions, cheap = frozen_coverage(expression)
    prices = {patient: CHEAP_PRICE if cheap[patient] else ETA for patient in PATIENTS}
    seed = (expression["RU581"] + expression["RU1083"]) / 2
    graphs = {patient: compile_graph(match_cost(expression[patient], seed), prices[patient]) for patient in PATIENTS}

    adata = ad.read_h5ad(INPUT, backed="r")
    all_patients = adata.obs["patient"].astype(str).to_numpy()
    selected = np.isin(all_patients, PATIENTS)
    patients = all_patients[selected]
    labels = adata.obs["cell_type_final2"].astype(str).to_numpy()[selected]
    raw = adata.obs["s_local"].to_numpy(float)[selected]
    markers = {patient: marker_landmarks(adata, patient, MARKERS) for patient in PATIENTS}
    marker_scale = np.std(np.vstack(list(markers.values())), axis=0)
    marker_scale = np.where(marker_scale > 1e-8, marker_scale, 1)

    summaries = {}
    roughness = {}
    for gamma in GAMMAS:
        run = run_em(expression, graphs, seed, gamma)
        summaries[gamma] = summarize(run, expression, markers, marker_scale, raw, patients, labels)
        roughness[gamma] = local_curvature(run["final_barycenter"])
        summaries[gamma]["_final_barycenter"] = run["final_barycenter"]

    unreg = summaries[0.0]
    candidates = [g for g in GAMMAS if g > 0]
    best_gamma = min(
        candidates,
        key=lambda g: summaries[g]["final_smoothness"]
        if not np.isnan(summaries[g]["final_smoothness"])
        else np.inf,
    )
    # Prefer the smallest gamma whose final smoothness is close to (within 20% of)
    # the smallest achieved, to avoid over-smoothing beyond what's needed.
    min_smooth = min(summaries[g]["final_smoothness"] for g in candidates)
    for g in sorted(candidates):
        if summaries[g]["final_smoothness"] <= 1.2 * min_smooth:
            best_gamma = g
            break

    with (OUT / "metrics.csv").open("w", newline="") as handle:
        all_rows = [row for gamma in GAMMAS for row in summaries[gamma]["rows"]]
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)

    report = {
        gamma: {
            "converged": summaries[gamma]["converged"],
            "iterations": summaries[gamma]["iterations"],
            "initial_smoothness_rms_second_difference": summaries[gamma]["initial_smoothness"],
            "final_smoothness_rms_second_difference": summaries[gamma]["final_smoothness"],
        }
        for gamma in GAMMAS
    }
    report["best_gamma"] = best_gamma
    with (OUT / "convergence_summary.json").open("w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    save_figure(
        raw,
        patients,
        labels,
        summaries[0.0]["final_position"],
        summaries[best_gamma]["final_position"],
        best_gamma,
        GAMMAS,
        roughness,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
