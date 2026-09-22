"""Stability/generalizability test of the label-free largest-gap coverage gate.

Pure analysis of the already-computed pre-fit coverage scores and their
underlying peer-position data (../auto_gate/frozen_gate.json) -- no alignment
model refit, no gate-rule retuning. Tests:
  1. leave-one-patient-out (drop each of the 8 patients from the score set,
     recompute the gate on the remaining 7),
  2. nonparametric bootstrap of each patient's own score (resample with
     replacement from the 7 peer positions that make up its median; this
     reuses only data already on disk from the coverage-score computation),
  3. an additive Gaussian noise-perturbation sweep at a few fixed sigma
     levels, as a direct stress test of "coverage-score noise" per the task.
No cell-type label, held-out marker, or alignment outcome is used anywhere.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
SOURCE = Path(__file__).resolve().parents[1] / "auto_gate" / "frozen_gate.json"
N_BOOTSTRAP = 2000
NOISE_SIGMAS = (0.02, 0.05, 0.10)
N_NOISE_REPS = 2000
RNG_SEED = 0


def largest_gap_gate(scores, patients):
    """Identical rule to auto_gate.py's largest_gap_gate, operating on a plain
    dict/list of scores instead of recomputing frozen_coverage."""
    ordered = sorted(patients, key=lambda p: scores[p])
    values = [scores[p] for p in ordered]
    gaps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    if not gaps:
        return {patients[0]: False}, {"threshold": None, "largest_gap": None, "second_largest_gap": None}
    split = int(np.argmax(gaps))
    threshold = (values[split] + values[split + 1]) / 2
    cheap = {p: scores[p] < threshold for p in patients}
    sorted_gaps = sorted(gaps, reverse=True)
    return cheap, {
        "threshold": threshold,
        "largest_gap": sorted_gaps[0],
        "second_largest_gap": sorted_gaps[1] if len(sorted_gaps) > 1 else None,
        "gap_ratio": sorted_gaps[0] / sorted_gaps[1] if len(sorted_gaps) > 1 and sorted_gaps[1] > 0 else None,
        "ordered": ordered,
        "values": values,
    }


def main():
    source = json.loads(SOURCE.read_text())
    scores = source["scores"]
    peer_positions = source["peer_positions"]
    patients = list(scores.keys())
    rng = np.random.default_rng(RNG_SEED)

    baseline_gate, baseline_diag = largest_gap_gate(scores, patients)
    print("baseline gate:", baseline_gate)
    print("baseline diagnostics:", {k: v for k, v in baseline_diag.items() if k not in ("ordered", "values")})

    results = {"leave_one_out": [], "bootstrap": [], "noise": {}}

    # 1. Leave-one-patient-out
    for dropped in patients:
        remaining = [p for p in patients if p != dropped]
        gate, diag = largest_gap_gate(scores, remaining)
        on_patients = sorted(p for p, v in gate.items() if v)
        results["leave_one_out"].append(
            {
                "dropped": dropped,
                "gate_on": on_patients,
                "threshold": diag["threshold"],
                "largest_gap": diag["largest_gap"],
                "second_largest_gap": diag["second_largest_gap"],
                "gap_ratio": diag["gap_ratio"],
                "ru1303_isolated": on_patients == ["RU1303"] if dropped != "RU1303" else None,
                "ru1303_present": dropped != "RU1303",
            }
        )

    # 2. Nonparametric bootstrap of each patient's own score (resample its 7
    # peer positions with replacement, recompute the median).
    gate_counts = {p: 0 for p in patients}
    thresholds = []
    gap_ratios = []
    ru1303_only_count = 0
    boot_gate_matrix = []  # (N_BOOTSTRAP, len(patients)) bool
    for _ in range(N_BOOTSTRAP):
        boot_scores = {}
        for p in patients:
            values = np.asarray(list(peer_positions[p].values()))
            sample = rng.choice(values, size=len(values), replace=True)
            boot_scores[p] = float(np.median(sample))
        gate, diag = largest_gap_gate(boot_scores, patients)
        on_patients = sorted(p for p, v in gate.items() if v)
        for p in patients:
            if gate[p]:
                gate_counts[p] += 1
        thresholds.append(diag["threshold"])
        if diag["gap_ratio"] is not None:
            gap_ratios.append(diag["gap_ratio"])
        if on_patients == ["RU1303"]:
            ru1303_only_count += 1
        boot_gate_matrix.append([gate[p] for p in patients])
    results["bootstrap"] = {
        "gate_selection_frequency": {p: gate_counts[p] / N_BOOTSTRAP for p in patients},
        "ru1303_only_split_frequency": ru1303_only_count / N_BOOTSTRAP,
        "threshold_mean": float(np.mean(thresholds)),
        "threshold_std": float(np.std(thresholds)),
        "threshold_min": float(np.min(thresholds)),
        "threshold_max": float(np.max(thresholds)),
        "gap_ratio_mean": float(np.mean(gap_ratios)) if gap_ratios else None,
        "gap_ratio_min": float(np.min(gap_ratios)) if gap_ratios else None,
        "n_replicates": N_BOOTSTRAP,
    }
    boot_gate_matrix = np.asarray(boot_gate_matrix, bool)

    # 3. Additive Gaussian noise-perturbation sweep at fixed sigma levels.
    score_values = np.asarray([scores[p] for p in patients])
    for sigma in NOISE_SIGMAS:
        gate_counts_s = {p: 0 for p in patients}
        ru1303_only_count_s = 0
        thresholds_s = []
        gap_ratios_s = []
        for _ in range(N_NOISE_REPS):
            noisy = score_values + rng.normal(0, sigma, size=len(patients))
            noisy = np.clip(noisy, 0, 1)
            noisy_scores = dict(zip(patients, noisy, strict=True))
            gate, diag = largest_gap_gate(noisy_scores, patients)
            on_patients = sorted(p for p, v in gate.items() if v)
            for p in patients:
                if gate[p]:
                    gate_counts_s[p] += 1
            thresholds_s.append(diag["threshold"])
            if diag["gap_ratio"] is not None:
                gap_ratios_s.append(diag["gap_ratio"])
            if on_patients == ["RU1303"]:
                ru1303_only_count_s += 1
        results["noise"][sigma] = {
            "gate_selection_frequency": {p: gate_counts_s[p] / N_NOISE_REPS for p in patients},
            "ru1303_only_split_frequency": ru1303_only_count_s / N_NOISE_REPS,
            "threshold_mean": float(np.mean(thresholds_s)),
            "threshold_std": float(np.std(thresholds_s)),
            "gap_ratio_mean": float(np.mean(gap_ratios_s)) if gap_ratios_s else None,
            "n_replicates": N_NOISE_REPS,
        }

    (OUT / "stability_results.json").write_text(
        json.dumps({"baseline": {"gate": baseline_gate, "diagnostics": baseline_diag}, **results}, indent=2, default=str)
        + "\n"
    )
    print(json.dumps(results["bootstrap"], indent=2))
    for sigma, r in results["noise"].items():
        print(f"noise sigma={sigma}:", json.dumps(r, indent=2))

    # --- figure 1: sorted coverage scores with inferred threshold across perturbations ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=170)
    ax = axes[0]
    order = sorted(patients, key=lambda p: scores[p])
    xs = np.arange(len(order))
    ax.scatter(xs, [scores[p] for p in order], color="black", zorder=5, label="observed score")
    for p, x in zip(order, xs, strict=True):
        ax.annotate(p, (x, scores[p]), textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
    ax.axhline(baseline_diag["threshold"], color="#2c7fb8", lw=1.5, label="baseline threshold")
    ax.axhspan(
        np.quantile(thresholds, 0.025),
        np.quantile(thresholds, 0.975),
        color="#2c7fb8",
        alpha=0.15,
        label="bootstrap threshold 95% range",
    )
    for sigma, color in zip(NOISE_SIGMAS, ("#f03b20", "#31a354", "#756bb1"), strict=True):
        t = results["noise"][sigma]["threshold_mean"]
        s = results["noise"][sigma]["threshold_std"]
        ax.errorbar([len(order) - 0.5], [t], yerr=[s], fmt="d", color=color, label=f"noise σ={sigma} mean±sd")
    ax.set(xticks=xs, xticklabels=order, ylabel="coverage score", title="Sorted coverage scores and inferred threshold")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=7, loc="lower right")

    # --- figure 2: patient x replicate gate-selection heatmap (bootstrap replicates) ---
    ax2 = axes[1]
    display_order = sorted(patients, key=lambda p: -gate_counts[p])
    idx = [patients.index(p) for p in display_order]
    subset = boot_gate_matrix[:200, :][:, idx].T  # show first 200 replicates for legibility
    ax2.imshow(subset, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1)
    ax2.set(
        yticks=np.arange(len(display_order)),
        yticklabels=display_order,
        xlabel="bootstrap replicate (first 200 of {})".format(N_BOOTSTRAP),
        title="Patient × replicate gate selection (bootstrap; black = gated on)",
    )
    fig.tight_layout()
    fig.savefig(OUT / "gate_stability.png", bbox_inches="tight")
    fig.savefig(OUT / "gate_stability.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "gate_stability.png")


if __name__ == "__main__":
    main()
