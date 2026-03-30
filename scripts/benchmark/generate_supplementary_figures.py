#!/usr/bin/env python3
"""Generate supplementary figures for parameter search and EM convergence.

This script creates:
1) Parameter-search diagnostics figure from a lightweight tuning sweep.
2) EM convergence figure from a representative scDeBussy alignment run.

Outputs are written as both PDF and PNG.

PYTHONPATH=/data1/chanj3/wangm10/scDeBussy conda run -p /usersoftware/chanj3/scdebussy-py311 \
    python scripts/benchmark/generate_supplementary_figures.py \
        --output-dir /scratch/chanj3/wangm10/compare_run_tuning/figures/supplementary \
        --scenario combined_challenge --seed 5 --n-trials 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scripts.benchmark.scenarios import SCENARIO_REGISTRY

import scdebussy
from scdebussy import tune_scdebussy
from scdebussy.pl import plot_em_convergence
from scdebussy.tl import initialize_structured_loadings, simulate_LF_MOGP


def _build_adata_for_scenario(scenario_name: str, seed: int):
    """Construct a synthetic AnnData object for a named benchmark scenario.

    Parameters
    ----------
    scenario_name : str
        Scenario key in ``SCENARIO_REGISTRY``.
    seed : int
        Random seed used for reproducible loadings and simulation draws.

    Returns
    -------
    tuple
        ``(adata, scenario)`` where ``adata`` is the simulated AnnData object
        and ``scenario`` is the scenario configuration dictionary.
    """
    scenario = SCENARIO_REGISTRY[scenario_name]
    sim_kwargs = dict(scenario["sim_kwargs"])

    k_genes = int(sim_kwargs["K"])
    n_factors = int(sim_kwargs["M"])
    time_grid = np.linspace(0.0, 1.0, 50)
    sim_kwargs["time_grid"] = time_grid

    rng = np.random.default_rng(seed)
    loadings, gene_categories = initialize_structured_loadings(K=k_genes, M=n_factors, rng=rng)

    adata = simulate_LF_MOGP(
        **sim_kwargs,
        W=loadings,
        gene_categories=gene_categories,
        rng=np.random.default_rng(seed),
    )
    return adata, scenario


def _save_figure(fig, output_stem: Path) -> None:
    """Save a Matplotlib figure as both PDF and PNG and close it.

    Parameters
    ----------
    fig
        Matplotlib figure instance to serialize.
    output_stem : Path
        Output path stem (without extension).
    """
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _apply_title(plot_obj, title: str) -> None:
    """Apply a title whether plotting API returned Figure or Axes."""
    if hasattr(plot_obj, "suptitle"):
        plot_obj.suptitle(title, y=1.02)
        return
    if hasattr(plot_obj, "set_title"):
        plot_obj.set_title(title)


def _plot_rmse_convergence(adata, barycenter_key: str = "barycenter"):
    """Plot per-iteration RMSE-to-truth when synthetic truth is available."""
    bary = adata.uns.get(barycenter_key, {})
    em = bary.get("em_convergence", {}) if isinstance(bary, dict) else {}
    rmse_curve = em.get("iteration_rmse_to_truth")

    if not rmse_curve:
        return None

    iterations = np.arange(1, len(rmse_curve) + 1)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(iterations, rmse_curve, marker="o", linewidth=1.8, markersize=4, color="#d62728")
    ax.set_xlabel("EM Iteration")
    ax.set_ylabel("RMSE to Ground Truth")
    ax.set_title("EM RMSE Convergence")
    ax.grid(True, alpha=0.3)

    baseline = em.get("baseline_rmse_to_truth")
    final_rmse = em.get("final_rmse_to_truth")
    if baseline is not None:
        ax.axhline(float(baseline), linestyle="--", linewidth=1.2, color="#7f7f7f", label="Baseline RMSE")
    if final_rmse is not None:
        ax.axhline(float(final_rmse), linestyle=":", linewidth=1.5, color="#2ca02c", label="Final RMSE")
    if baseline is not None or final_rmse is not None:
        ax.legend(frameon=False)

    plt.tight_layout()
    return fig


def _plot_parameter_search_dashboard(sweep_df, objective_col: str = "objective_score"):
    """Plot objective trajectory plus key parameter diagnostics."""
    ok = sweep_df.copy()
    if "status" in ok.columns:
        ok = ok[ok["status"] == "ok"].copy()
    if objective_col not in ok.columns:
        raise ValueError(f"Expected {objective_col!r} in sweep_df.")

    ok = ok[np.isfinite(ok[objective_col].to_numpy(dtype=float))].copy()
    if ok.empty:
        raise ValueError("No successful finite-objective trials in sweep_df.")

    ok = ok.sort_values("trial") if "trial" in ok.columns else ok.reset_index(drop=True)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axs = axes.ravel()

    # Panel 1: objective vs trial with running best.
    trials = ok["trial"].to_numpy(dtype=float) if "trial" in ok.columns else np.arange(1, len(ok) + 1, dtype=float)
    objective = ok[objective_col].to_numpy(dtype=float)
    running_best = np.minimum.accumulate(objective)
    best_idx = int(np.argmin(objective))

    axs[0].plot(trials, objective, marker="o", linewidth=1.5, markersize=4, color="#1f77b4", label="Trial")
    axs[0].plot(trials, running_best, linewidth=2.0, color="#2ca02c", label="Running best")
    axs[0].scatter([trials[best_idx]], [objective[best_idx]], color="#d62728", zorder=3, label="Best trial")
    axs[0].set_xlabel("Trial")
    axs[0].set_ylabel(objective_col)
    axs[0].set_title("Objective Trajectory")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(frameon=False, fontsize=8)

    # Panels 2-4: objective vs key numeric hyperparameters.
    numeric_params = ["n_bins", "bandwidth", "gamma"]
    for i, col in enumerate(numeric_params, start=1):
        ax = axs[i]
        if col in ok.columns:
            x = ok[col].to_numpy(dtype=float)
            ax.scatter(x, objective, s=28, alpha=0.75, color="#1f77b4")
            if np.nanmax(x) > 0 and np.nanmax(x) / max(np.nanmin(x[x > 0]) if np.any(x > 0) else 1, 1e-8) > 20:
                ax.set_xscale("log")
            ax.set_xlabel(col)
            if i % 3 == 0:
                ax.set_ylabel(objective_col)
            else:
                ax.set_ylabel("")
            ax.set_title(f"{col}")
            ax.grid(True, alpha=0.3)
        else:
            ax.set_axis_off()

    # Panel 5: categorical dist-method effect.
    if "dtw_dist_method" in ok.columns:
        labels = sorted(ok["dtw_dist_method"].dropna().astype(str).unique().tolist())
        data = [
            ok.loc[ok["dtw_dist_method"].astype(str) == label, objective_col].to_numpy(dtype=float) for label in labels
        ]
        axs[4].boxplot(data, tick_labels=labels, patch_artist=True)
        axs[4].set_title("DTW distance method")
        axs[4].set_ylabel("")
        axs[4].grid(True, alpha=0.3, axis="y")
    else:
        axs[4].set_axis_off()

    # Panel 6: categorical step-pattern effect (top 6 most frequent).
    if "dtw_step_pattern" in ok.columns:
        counts = ok["dtw_step_pattern"].astype(str).value_counts()
        labels = counts.index.tolist()[:6]
        data = [
            ok.loc[ok["dtw_step_pattern"].astype(str) == label, objective_col].to_numpy(dtype=float) for label in labels
        ]
        axs[5].boxplot(data, tick_labels=labels, patch_artist=True)
        axs[5].set_title("DTW step pattern")
        axs[5].set_ylabel("")
        axs[5].tick_params(axis="x", rotation=30)
        axs[5].grid(True, alpha=0.3, axis="y")
    else:
        axs[5].set_axis_off()

    # Add panel labels (A-F) in the upper-left corner of each visible axis.
    panel_labels = ["A", "B", "C", "D", "E", "F"]
    for ax, label in zip(axs, panel_labels, strict=False):
        if not ax.axison:
            continue
        ax.text(
            -0.16,
            1.08,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
            clip_on=False,
        )

    plt.tight_layout()
    return fig


def main() -> None:
    """Run the supplementary-figure generation workflow from CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate supplementary EM and parameter-search figures.")
    parser.add_argument(
        "--output-dir",
        default="/scratch/chanj3/wangm10/compare_run_tuning/figures/supplementary",
        help="Directory for figure outputs.",
    )
    parser.add_argument(
        "--scenario",
        default="combined_challenge",
        choices=sorted(SCENARIO_REGISTRY.keys()),
        help="Benchmark scenario used for representative figure generation.",
    )
    parser.add_argument("--seed", type=int, default=5, help="Random seed for synthetic data generation.")
    parser.add_argument("--n-trials", type=int, default=30, help="Tuning trials for sweep history figure.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adata, scenario = _build_adata_for_scenario(args.scenario, args.seed)

    base_params = dict(scenario["scdebussy_params"])
    tuning_result = tune_scdebussy(
        adata,
        method_params=base_params,
        n_trials=args.n_trials,
        seed=42,
        objective="unsupervised",
        verbose=False,
    )

    fig_sweep = _plot_parameter_search_dashboard(tuning_result.sweep_df, objective_col="objective_score")
    _apply_title(
        fig_sweep,
        f"Parameter Search Diagnostics ({args.scenario}, seed={args.seed}, trials={args.n_trials})",
    )
    _save_figure(fig_sweep, out_dir / "parameter_search")

    best_params = dict(tuning_result.best_params)
    adata_em, _ = _build_adata_for_scenario(args.scenario, args.seed)
    adata_em = scdebussy.tl.scDeBussy(adata_em, **best_params)

    fig_em = plot_em_convergence(adata_em, barycenter_key=best_params.get("barycenter_key", "barycenter"))
    _apply_title(
        fig_em,
        f"EM Convergence ({args.scenario}, seed={args.seed})",
    )
    _save_figure(fig_em, out_dir / "em_convergence")

    fig_rmse = _plot_rmse_convergence(adata_em, barycenter_key=best_params.get("barycenter_key", "barycenter"))
    if fig_rmse is not None:
        _apply_title(
            fig_rmse,
            f"RMSE Convergence ({args.scenario}, seed={args.seed})",
        )
        _save_figure(fig_rmse, out_dir / "em_rmse_convergence")

    print("Generated supplementary figures:")
    print(f"  - {(out_dir / 'parameter_search.pdf')}")
    print(f"  - {(out_dir / 'parameter_search.png')}")
    print(f"  - {(out_dir / 'em_convergence.pdf')}")
    print(f"  - {(out_dir / 'em_convergence.png')}")
    if fig_rmse is not None:
        print(f"  - {(out_dir / 'em_rmse_convergence.pdf')}")
        print(f"  - {(out_dir / 'em_rmse_convergence.png')}")
    print(f"Scenario: {args.scenario} | Seed: {args.seed} | Trials: {args.n_trials}")


if __name__ == "__main__":
    main()
