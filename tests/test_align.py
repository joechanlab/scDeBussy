import numpy as np
import pytest
from matplotlib import pyplot as plt
from scripts.benchmark.compare_methods import (
    _filter_by_tuning,
    _pairwise_only_subset,
    _single_axis_subset,
    build_global_ranking,
    build_head_to_head_table,
)
from scripts.benchmark.methods import available_methods

from scdebussy.pl import (
    plot_em_convergence,
    plot_running_enrichment_ridge,
    plot_runtime_performance_tradeoff,
    plot_sweep_history,
)
from scdebussy.tl import scDeBussy, smooth_patient_trajectory


def test_scDeBussy_writes_results_to_anndata(sample_adata):
    result = scDeBussy(
        sample_adata,
        patient_key="patient",
        pseudotime_key="s_local",
        n_bins=12,
        bandwidth=0.15,
        max_iter=3,
    )

    assert result is sample_adata
    assert "aligned_pseudotime" in sample_adata.obs
    assert "barycenter" in sample_adata.uns

    barycenter = sample_adata.uns["barycenter"]
    assert barycenter["expression"].shape == (12, sample_adata.n_vars)
    assert len(barycenter["aligned_pseudotime"]) == 12
    assert barycenter["patient_ids"] == ["patient1", "patient2"]
    assert len(barycenter["warp_paths"]) == 2
    assert "em_convergence" in barycenter
    assert len(barycenter["em_convergence"]["iteration_costs"]) >= 1
    assert "converged" in barycenter["em_convergence"]
    assert sample_adata.obs["aligned_pseudotime"].notna().all()


def test_plot_em_convergence_returns_figure(sample_adata):
    scDeBussy(
        sample_adata,
        patient_key="patient",
        pseudotime_key="s_local",
        n_bins=12,
        bandwidth=0.15,
        max_iter=3,
    )

    fig = plot_em_convergence(sample_adata, barycenter_key="barycenter")
    assert fig is not None
    assert hasattr(fig, "axes")


def test_plot_sweep_history_and_runtime_tradeoff_return_figures():
    import pandas as pd

    sweep_df = pd.DataFrame(
        [
            {"trial": 1, "objective_score": 0.42, "status": "ok"},
            {"trial": 2, "objective_score": 0.31, "status": "ok"},
            {"trial": 3, "objective_score": 0.36, "status": "ok"},
        ]
    )
    fig_sweep = plot_sweep_history(sweep_df)
    assert fig_sweep is not None

    ranking_df = pd.DataFrame(
        [
            {"method": "scdebussy", "runtime_s": 5.2, "aligned_global_pearson_r": 0.93},
            {"method": "genes2genes_consensus", "runtime_s": 1800.0, "aligned_global_pearson_r": 0.88},
        ]
    )
    fig_tradeoff = plot_runtime_performance_tradeoff(ranking_df)
    assert fig_tradeoff is not None


def test_plot_running_enrichment_ridge_draws_transition_window_boundaries():
    import pandas as pd

    ordered_genes = ["geneA", "geneB", "geneC", "geneD", "geneE"]
    enrichment_df = pd.DataFrame(
        {
            "gene_set": ["set1"] * 5 + ["set2"] * 5,
            "position": list(range(5)) + list(range(5)),
            "score": [0.2, 0.4, 0.8, 0.5, 0.3, 0.1, 0.5, 0.9, 0.4, 0.2],
        }
    )
    gene_curve = pd.DataFrame(
        {
            "pseudotime": [0.0, 0.5, 1.0],
            "geneA": [1.0, 0.1, 0.0],
            "geneB": [0.2, 1.0, 0.1],
            "geneC": [0.1, 0.9, 0.2],
            "geneD": [0.0, 0.2, 1.0],
            "geneE": [0.0, 0.1, 1.0],
        }
    )

    fig = plot_running_enrichment_ridge(
        enrichment_df,
        gene_sets_to_plot=["set1", "set2"],
        ordered_genes=ordered_genes,
        gene_curve=gene_curve,
        transition_window=(0.4, 0.6),
    )

    ridge_axes = fig.axes[1:]
    assert ridge_axes
    for ax in ridge_axes:
        vertical_lines = [
            line
            for line in ax.lines
            if len(line.get_xdata()) == 2 and np.allclose(line.get_xdata(), line.get_xdata()[0])
        ]
        assert len(vertical_lines) == 2
        assert sorted(line.get_xdata()[0] for line in vertical_lines) == [1.0, 3.0]
    plt.close(fig)


def test_plot_running_enrichment_ridge_transition_window_no_overlap_is_noop():
    import pandas as pd

    ordered_genes = ["geneA", "geneB", "geneC"]
    enrichment_df = pd.DataFrame(
        {
            "gene_set": ["set1"] * 3,
            "position": [0, 1, 2],
            "score": [0.2, 0.4, 0.6],
        }
    )
    gene_curve = pd.DataFrame(
        {
            "pseudotime": [0.0, 0.5, 1.0],
            "geneA": [1.0, 0.0, 0.0],
            "geneB": [0.0, 1.0, 0.0],
            "geneC": [0.0, 0.0, 1.0],
        }
    )

    fig = plot_running_enrichment_ridge(
        enrichment_df,
        gene_sets_to_plot=["set1"],
        ordered_genes=ordered_genes,
        gene_curve=gene_curve,
        transition_window=(1.5, 1.8),
    )

    ridge_axes = fig.axes[1:]
    assert ridge_axes
    for ax in ridge_axes:
        vertical_lines = [
            line
            for line in ax.lines
            if len(line.get_xdata()) == 2 and np.allclose(line.get_xdata(), line.get_xdata()[0])
        ]
        assert len(vertical_lines) == 0
    plt.close(fig)


def test_plot_running_enrichment_ridge_transition_window_requires_gene_curve():
    import pandas as pd

    enrichment_df = pd.DataFrame(
        {
            "gene_set": ["set1"],
            "position": [0],
            "score": [0.1],
        }
    )

    with pytest.raises(ValueError, match="transition_window requires gene_curve"):
        plot_running_enrichment_ridge(
            enrichment_df,
            gene_sets_to_plot=["set1"],
            ordered_genes=["geneA"],
            transition_window=(0.2, 0.4),
        )


def test_scDeBussy_logs_iteration_rmse_for_synthetic_truth(sample_adata):
    adata = sample_adata.copy()
    adata.obs["tau_global"] = adata.obs["s_local"].to_numpy(dtype=float)

    scDeBussy(
        adata,
        patient_key="patient",
        pseudotime_key="s_local",
        n_bins=12,
        bandwidth=0.15,
        max_iter=3,
    )

    em = adata.uns["barycenter"]["em_convergence"]
    assert "iteration_rmse_to_truth" in em
    assert len(em["iteration_rmse_to_truth"]) == len(em["iteration_costs"])
    assert em["rmse_truth_key"] == "tau_global"
    assert em["baseline_rmse_to_truth"] is not None
    assert em["final_rmse_to_truth"] is not None


def test_scDeBussy_preserves_patientwise_monotonicity(sample_adata):
    scDeBussy(
        sample_adata,
        patient_key="patient",
        pseudotime_key="s_local",
        n_bins=10,
        bandwidth=0.2,
        max_iter=2,
    )

    for _, patient_obs in sample_adata.obs.groupby("patient"):
        sorted_obs = patient_obs.sort_values("s_local")
        aligned = sorted_obs["aligned_pseudotime"].to_numpy()
        assert np.all(np.diff(aligned) >= -1e-8)


def test_scDeBussy_validates_required_columns(sample_adata):
    broken = sample_adata.copy()
    del broken.obs["patient"]

    with pytest.raises(ValueError, match="patient"):
        scDeBussy(broken, patient_key="patient", pseudotime_key="s_local")


def test_scDeBussy_accepts_advanced_dtw_and_barycenter_args(sample_adata):
    result = scDeBussy(
        sample_adata,
        patient_key="patient",
        pseudotime_key="s_local",
        n_bins=12,
        bandwidth=0.15,
        gamma=0.1,
        max_iter=3,
        dtw_dist_method="euclidean",
        dtw_step_pattern="asymmetric",
        dtw_open_begin=False,
        dtw_open_end=True,
        dtw_window_fraction=0.3,
        barycenter_init_iter=3,
        barycenter_update_iter=2,
    )

    params = result.uns["barycenter"]["params"]
    assert params["dtw_dist_method"] == "euclidean"
    assert params["dtw_step_pattern"] == "asymmetric"
    assert params["dtw_open_begin"] is False
    assert params["dtw_open_end"] is True
    assert params["dtw_window_fraction"] == 0.3
    assert params["barycenter_init_iter"] == 3
    assert params["barycenter_update_iter"] == 2


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"dtw_dist_method": "invalid"}, "dtw_dist_method"),
        ({"dtw_step_pattern": "invalid"}, "dtw_step_pattern"),
        ({"dtw_open_begin": "yes"}, "dtw_open_begin"),
        ({"dtw_open_end": "no"}, "dtw_open_end"),
        ({"dtw_window_fraction": 0.0}, "dtw_window_fraction"),
        ({"dtw_window_fraction": 1.2}, "dtw_window_fraction"),
        ({"barycenter_init_iter": 0}, "barycenter_init_iter"),
        ({"barycenter_update_iter": 0}, "barycenter_update_iter"),
    ],
)
def test_scDeBussy_validates_advanced_parameters(sample_adata, kwargs, match):
    with pytest.raises(ValueError, match=match):
        scDeBussy(sample_adata, **kwargs)


def test_smooth_patient_trajectory_returns_expected_shapes(sample_adata):
    smoothed, density = smooth_patient_trajectory(
        sample_adata,
        patient_id="patient1",
        patient_key="patient",
        pseudotime_key="s_local",
        n_bins=9,
        bandwidth=0.1,
    )

    assert tuple(smoothed.shape) == (9, sample_adata.n_vars)
    assert tuple(density.shape) == (9,)
    assert np.all(density.numpy() > 0)


def test_compare_methods_excludes_pairwise_only_from_single_axis_ranking():
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "method": "scdebussy",
                "evaluation_mode": "single_axis",
                "scenario": "clean_baseline",
                "seed": 0,
                "aligned_global_pearson_r": 0.95,
                "runtime_s": 1.0,
            },
            {
                "method": "genes2genes_pairwise",
                "evaluation_mode": "pairwise_only",
                "scenario": "clean_baseline",
                "seed": 0,
                "aligned_global_pearson_r": np.nan,
                "runtime_s": 10.0,
                "unsupervised_method_fraction_failed_pairs": 0.0,
            },
        ]
    )

    axis_df = _single_axis_subset(df)
    pairwise_df = _pairwise_only_subset(df)
    ranking = build_global_ranking(axis_df)

    assert list(axis_df["method"]) == ["scdebussy"]
    assert list(pairwise_df["method"]) == ["genes2genes_pairwise"]
    assert list(ranking["method"]) == ["scdebussy"]


def test_compare_methods_handles_pairwise_only_inputs_without_ranking_failure():
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "method": "genes2genes_pairwise",
                "evaluation_mode": "pairwise_only",
                "scenario": "clean_baseline",
                "seed": 0,
                "aligned_global_pearson_r": np.nan,
                "runtime_s": 10.0,
                "unsupervised_method_fraction_failed_pairs": 0.0,
            }
        ]
    )

    axis_df = _single_axis_subset(df)
    ranking = build_global_ranking(axis_df)
    h2h = build_head_to_head_table(axis_df)

    assert axis_df.empty
    assert ranking.empty
    assert h2h.empty


def test_compare_methods_can_filter_to_tuned_rows_only():
    import pandas as pd

    df = pd.DataFrame(
        [
            {"method": "scdebussy", "scenario": "clean_baseline", "seed": 0, "tuning_enabled": True},
            {"method": "identity", "scenario": "clean_baseline", "seed": 0, "tuning_enabled": False},
        ]
    )

    filtered = _filter_by_tuning(df, "true")

    assert list(filtered["method"]) == ["scdebussy"]


def test_compare_methods_global_ranking_keeps_tuning_columns():
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "method": "scdebussy",
                "evaluation_mode": "single_axis",
                "scenario": "clean_baseline",
                "seed": 0,
                "aligned_global_pearson_r": 0.95,
                "runtime_s": 1.0,
                "tuning_enabled": True,
                "tuning_n_trials": 30,
                "tuning_best_score": 0.12,
            },
            {
                "method": "cellalign_consensus",
                "evaluation_mode": "single_axis",
                "scenario": "clean_baseline",
                "seed": 0,
                "aligned_global_pearson_r": 0.91,
                "runtime_s": 2.0,
                "tuning_enabled": True,
                "tuning_n_trials": 30,
                "tuning_best_score": 0.18,
            },
        ]
    )

    ranking = build_global_ranking(df)

    assert {"tuning_enabled", "tuning_n_trials", "tuning_best_score"}.issubset(ranking.columns)
    assert ranking["tuning_enabled"].tolist() == [True, True]


def test_genes2genes_ambiguous_alias_is_not_registered():
    methods = available_methods()

    assert "genes2genes" not in methods
    assert "genes2genes_fixed_reference" in methods
    assert "genes2genes_consensus" in methods
    assert "genes2genes_pairwise" in methods
