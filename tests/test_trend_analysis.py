import anndata as ad
import numpy as np
import pandas as pd

from scdebussy.tl import (
    cluster_and_order_genes,
    compute_composite_alignment_score,
    compute_gene_trend_features,
    cosegment_ordered_heatmap,
    evaluate_basal_signature_prominence,
    scDeBussy,
    temporal_kernel_gene_set_rankings,
    temporal_kernel_pseudotime_enrichment,
)
from scdebussy.tl import (
    test_differential_trajectory as diff_trajectory,
)


def _fit_alignment(adata):
    return scDeBussy(
        adata,
        patient_key="patient",
        pseudotime_key="s_local",
        n_bins=12,
        bandwidth=0.15,
        max_iter=3,
    )


def test_compute_gene_trend_features_returns_expected_schema(sample_adata):
    adata = _fit_alignment(sample_adata.copy())
    features = compute_gene_trend_features(adata)

    assert len(features) == adata.n_vars
    expected = {
        "gene",
        "peak_bin",
        "peak_tau",
        "peak_value",
        "mean_expression",
        "std_expression",
        "temporal_centroid",
        "temporal_spread",
        "rise_total",
        "fall_total",
    }
    assert expected.issubset(set(features.columns))


def test_cluster_and_order_genes_contains_all_genes(sample_adata):
    adata = _fit_alignment(sample_adata.copy())
    ordered = cluster_and_order_genes(adata, n_clusters=3, random_state=1)

    assert len(ordered) == adata.n_vars
    assert set(ordered["gene"]) == set(adata.var_names)
    assert ordered["global_order_rank"].is_monotonic_increasing


def test_cluster_and_order_genes_keeps_edge_bimodal_genes_off_midpoint():
    expression = np.array(
        [
            [1.0, 0.1, 0.0, 1.0],
            [0.7, 0.4, 0.1, 0.2],
            [0.4, 1.0, 0.2, 0.1],
            [0.2, 0.7, 0.5, 0.1],
            [0.1, 0.3, 0.8, 0.2],
            [0.0, 0.1, 1.0, 0.98],
        ],
        dtype=float,
    )
    adata = ad.AnnData(
        X=np.zeros((2, 4), dtype=float),
        obs=pd.DataFrame(index=["cell0", "cell1"]),
        var=pd.DataFrame(index=["early", "middle", "late", "edge_bimodal"]),
    )
    adata.uns["barycenter"] = {"expression": expression}

    ordered = cluster_and_order_genes(adata, n_clusters=1, random_state=0)
    genes = ordered["gene"].tolist()

    assert genes.index("edge_bimodal") < genes.index("middle")
    assert bool(ordered.set_index("gene").loc["edge_bimodal", "edge_bimodal"])
    assert ordered.set_index("gene").loc["edge_bimodal", "ordering_anchor_tau"] == 0.0


def test_cluster_and_order_genes_prioritizes_peak_location_over_shape():
    expression = np.array(
        [
            [1.0, 0.2, 0.0, 1.0, 0.1],
            [0.8, 0.5, 0.2, 0.4, 0.2],
            [0.4, 1.0, 0.5, 0.1, 0.5],
            [0.2, 0.6, 0.9, 0.1, 0.8],
            [0.1, 0.2, 1.0, 0.4, 1.0],
            [0.0, 0.1, 0.8, 0.95, 0.9],
        ],
        dtype=float,
    )
    adata = ad.AnnData(
        X=np.zeros((2, 5), dtype=float),
        obs=pd.DataFrame(index=["cell0", "cell1"]),
        var=pd.DataFrame(index=["early", "mid_early", "late", "edge_bimodal", "late_2"]),
    )
    adata.uns["barycenter"] = {"expression": expression}

    ordered = cluster_and_order_genes(adata, n_clusters=3, random_state=0)
    genes = ordered["gene"].tolist()

    assert genes.index("edge_bimodal") < genes.index("mid_early")
    assert genes.index("edge_bimodal") < genes.index("late")


def test_compute_composite_alignment_score_has_finite_output(sample_adata):
    adata = _fit_alignment(sample_adata.copy())
    result = compute_composite_alignment_score(adata, label_key="cell_type", patient_key="patient", n_neighbors=5)

    assert "composite_score" in result
    assert np.isfinite(result["composite_score"])
    assert set(result["components"]) == {"global_purity", "class_stability", "recurrence"}


def test_cosegment_ordered_heatmap_recovers_consecutive_blocks():
    matrix = np.array(
        [
            [5.0, 5.0, 0.0, 0.0],
            [5.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 4.0],
            [0.0, 0.0, 4.0, 4.0],
        ],
        dtype=float,
    )

    result = cosegment_ordered_heatmap(
        matrix, n_row_clusters=2, n_col_clusters=2, min_rows_per_cluster=1, min_cols_per_cluster=1
    )

    assert result["row_segments"] == [(0, 2), (2, 4)]
    assert result["col_segments"] == [(0, 2), (2, 4)]
    assert result["block_means"].shape == (2, 2)


def test_temporal_kernel_gene_set_rankings_runs(sample_adata):
    adata = _fit_alignment(sample_adata.copy())
    ordered = cluster_and_order_genes(adata)
    ordered_genes = ordered["gene"].tolist()

    ranking = temporal_kernel_gene_set_rankings(
        ordered_genes,
        gene_sets={"set_a": [ordered_genes[0], ordered_genes[-1]], "set_b": [ordered_genes[1]]},
        sigma=2.0,
    )

    assert not ranking.empty
    assert {"gene_set", "center_rank", "kernel_score", "set_rank"}.issubset(set(ranking.columns))


def test_temporal_kernel_pseudotime_enrichment_runs(sample_adata):
    adata = _fit_alignment(sample_adata.copy())
    enrichment = temporal_kernel_pseudotime_enrichment(adata, gene_sets={"set_a": ["gene1", "gene3"]})

    assert not enrichment.empty
    assert {"gene_set", "bin", "enrichment_score", "zscore_within_set"}.issubset(set(enrichment.columns))


def test_evaluate_basal_signature_prominence_returns_stats(sample_adata):
    adata = _fit_alignment(sample_adata.copy())
    ordered = cluster_and_order_genes(adata)
    ordered_genes = ordered["gene"].tolist()

    stats = evaluate_basal_signature_prominence(
        ordered_genes=ordered_genes,
        basal_genes=[ordered_genes[0], ordered_genes[1]],
        n_permutations=100,
        seed=7,
    )

    assert stats["n_hits"] == 2
    assert 0.0 <= stats["permutation_pvalue"] <= 1.0
    assert np.isfinite(stats["effect_size_z"])


# ---------------------------------------------------------------------------
# Helpers for differential trajectory tests
# ---------------------------------------------------------------------------


def _make_condition_adata(n_per_group: int = 5, n_bins_cells: int = 20, seed: int = 0):
    """Synthetic AnnData with 2 condition groups and a planted XDE signal.

    Patients in group B have gene0 shifted upward by 0.5 uniformly (mean-shift)
    and gene1 modulated by a sinusoidal interaction (trend-diff). gene2 is null.
    """
    rng = np.random.default_rng(seed)
    n_patients = n_per_group * 2
    patients, pts, groups, cell_types, X_list = [], [], [], [], []

    for i in range(n_patients):
        pid = f"pat{i}"
        group = "B" if i >= n_per_group else "A"
        pt_local = np.linspace(0.02, 0.98, n_bins_cells) + rng.normal(0, 0.01, n_bins_cells)
        pt_local = np.clip(pt_local, 0.0, 1.0)

        g0 = pt_local + (0.5 if group == "B" else 0.0) + rng.normal(0, 0.05, n_bins_cells)
        g1 = np.sin(np.pi * pt_local) + (0.4 * np.cos(2 * np.pi * pt_local) if group == "B" else 0.0)
        g1 += rng.normal(0, 0.05, n_bins_cells)
        g2 = rng.normal(0, 0.1, n_bins_cells)  # null gene

        patients.extend([pid] * n_bins_cells)
        pts.extend(pt_local.tolist())
        groups.extend([group] * n_bins_cells)
        cell_types.extend(["early" if v < 0.5 else "late" for v in pt_local])
        X_list.append(np.column_stack([g0, g1, g2]))

    X = np.vstack(X_list)
    obs = pd.DataFrame(
        {"patient": patients, "s_local": pts, "condition": groups, "cell_type": cell_types},
        index=[f"c{i}" for i in range(len(patients))],
    )
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=["gene0", "gene1", "gene2"]))
    return adata


def _fit_alignment_condition(adata):
    return scDeBussy(adata, patient_key="patient", pseudotime_key="s_local", n_bins=20, bandwidth=0.15, max_iter=3)


# ---------------------------------------------------------------------------
# test_differential_trajectory tests
# ---------------------------------------------------------------------------


def test_differential_trajectory_returns_expected_schema():
    adata = _fit_alignment_condition(_make_condition_adata())
    results = diff_trajectory(
        adata,
        condition_key="condition",
        n_permutations=50,
        seed=1,
    )
    expected_cols = {
        "gene",
        "tde_stat",
        "tde_pval",
        "tde_fdr",
        "xde_stat",
        "xde_pval",
        "xde_fdr",
        "mean_shift_stat",
        "mean_shift_pval",
        "mean_shift_fdr",
        "trend_diff_stat",
        "trend_diff_pval",
        "trend_diff_fdr",
        "xde_type",
    }
    assert expected_cols.issubset(set(results.columns))
    assert len(results) == adata.n_vars
    assert set(results["gene"]) == set(adata.var_names)


def test_differential_trajectory_stores_results_in_uns():
    adata = _fit_alignment_condition(_make_condition_adata())
    diff_trajectory(adata, condition_key="condition", n_permutations=50, seed=2)

    assert "differential_trajectory" in adata.uns
    stored = adata.uns["differential_trajectory"]
    assert "results" in stored and "params" in stored
    assert stored["params"]["condition_key"] == "condition"
    assert stored["params"]["n_patients"] == 10


def test_differential_trajectory_pvals_in_range():
    adata = _fit_alignment_condition(_make_condition_adata())
    results = diff_trajectory(adata, condition_key="condition", n_permutations=50, seed=3)

    for col in ("tde_pval", "xde_pval", "mean_shift_pval", "trend_diff_pval"):
        assert ((results[col] >= 0.0) & (results[col] <= 1.0)).all(), f"{col} out of [0, 1]"
    for col in ("tde_stat", "xde_stat", "mean_shift_stat", "trend_diff_stat"):
        assert (results[col] >= 0.0).all(), f"{col} has negative values"


def test_differential_trajectory_detects_planted_mean_shift():
    """gene0 has a strong mean-shift signal; should rank lower p-value than null gene2."""
    adata = _fit_alignment_condition(_make_condition_adata(n_per_group=5, seed=42))
    results = diff_trajectory(
        adata,
        condition_key="condition",
        n_permutations=200,
        seed=42,
    )
    # gene0 (mean shift) should have lower XDE p-value than gene2 (null)
    assert results.loc["gene0", "xde_pval"] < results.loc["gene2", "xde_pval"]  # type: ignore[operator]
    assert results.loc["gene0", "mean_shift_pval"] < results.loc["gene2", "mean_shift_pval"]  # type: ignore[operator]


def test_differential_trajectory_xde_type_values():
    adata = _fit_alignment_condition(_make_condition_adata())
    results = diff_trajectory(adata, condition_key="condition", n_permutations=50, seed=5)
    assert set(results["xde_type"]).issubset({"none", "mean_shift", "trend_diff", "both"})


def test_differential_trajectory_gene_subset():
    adata = _fit_alignment_condition(_make_condition_adata())
    results = diff_trajectory(
        adata,
        condition_key="condition",
        genes=["gene0", "gene2"],
        n_permutations=50,
        seed=6,
    )
    assert len(results) == 2
    assert set(results["gene"]) == {"gene0", "gene2"}


def test_differential_trajectory_null_uniform_pvals():
    """Under a null (no condition effect), permutation p-values should be approximately uniform."""
    rng = np.random.default_rng(99)
    n_patients, n_cells = 8, 15
    patients = np.repeat([f"p{i}" for i in range(n_patients)], n_cells)
    pt = np.tile(np.linspace(0.0, 1.0, n_cells), n_patients)
    # Random condition assignment (no real effect)
    condition = np.repeat(rng.choice(["A", "B"], size=n_patients), n_cells)
    X = rng.normal(0, 1, (n_patients * n_cells, 5))
    obs = pd.DataFrame(
        {"patient": patients, "s_local": pt, "condition": condition},
        index=[f"c{i}" for i in range(len(patients))],
    )
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(5)]))
    adata = scDeBussy(adata, patient_key="patient", pseudotime_key="s_local", n_bins=15, bandwidth=0.15, max_iter=2)

    results = diff_trajectory(adata, condition_key="condition", n_permutations=200, seed=0)
    # Under the null, median p-value should be near 0.5 (not stochastically dominated by small values)
    assert results["xde_pval"].median() > 0.1
