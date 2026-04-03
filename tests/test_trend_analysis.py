import anndata as ad
import numpy as np
import pandas as pd

from scdebussy.tl import (
    bicluster_ordered_heatmap,
    cluster_and_order_genes,
    compute_composite_alignment_score,
    compute_gene_trend_features,
    evaluate_basal_signature_prominence,
    scDeBussy,
    temporal_kernel_gene_set_rankings,
    temporal_kernel_pseudotime_enrichment,
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


def test_bicluster_ordered_heatmap_recovers_consecutive_blocks():
    matrix = np.array(
        [
            [5.0, 5.0, 0.0, 0.0],
            [5.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 4.0],
            [0.0, 0.0, 4.0, 4.0],
        ],
        dtype=float,
    )

    result = bicluster_ordered_heatmap(
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
