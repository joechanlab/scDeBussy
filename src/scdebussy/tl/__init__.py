from ._alignment import scDeBussy, smooth_patient_trajectory
from ._metrics import cross_batch_knn_purity, cross_batch_purity_sweep
from ._synthetic_dataset import initialize_structured_loadings, simulate_LF_MOGP
from ._trend_analysis import (
    bicluster_ordered_heatmap,
    cluster_and_order_genes,
    compute_composite_alignment_score,
    compute_gene_trend_features,
    compute_running_enrichment,
    compute_trend_recurrence_score,
    evaluate_basal_signature_prominence,
    temporal_kernel_gene_set_rankings,
    temporal_kernel_pseudotime_enrichment,
)

__all__ = [
    "scDeBussy",
    "smooth_patient_trajectory",
    "cross_batch_knn_purity",
    "cross_batch_purity_sweep",
    "compute_gene_trend_features",
    "bicluster_ordered_heatmap",
    "cluster_and_order_genes",
    "compute_trend_recurrence_score",
    "compute_composite_alignment_score",
    "temporal_kernel_gene_set_rankings",
    "temporal_kernel_pseudotime_enrichment",
    "evaluate_basal_signature_prominence",
    "compute_running_enrichment",
    "initialize_structured_loadings",
    "simulate_LF_MOGP",
]
