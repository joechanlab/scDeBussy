from ._align import aligner
from ._barycenter_align import align_pseudotime
from ._cluster import enrichr, order_genes, process_gene_data
from ._dba import dtw_barycenter_averaging_with_categories, dtw_barycenter_averaging_with_segments
from ._gam import gam_smooth_expression
from ._sliding_window import (
    analyze_and_plot_enrichment,
    calculate_max_position,
    calculate_significant_gene_sets,
    compute_enrichment_signals,
    perform_sliding_window_enrichr_analysis,
    permutation_test_with_nes,
    pivot_results,
    plot_enrichment_signals,
    plot_heatmap,
    sliding_window_enrichment,
)
