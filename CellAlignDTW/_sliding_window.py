import numpy as np
import pandas as pd
from gseapy import enrichr
from tqdm import tqdm
import matplotlib.pyplot as plt
import PyComplexHeatmap as pch

def enrich_until_success(gene_list, gene_sets):
    while True:
        try:
            return enrichr(gene_list=gene_list, gene_sets=gene_sets, organism="Human")
        except Exception as e:
            pass

def perform_sliding_window_enrichr_analysis(ordered_genes, window_size=150, stride=50, gene_set_library="CellMarker_2024"):
    results = []
    for i in tqdm(range(0, len(ordered_genes) - window_size + 1, stride), desc="Processing windows"):
        window_genes = ordered_genes[i:i + window_size]
        enrichment = enrich_until_success(gene_list=window_genes, gene_sets=gene_set_library)
        result = enrichment.results
        result['start_index'] = i
        result['end_index'] = i + window_size
        results.append(result)
    return pd.concat(results)

def pivot_results(results_df, p_threshold, or_threshold, exclude_gene_sets):
    """Pivot the results DataFrame for Adjusted P-value and Odds Ratio."""
    # Pivot for Adjusted P-value
    if exclude_gene_sets is not None:
        results_df = results_df[~results_df.Term.str.contains('|'.join(exclude_gene_sets))]
    pivot_df_p = results_df.pivot(index="Term", columns="start_index", values="Adjusted P-value")
    pivot_df_p = pivot_df_p.fillna(1)
    pivot_df_p = pivot_df_p[pivot_df_p.min(axis=1) < p_threshold]

    # Pivot for Odds Ratio
    pivot_df_or = results_df.pivot(index="Term", columns="start_index", values="Odds Ratio")
    pivot_df_or = pivot_df_or.fillna(0)
    pivot_df_or = pivot_df_or[pivot_df_or.max(axis=1) > or_threshold]
    
    return pivot_df_p, pivot_df_or

def calculate_significant_gene_sets(pivot_df_or, pivot_df_p):
    """Calculate significant gene sets based on the pivoted DataFrames."""
    gene_set_sig = set(pivot_df_or.index) & set(pivot_df_p.index)
    pivot_df_or = pivot_df_or.loc[list(gene_set_sig), :]
    pivot_df_lor = np.log10(pivot_df_or)
    pivot_df_lor = pivot_df_lor.replace(-np.inf, 0)
    pivot_df_lor = pivot_df_lor.replace(np.inf, 0)
    return pivot_df_lor, len(gene_set_sig)

def calculate_max_position(pivot_df_lor):
    """Calculate the position of the maximum value for each row."""
    positions = np.array(pivot_df_lor.columns)

    def get_max_position(row):
        return positions[np.argmax(row)]  # Get the position where max value occurs

    pivot_df_lor['Max_Position'] = pivot_df_lor.apply(get_max_position, axis=1)
    return pivot_df_lor.sort_values(by='Max_Position').drop(columns=['Max_Position'])

def plot_heatmap(pivot_df_lor_sorted):
    """Plot the heatmap for the sorted DataFrame."""
    plt.figure(figsize=(5, len(pivot_df_lor_sorted) * 0.2))  # Adjust height based on number of rows

    # Create the heatmap using ClusterMapPlotter
    heatmap_plotter = pch.ClusterMapPlotter(
        data=pivot_df_lor_sorted,
        row_cluster=False,
        col_cluster=False,
        show_colnames=True,
        show_rownames=True,
        row_split_gap=1,
        cmap='Spectral_r',
        vmin=-2,
        vmax=2
    )

    # Get the heatmap's axis and adjust y-axis labels
    ax = heatmap_plotter.ax_heatmap
    ax.set_yticks(np.arange(len(pivot_df_lor_sorted)))  # Set ticks to match the number of rows
    ax.set_yticklabels(pivot_df_lor_sorted.index, fontsize=5)  # Use smaller font size for clarity

    # Customize x-axis and title
    ax.set_xlabel("Sliding Window Start Index")
    ax.set_ylabel("Gene Set")
    ax.set_title("Gene Set Enrichment Across Sliding Windows", pad=20)

    # Adjust layout to prevent label clipping
    plt.tight_layout()
    plt.show()

def analyze_and_plot_enrichment(results_df, p_threshold=1e-4, or_threshold=5, exclude_gene_sets=None):
    """Main function to analyze and plot enrichment."""
    pivot_df_p, pivot_df_or = pivot_results(results_df, p_threshold, or_threshold, exclude_gene_sets)
    pivot_df_lor, num_significant = calculate_significant_gene_sets(pivot_df_or, pivot_df_p)
    pivot_df_lor_sorted = calculate_max_position(pivot_df_lor)
    print(num_significant)  # Print the number of significant gene sets
    plot_heatmap(pivot_df_lor_sorted)
