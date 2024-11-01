import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import itertools
from matplotlib.colors import LogNorm

def plot_sigmoid_fits(aligned_obj):
    samples = aligned_obj.df[aligned_obj.sample_col].unique()
    fig, axes = plt.subplots(nrows=1, ncols=len(samples), figsize=(15, 3), sharey=True)
    for i, sample in enumerate(samples):
        ax = axes[i]
        sample_data = aligned_obj.df[aligned_obj.df[aligned_obj.sample_col] == sample]
        x_data = sample_data[aligned_obj.score_col].values
        y_data = sample_data['numeric_label'].values
        cutoff_point = aligned_obj.cutoff_points[sample]
        ax.scatter(x_data, y_data, alpha=0.01)
        for i, cutoff in enumerate(cutoff_point):
            ax.axvline(x=cutoff, color='green', linestyle='--', label=f'{cutoff:.2f}')
        ax.set_ylim(None)
        ax.set_title(f'Sample: {sample}')
        ax.set_xlabel('Score')
        if i == 0:
            ax.set_ylabel('Label')
        ax.legend()
    plt.show()

def plot_summary_curve(summary_df, aggregated_curve, scores, genes, fig_size=(2, 0.8), pt_alpha=0.2, line_alpha=0.5, ylim=[-3, 3]):
    """
    Plots summary curves for a list of genes.

    Parameters:
    summary_df (DataFrame): DataFrame containing sample data with columns ['gene', 'sample', 'aligned_score', 'expression', 'smoothed'].
    aggregated_curve (DataFrame): DataFrame containing aggregated curve data with columns ['aligned_score'] and gene columns.
    scores (DataFrame): DataFrame containing scores with columns ['gene', 'MI', 'Max'].
    genes (list of str): List of gene names to plot.
    """
    num_genes = len(genes)
    fig, axes = plt.subplots(num_genes, 1, figsize=(fig_size[0], fig_size[1] * num_genes), sharex=True)

    if num_genes == 1:
        axes = [axes]  # Ensure axes is iterable if there's only one subplot

    # Define a color cycle for samples
    color_cycle = itertools.cycle(mcolors.TABLEAU_COLORS)

    scatter_handles = []
    scatter_labels = []

    for i, (ax, gene) in enumerate(zip(axes, genes)):
        gene_summary = summary_df.loc[summary_df['gene'] == gene]
        
        # Plot each sample's smoothed curve and collect scatter handles
        sample_colors = {}
        for sample in gene_summary['sample'].unique():
            color = next(color_cycle)
            sample_colors[sample] = color
            sample_data = gene_summary[gene_summary['sample'] == sample]
            ax.scatter(sample_data['aligned_score'], sample_data['expression'], alpha=pt_alpha, s=0.1, color=color)
            ax.plot(sample_data['aligned_score'], sample_data['smoothed'], alpha=line_alpha, color=color)
            if i == 0:  # Collect labels only from the first subplot
                scatter_handles.append(ax.scatter([], [], alpha=1, s=10, color=color))
                scatter_labels.append(sample)
        
        # Plot the aggregated curve
        agg_line, = ax.plot(aggregated_curve['aligned_score'], aggregated_curve[gene], color='black', linewidth=2)

        ax.set_ylim(ylim)
        ax.set_ylabel('z-score')
        ax.set_title(gene)

    axes[-1].set_xlabel('Aligned score')

    # Create a single legend outside the plot area using scatter handles
    fig.legend(scatter_handles, scatter_labels, loc='center left', bbox_to_anchor=(0.8, 0.5), fontsize='small', frameon=False)

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for the legend
    plt.show()


def plot_kshape_clustering(sorted_gene_curve, categories, label_orders):
    plt.figure(figsize=(2, 5/3*len(label_orders)))
    for i, category in enumerate(label_orders):
        plt.subplot(len(label_orders), 1, i + 1)
        cluster_curves = sorted_gene_curve.values[[x == category for x in categories],:]
        for xx in cluster_curves:
            plt.plot(xx.ravel(), "k-", alpha=.05)
        centroid = np.mean(cluster_curves, axis=0)
        plt.plot(centroid.ravel(), "r-", linewidth=1.5, label='Centroid')
        #plt.text(0.36, 0.1, category, transform=plt.gca().transAxes)
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.title("KShape Clustering")

    plt.tight_layout()
    plt.show()

def shuffle_genes(sorted_gene_curve, row_colors, display_genes, min_distance):
    """
    Rearranges the rows of sorted_gene_curve so that any two genes in display_genes are at least min_distance apart,
    while ensuring that genes are only rearranged within their respective row_colors groups.

    Parameters:
    sorted_gene_curve (DataFrame): DataFrame with gene expression data.
    row_colors (list): List of colors for each row in sorted_gene_curve.
    display_genes (list of str): List of genes to display on the y-axis.
    min_distance (int): Minimum distance between any two display genes.

    Returns:
    DataFrame: Rearranged DataFrame with shuffled gene rows.
    list: Rearranged row colors corresponding to the shuffled gene rows.
    """
    
    # Get a list of all genes
    all_genes = sorted_gene_curve.index.tolist()

    # Create a dictionary to group genes by their row colors
    color_to_genes = {}
    color_to_row_colors = {}

    for gene, color in zip(all_genes, row_colors):
        if color not in color_to_genes:
            color_to_genes[color] = []
            color_to_row_colors[color] = []
        color_to_genes[color].append(gene)
        color_to_row_colors[color].append(color)

    # Rearrange each color group separately
    new_order = []
    new_row_colors = []

    for color, genes in color_to_genes.items():
        # Separate display and non-display genes within this color group
        display_in_group = [gene for gene in genes if gene in display_genes]
        non_display_in_group = [gene for gene in genes if gene not in display_genes]

        # Start with non-display genes
        new_group_order = non_display_in_group.copy()

        # Insert display genes ensuring they are at least min_distance apart within this group
        for i, gene in enumerate(display_in_group):
            insert_pos = i * (min_distance + 1)
            if insert_pos < len(new_group_order):
                new_group_order.insert(insert_pos, gene)
            else:
                new_group_order.append(gene)

        # Add the rearranged group to the final order
        new_order.extend(new_group_order)
        new_row_colors.extend([color] * len(new_group_order))

    # Create a new DataFrame with shuffled rows and corresponding row colors
    shuffled_df = sorted_gene_curve.loc[new_order]

    return shuffled_df, new_row_colors


def plot_gene_clusters(sorted_gene_curve, row_colors, col_cols, cluster_ordering, 
                       display_genes=None, min_distance=20, yticklabels=True, labelsize=8, tick_width=0.01, figsize=(4, 8), save_path=None): 
    """
    Plots a heatmap with optional gene labeling on the y-axis.

    Parameters:
    sorted_gene_curve (DataFrame): DataFrame with gene expression data.
    row_colors (list): List of colors for rows.
    col_cols (list): List of colors for columns.
    cluster_ordering (str): String to label x-axis.
    display_genes (list of str): List of genes to display on the y-axis. If None, all genes are labeled.
    yticklabels (bool or list): Whether to display y-tick labels or a list of labels.
    labelsize (int): Font size for y-tick labels.
    figsize (tuple): Size of the figure.
    save_path (str): Path to save the figure. If None, the figure is not saved.
    """
    if display_genes is not None:
        sorted_gene_curve, row_colors = shuffle_genes(sorted_gene_curve, row_colors, display_genes, min_distance)

    # Create clustermap
    g = sns.clustermap(
        sorted_gene_curve,
        col_cluster=False,
        row_cluster=False,
        cmap='Spectral_r',
        figsize=figsize,
        vmax=3,
        vmin=-3,
        col_colors=[col_cols],
        row_colors=[row_colors],
        cbar_pos=(0.95, 0.35, 0.01, 0.2),
        yticklabels=yticklabels
    )

    # Get current y-tick labels (which are gene names)
    current_labels = g.ax_heatmap.get_yticklabels()

    # If display_genes is provided, modify y-tick labels
    if display_genes is not None:
        # Create a list of new labels: keep gene name if it's in display_genes, otherwise set it to an empty string
        new_labels = [label.get_text() if label.get_text() in display_genes else '' for label in current_labels]
        g.ax_heatmap.set_yticklabels(new_labels)

    # Set y-axis tick label size and make ticks thinner
    g.ax_heatmap.yaxis.set_tick_params(labelsize=labelsize, width=tick_width)

    # Remove x-ticks and set x-label based on cluster ordering
    g.ax_heatmap.set_xticks([])
    g.ax_heatmap.set_xlabel(cluster_ordering.replace("_", "->"))

    plt.tight_layout()  # Automatically adjust subplots to fit into figure area
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)

    plt.show()

def plot_heatmap(results, terms):
    heatmap_long = results.loc[np.isin(results.Term, terms), ['Term', 'category', 'Odds Ratio', 'Adjusted P-value']]
    heatmap_long['Term'] = pd.Categorical(heatmap_long['Term'], categories=terms[::-1], ordered=True)
    heatmap_long['category'] = pd.Categorical(heatmap_long['category'], categories=['early', 'intermediate', 'late'], ordered=True)

    plt.figure(figsize=(8, 4))
    sns.scatterplot(data=heatmap_long, x='category', y='Term', size='Odds Ratio', sizes=(20, 200), 
                    hue='Adjusted P-value', palette='viridis_r', hue_norm=LogNorm())
    plt.xticks(rotation=45)
    plt.xlabel('')
    plt.ylabel('')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(-0.5, len(heatmap_long['category'].unique()) - 0.5)
    plt.ylim(-0.5, len(heatmap_long['Term'].unique()) - 0.5)
    plt.tight_layout()
    plt.savefig('dotplot.png', bbox_inches='tight')
    plt.show()
