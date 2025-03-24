import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import itertools
from sklearn.neighbors import KernelDensity
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
import PyComplexHeatmap as pch

def plot_sigmoid_fits(aligned_obj):
    subjects = aligned_obj.df[aligned_obj.subject_col].unique()
    fig, axes = plt.subplots(nrows=1, ncols=len(subjects), figsize=(15, 3), sharey=True)
    for i, subject in enumerate(subjects):
        ax = axes[i]
        subject_data = aligned_obj.df[aligned_obj.df[aligned_obj.subject_col] == subject]
        x_data = subject_data[aligned_obj.score_col].values
        y_data = subject_data['numeric_label'].values
        cutoff_point = aligned_obj.cutoff_points[subject]
        ax.scatter(x_data, y_data, alpha=0.01)
        for i, cutoff in enumerate(cutoff_point):
            ax.axvline(x=cutoff, color='green', linestyle='--', label=f'{cutoff:.2f}')
        ax.set_ylim(None)
        ax.set_title(f'Subject: {subject}')
        ax.set_xlabel('Score')
        if i == 0:
            ax.set_ylabel('Label')
        ax.legend()
    plt.show()

def plot_summary_curve(summary_df, aggregated_curve, scores, genes, fig_size=(2, 0.8), pt_alpha=0.2, line_alpha=0.5, dpi=300):
    """
    Plots summary curves for a list of genes.

    Parameters:
    summary_df (DataFrame): DataFrame containing subject data with columns ['gene', 'subject', 'aligned_score', 'expression', 'smoothed'].
    aggregated_curve (DataFrame): DataFrame containing aggregated curve data with columns ['aligned_score'] and gene columns.
    scores (DataFrame): DataFrame containing scores with columns ['gene', 'MI', 'Max'].
    genes (list of str): List of gene names to plot.
    """
    num_genes = len(genes)
    fig, axes = plt.subplots(num_genes, 1, figsize=(fig_size[0], fig_size[1] * num_genes), sharex=True, dpi=dpi)

    if num_genes == 1:
        axes = [axes]  # Ensure axes is iterable if there's only one subplot

    # Define a color cycle for subjects
    color_cycle = itertools.cycle(mcolors.TABLEAU_COLORS)

    scatter_handles = []
    scatter_labels = []

    for i, (ax, gene) in enumerate(zip(axes, genes)):
        gene_summary = summary_df.loc[summary_df['gene'] == gene]
        
        # Plot each subject's smoothed curve and collect scatter handles
        subject_colors = {}
        for subject in gene_summary['subject'].unique():
            color = next(color_cycle)
            subject_colors[subject] = color
            subject_data = gene_summary[gene_summary['subject'] == subject]
            ax.scatter(subject_data['aligned_score'], subject_data['expression'], alpha=pt_alpha, s=0.1, color=color)
            ax.plot(subject_data['aligned_score'], subject_data['smoothed'], alpha=line_alpha, color=color)
            if i == 0:  # Collect labels only from the first subplot
                scatter_handles.append(ax.scatter([], [], alpha=1, s=10, color=color))
                scatter_labels.append(subject)
        
        # Plot the aggregated curve
        agg_line, = ax.plot(aggregated_curve['aligned_score'], aggregated_curve[gene], color='black', linewidth=2)
        
        # Calculate ylim based on smoothed values
        y_min = min(gene_summary['smoothed'])
        y_max = max(gene_summary['smoothed'])
        
        # Add some padding (e.g., 5% of the range)
        y_range = y_max - y_min
        padding = 0.05 * y_range
        ylim = (y_min - padding, y_max + padding)
        
        ax.set_ylim(ylim)
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(gene)

    axes[-1].set_xlabel('Aligned score')

    # Create a single legend outside the plot area using scatter handles
    fig.legend(scatter_handles, scatter_labels, loc='center left', bbox_to_anchor=(0.8, 0.5), fontsize='small', frameon=False)

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for the legend
    plt.show()


def plot_kshape_clustering(sorted_gene_curve, categories, label_orders=None, alpha=0.05):
    if label_orders is None:
        label_orders = sorted(set(categories))
    plt.figure(figsize=(2, 5/3*len(label_orders)))
    for i, category in enumerate(label_orders):
        plt.subplot(len(label_orders), 1, i + 1)
        cluster_curves = sorted_gene_curve.values[[x == category for x in categories],:]
        for xx in cluster_curves:
            plt.plot(xx.ravel(), "k-", alpha=alpha)
        centroid = np.mean(cluster_curves, axis=0)
        plt.plot(centroid.ravel(), "r-", linewidth=1.5, label='Centroid')
        #plt.text(0.36, 0.1, category, transform=plt.gca().transAxes)
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.title("KShape Clustering")

    plt.tight_layout()
    plt.show()

def fit_kde(sorted_gene_curve, df, cell_types, bandwidth=50):
    gene_density = pd.DataFrame(index=sorted_gene_curve.index)
    for cell_type in cell_types:
        is_cell_type_gene = [str(cell_type).lower() in str(x).lower() if pd.notna(x) else False for x in df] # (np.isin(df, cell_type)).astype(int)
        x = np.arange(len(is_cell_type_gene)).reshape(-1, 1)
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde.fit(x, sample_weight=is_cell_type_gene)

        # Evaluate the density model on a grid
        x_d = np.linspace(0, len(is_cell_type_gene)-1, len(is_cell_type_gene)).reshape(-1, 1)
        log_density = kde.score_samples(x_d)
        gene_density[cell_type] = np.exp(log_density)
    return gene_density


def plot_kde_density(density, title="", clusters=None, cluster_colors=None, name_map=None):
    fig, ax = plt.subplots(2, 1, figsize=(5, 3), gridspec_kw={'height_ratios': [3, 0.1], 'hspace': 0})
    if name_map is not None:
        density.columns = density.columns.map(name_map)
    # Plot KDE density
    for cell_type, density_values in density.items():
        ax[0].plot(density.index, density_values, label=cell_type)
        ax[0].fill_between(density.index, density_values, alpha=0.5)
    
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', frameon=False)
    ax[0].set_title(title)
    ax[0].set_yticks([])
    ax[0].set_xticks([])
    ax[0].set_xlabel('')
    ax[0].set_ylabel('Density', fontsize=15)

    # Add bar plot for cell type annotations
    if clusters is not None:
        unique_clusters = cluster_colors.keys()
        color_map = cluster_colors
        for i, cluster in enumerate(unique_clusters):
            subset = density.index[clusters == cluster]
            ax[1].bar(subset, height=0.5, bottom=-0.5, width=1, 
                      color=cluster_colors[cluster], align='edge', alpha=1)
        
        # Add legend for cell types
        handles = [plt.Line2D([0], [0], color=color_map[ct], lw=4) for ct in unique_clusters]
        labels = unique_clusters
        ax[1].legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', frameon=False)
        
        ax[1].set_yticks([])
        ax[1].set_xticks([])
        for spine in ax[0].spines.values():
            spine.set_visible(False)
        for spine in ax[1].spines.values():
            spine.set_visible(False)
        ax[1].set_xlabel('Genes', fontsize=15)
    
    plt.tight_layout()
    plt.show()

def plot_kde_density_ridge(density, scores=None, clusters=None, cluster_colors=None, 
                           name_map=None, cell_type_order=None, figsize=(6, 4), fontsize=12, save_path=None):
    if name_map is not None:
        density.columns = density.columns.map(name_map)
    n_cell_types = len(density.columns)
    height_ratios = [2] * n_cell_types
    n_rows = n_cell_types
    if clusters is not None:
        n_rows = n_cell_types + 1
        height_ratios.append(0.5)

    fig, ax = plt.subplots(n_rows, 1, 
                           figsize=figsize, 
                           gridspec_kw={'hspace': 0, 'height_ratios': height_ratios})
    ordering = density.columns
    if cell_type_order is not None:
        ordering = cell_type_order

    # Create color map for scores
    if scores is not None:
        scores = scores.fillna(0)  # Fill NA values with 0
        vmin = 0
        vmax = scores.values.max()
        norm_scores = (scores - vmin) / (vmax - vmin)
        cmap = plt.cm.Blues

    for i, cell_type in enumerate(ordering):
        if cell_type in density.columns:
            x = density.index
            y = density[cell_type]
            ax[i].plot(x, y, label=cell_type, color='black')
            
            # Create gradient fill
            if scores is not None and cell_type in scores.columns:
                cell_score = norm_scores[cell_type]
                # Default fill with white for zero/NA values
                ax[i].fill_between(x, y, color='white', alpha=0.5)
                
                # Only color non-zero scores
                if any(cell_score > 0):  # or use a small threshold like 0.01
                    colors = cmap(np.full(len(x), cell_score))
                    for j in range(len(x)-1):
                        ax[i].fill_between(x[j:j+2], y[j:j+2], 
                                         color=colors[j],
                                         alpha=0.8)  # Increased alpha for better visibility
            else:
                ax[i].fill_between(x, y, color='white', alpha=0.5)
            
            ax[i].set_yticks([])
            ax[i].set_xticks([])
            ax[i].set_xlabel('')
            for spine in ax[i].spines.values():
                spine.set_visible(False)
            ax[i].set_ylabel(cell_type, fontsize=fontsize, rotation=0, labelpad=10)

    # Add bar plot for cell type annotations
    if clusters is not None:
        unique_clusters = cluster_colors.keys()
        color_map = cluster_colors
        for i, cluster in enumerate(unique_clusters):
            subset = density.index[clusters == cluster]
            ax[n_rows-1].bar(subset, height=0.1, bottom=-0.5, width=1, 
                      color=cluster_colors[cluster], align='edge', alpha=1)
        
        # Add legend for cell types
        handles = [plt.Line2D([0], [0], color=color_map[ct], lw=4) for ct in unique_clusters]
        labels = unique_clusters
        ax[n_rows-1].legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', frameon=False)
        ax[n_rows-1].set_yticks([])
        ax[n_rows-1].set_xticks([])
        for spine in ax[n_rows-1].spines.values():
            spine.set_visible(False)
        ax[n_rows-1].set_xlabel('Genes', fontsize=fontsize)
    
    # Add colorbar if scores are provided
    # if scores is not None:
    #     sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    #     cbar = plt.colorbar(sm, ax=ax, location='right', pad=0.1)
    #     cbar.set_label('Score', fontsize=fontsize)
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_kde_heatmap(cluster_colors, cell_types, cell_type_colors, sorted_gene_curve, df_left, density=None, figsize=(4, 8), left_annotation_columns=None, vmin=-3, vmax=3, save_path=None):
    """
    Function to plot a KDE heatmap using PyComplexHeatmap.

    Parameters:
    - cluster_colors: Dictionary of cluster names to colors.
    - cell_types: List or array-like of cell types.
    - cell_type_colors: Dictionary of cell type names to colors.
    - density: DataFrame containing density information for different cell types.
    - sorted_gene_curve: DataFrame or array-like containing the gene expression data to plot.
    - df_left: DataFrame or array-like for left annotations (e.g., groupings).
    - output_file: Path to save the output heatmap image (default is "heatmap.png").
    """

    # Define custom colormaps for Early, Middle, and Late groups
    cmap = {'early': LinearSegmentedColormap.from_list("custom_cmap", ["#FFFFFF", cluster_colors['Early']]),
            'middle': LinearSegmentedColormap.from_list("custom_cmap", ["#FFFFFF", cluster_colors['Middle']]),
            'late': LinearSegmentedColormap.from_list("custom_cmap", ["#FFFFFF", cluster_colors['Late']])}

    # Create column annotation (top)
    col_ha = pch.HeatmapAnnotation(
        label=pch.anno_label(cell_types, merge=True, extend=True, rotation=0,
                             colors=cell_type_colors, adjust_color=True, luminance=0.75, relpos=(0,0)),
        Group=pch.anno_simple(cell_types, colors=cell_type_colors),
        verbose=1, axis=1, plot_legend=False, label_kws=dict(visible=False)
    )

    # Create left annotation
    if density is not None:
        left_annotations = {}
        for name, col_name, cmap_name in left_annotation_columns:
            if col_name in density.columns:
                left_annotations[name] = pch.anno_simple(density.loc[:, col_name], cmap=cmap[cmap_name], height=4)
        left_ha = pch.HeatmapAnnotation(
            **left_annotations,
            verbose=1,
            axis=0,
            plot_legend=False,
            label_kws=dict(visible=True, rotation=90, horizontalalignment='left', verticalalignment='center')
        )
    else:
        left_ha = None

    # Create right annotation
    right_ha = pch.HeatmapAnnotation(
        Group=pch.anno_simple(df_left, colors=cluster_colors),
        verbose=1, axis=0, plot_legend=True,
        label_kws=dict(visible=False)
    )

    # Plot the heatmap using PyComplexHeatmap's ClusterMapPlotter
    plt.figure(figsize=figsize)
    
    heatmap_plotter = pch.ClusterMapPlotter(
        data=sorted_gene_curve,
        top_annotation=col_ha,
        left_annotation=left_ha,
        right_annotation=right_ha,
        legend_gap=7,
        legend_hpad=-10,
        row_cluster=False,
        col_cluster=False,
        row_split_gap=1,
        cmap='Spectral_r',  # Use a colormap of your choice (e.g., 'viridis', 'parula')
        vmin=vmin,
        vmax=vmax
    )

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    # Show the plot
    plt.show()

def plot_gene_clusters(sorted_gene_curve, row_colors, col_cols, cluster_ordering, 
                       display_genes=None, yticklabels=True, labelsize=8, tick_width=0.01, figsize=(4, 8), save_path=None): 
    """
    Plots a heatmap with optional gene labeling on the y-axis.
    Dynamically adjusts y positions of labels to prevent overlap by lowering them if they are too close.
    Shifts x position to the right every two consecutive overlaps.

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

    # If display_genes is provided, modify y-tick labels and adjust their positions
    if display_genes is not None:
        # Create a list of new labels: keep gene name if it's in display_genes, otherwise set it to an empty string
        new_labels = [label.get_text() if label.get_text() in display_genes else '' for label in current_labels]
        # Get current y-tick positions
        yticks = g.ax_heatmap.get_yticks()
        display_positions = [(i, yticks[i]) for i, label in enumerate(current_labels) if label.get_text() in display_genes]
        display_positions.sort(key=lambda x: x[1])
        
        # Minimum distance threshold between labels on the y-axis
        min_distance = 10
        
        # Track consecutive overlaps and shift amount for x position
        overlap_count = 0
        shift_amount_x = 0.19  # Amount to shift right after every two overlaps
        
        # Loop through each display gene and adjust its position if necessary
        for i in range(1, len(display_positions)):
            current_index, current_y = display_positions[i]
            prev_index, prev_y = display_positions[i - 1]
            closest_gene = min(display_positions[:i], key=lambda x: abs(x[1] - current_y))
            # If the current label is too close to the previous one in terms of y-position
            if (abs(current_y - prev_y) < min_distance) or (abs(closest_gene[1] - current_y) < min_distance):
                overlap_count += 1
                new_x_position = current_labels[current_index].get_position()[0] + shift_amount_x * (overlap_count % 6)
                print(f"Shifting X Position: {new_x_position}")
                current_labels[current_index].set_position((new_x_position, current_y))
            else:
                # check for the closest genes previous to the current gene
                overlap_count = 0
                
        for label in current_labels:
            label.set_ha('left')  # Align horizontally to the left
        # Apply new tick labels after adjusting positions
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
