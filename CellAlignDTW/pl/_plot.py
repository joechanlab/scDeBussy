import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

def plot_summary_curve(summary_df, aggregated_curve, scores, gene):
    plt.figure(figsize=(5, 4))
    summary_df = summary_df.loc[summary_df.gene == gene, :]
    scores = scores.loc[scores.gene == gene, :]
    
    # Plot each sample's smoothed curve
    for sample in summary_df['sample'].unique():
        sample_data = summary_df[summary_df['sample'] == sample]
        plt.scatter(sample_data['aligned_score'], sample_data['expression'], alpha=0.1, s=0.1, label=f'{sample}')
        plt.plot(sample_data['aligned_score'], sample_data['smoothed'], alpha=0.3, label=f'{sample}')
    
    # Plot the aggregated curve
    plt.plot(aggregated_curve['aligned_score'], aggregated_curve[gene], color='black', linewidth=2, label='Aggregated Curve')
    
    plt.xlabel('Aligned score')
    plt.ylabel(f'{gene} z-score')
    plt.title(f'{gene} (MI = {round(scores['MI'].iloc[0], 3)}, Max @ {round(scores['Max'].iloc[0], 2)} %)')
    #plt.legend()
    plt.show()

def plot_kshape_clustering(sorted_gene_curve, categories, label_orders):
    plt.figure(figsize=(3, 5/3*len(label_orders)))
    for i, category in enumerate(label_orders):
        plt.subplot(len(label_orders), 1, i + 1)
        cluster_curves = sorted_gene_curve.values[[x == category for x in categories],:]
        for xx in cluster_curves:
            plt.plot(xx.ravel(), "k-", alpha=.05)
        centroid = np.mean(cluster_curves, axis=0)
        plt.plot(centroid.ravel(), "r-", linewidth=1.5, label='Centroid')
        plt.text(0.36, 0.1, category,
                 transform=plt.gca().transAxes)
        if i == 0:
            plt.title("KShape Clustering")

    plt.tight_layout()
    plt.show()


def plot_gene_clusters(sorted_gene_curve, row_colors, col_cols, cluster_ordering): 
    g = sns.clustermap(
        sorted_gene_curve,
        col_cluster=False,
        row_cluster=False,
        cmap='Spectral_r',
        figsize=(4, 8),
        vmax=3,
        vmin=-3,
        col_colors = [col_cols],
        row_colors=[row_colors],
        cbar_pos=(0.95, 0.35, 0.01, 0.2) 
    )

    g.ax_heatmap.yaxis.set_tick_params(labelsize=8)
    g.ax_heatmap.set_xticks([])
    g.ax_heatmap.set_xlabel(cluster_ordering.replace("_", "->"))
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
