import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import KShape
from collections import defaultdict

def order_categories_by_average_position(categories):
    # Dictionary to store total positions and counts
    position_data = defaultdict(lambda: {'sum': 0, 'count': 0})
    
    # Iterate through the list and update position data
    for index, category in enumerate(categories):
        position_data[category]['sum'] += index
        position_data[category]['count'] += 1
    
    # Calculate average positions
    averages = {category: data['sum'] / data['count'] for category, data in position_data.items()}
    
    # Sort categories by average position
    sorted_categories = sorted(averages, key=averages.get)
    
    return sorted_categories

def max_idx(x, window_size):
    # Calculate rolling maximum and find the index of the maximum value
    rolling_mean = x.rolling(window=window_size).mean()
    max_index = rolling_mean.idxmax()
    return max_index

def process_gene_data(scores_df, gene_curve, cell_colors, MI_threshold=0.5, MASE_threshold=500, n_clusters=3):
    is_filtered_genes = scores_df.gene[(scores_df.MI > MI_threshold) & (scores_df.MASE < MASE_threshold)]
    aligned_score = gene_curve.aligned_score
    gene_curve = gene_curve.loc[:, is_filtered_genes]
    zscore_scaler = StandardScaler()
    zscore_data = zscore_scaler.fit_transform(gene_curve)
    normalized_gene_curve = pd.DataFrame(
        zscore_data,
        columns=gene_curve.columns,
        index=gene_curve.index
    )

    # Calculate the index of the first maximum for each row
    ks = KShape(n_clusters=n_clusters, random_state=0, 
                verbose=True, n_init=5).fit(normalized_gene_curve.values.T)
    clusters = ks.labels_

    max_index = normalized_gene_curve.apply(lambda col: max_idx(col, 3), axis=0)
    combined_order = sorted(range(len(max_index)), key=lambda i: (max_index[i]))
    sorted_gene_curve = normalized_gene_curve.iloc[:, combined_order].T

    labels = clusters[combined_order]
    label_ordering = order_categories_by_average_position(labels)
    map_gene_labels = dict(zip(label_ordering, ["early", "intermediate", "late"]))
    category = [map_gene_labels[x] for x in labels]
    lut = dict(zip(['early', 'intermediate', 'late'], sns.hls_palette(3, l=0.5, s=0.8)))
    row_colors = pd.DataFrame(category)[0].map(lut)

    lut = dict(zip([True, False], cell_colors), l=0.5, s=0.8)
    
    col_cols = (aligned_score < 0).map(lut)

    return sorted_gene_curve, row_colors, col_cols, category
