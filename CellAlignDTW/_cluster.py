import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import KShape, KernelKMeans
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

def categorize_values(X, Y):
    result = []
    
    for x in X:
        # Initialize category index
        category = 0
        
        # Find the appropriate category for x
        for i, cutoff in enumerate(Y):
            if x < cutoff:
                break
            category = i + 1
        
        result.append(category)
    
    return pd.Series(result)


def process_gene_data(scores_df, gene_curve, cell_colors, cutoff_points,
                      MI_threshold=0.5, n_clusters=3,
                      label_orders=None, n_init=5):
    is_filtered_genes = scores_df.gene[(scores_df.MI > MI_threshold)]
    aligned_score = gene_curve.aligned_score
    gene_curve = gene_curve.loc[:, is_filtered_genes]
    zscore_scaler = StandardScaler()
    zscore_data = zscore_scaler.fit_transform(gene_curve)
    normalized_gene_curve = pd.DataFrame(
        zscore_data,
        columns=gene_curve.columns,
        index=gene_curve.index
    )
    ks = KShape(n_clusters=n_clusters, random_state=0, 
                verbose=True, n_init=n_init).fit(normalized_gene_curve.values.T)
    clusters = ks.labels_

    max_index = normalized_gene_curve.apply(lambda col: max_idx(col, 3), axis=0)
    combined_order = sorted(range(len(max_index)), key=lambda i: (max_index[i]))
    sorted_gene_curve = normalized_gene_curve.iloc[:, combined_order].T

    labels = clusters[combined_order]
    label_ordering = order_categories_by_average_position(labels)
    if label_orders is None: 
        label_orders = range(n_clusters)
    map_gene_labels = dict(zip(range(n_clusters), label_orders))
    category = [map_gene_labels[x] for x in labels]
    lut = dict(zip(label_orders, sns.hls_palette(len(label_orders), l=0.5, s=0.8)))
    row_colors = pd.DataFrame(category)[0].map(lut)

    lut = dict(zip(range(len(cell_colors)), cell_colors), l=0.5, s=0.8)
    col_cols = (categorize_values(aligned_score, cutoff_points)).map(lut)

    return sorted_gene_curve, row_colors, col_cols, category
