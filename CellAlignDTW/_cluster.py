import pandas as pd
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import KShape, KernelKMeans
from collections import defaultdict
import gseapy as gp
import time

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
                      MI_threshold=0, GCV_threshold=1, AIC_threshold=np.Inf, n_clusters=3,
                      label_names=None, n_init=5, hierarchical=False):
    is_filtered_genes = scores_df.gene[(scores_df.MI > MI_threshold) & (scores_df.GCV < GCV_threshold) & (scores_df.AIC < AIC_threshold)]
    aligned_score = gene_curve.aligned_score
    gene_curve = gene_curve.loc[:, is_filtered_genes]
    zscore_scaler = StandardScaler()
    zscore_data = zscore_scaler.fit_transform(gene_curve)
    normalized_gene_curve = pd.DataFrame(
        zscore_data,
        columns=gene_curve.columns,
        index=gene_curve.index
    )
    if hierarchical:
        hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        clusters = hc.fit_predict(normalized_gene_curve.values.T)
    else:
        ks = KShape(n_clusters=n_clusters, random_state=0, 
                    verbose=True, n_init=n_init).fit(normalized_gene_curve.values.T)
        clusters = ks.labels_

    max_index = normalized_gene_curve.apply(lambda col: max_idx(col, 400), axis=0)
    combined_order = sorted(range(len(max_index)), key=lambda i: (max_index[i]))
    sorted_gene_curve = normalized_gene_curve.iloc[:, combined_order].T
    labels = clusters[combined_order]
    label_orders = order_categories_by_average_position(labels)
    new_label_mapping = {old_label: new_label for new_label, old_label in enumerate(label_orders)}
    labels = pd.Series(labels).map(new_label_mapping)

    if label_names is None: 
        category = labels
    else:
        map_gene_labels = dict(zip(range(n_clusters), label_names))
        category = [map_gene_labels[x] for x in labels]
    row_colors = pd.DataFrame(category)[0]
    col_cols = (categorize_values(aligned_score, cutoff_points))

    return sorted_gene_curve, row_colors, col_cols, category

def enrichr(gene_list, gene_sets):
    while True:
        try:
            enr = gp.enrichr(
                gene_list=gene_list,
                gene_sets=gene_sets,
                organism='human',
                outdir=None,
                cutoff=1,
                
            )
            early_results = enr.results
            return early_results  # Return the results if successful
        except Exception as e:
            print(f"An error occurred: {e}. Retrying...")
            time.sleep(5)
