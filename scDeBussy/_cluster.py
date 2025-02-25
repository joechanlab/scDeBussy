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


def bin_and_average_time_series(time_series, window_size, stride):
    """
    Bins the time series using the specified window size and stride, and computes the average value for each bin.
    
    Parameters:
        time_series (np.ndarray): The input time series data.
        window_size (int): The size of the window for binning.
        stride (int): The number of steps to move the window after each bin.
    
    Returns:
        np.ndarray: A 1D array where each element is the average value of a bin.
    """
    binned_averages = []
    length = len(time_series)
    
    for start in range(0, length - window_size + 1, stride):
        bin = time_series[start:start + window_size]
        binned_averages.append(np.mean(bin))  # Compute the average of the bin
    
    return np.array(binned_averages)

def argmax_with_largest_index_tie_breaking(array):
    """
    Returns the largest index of the maximum value in the array.

    Parameters:
        array (np.ndarray): The input array.

    Returns:
        int: The largest index of the maximum value.
    """
    # Find the maximum value in the array
    max_value = np.max(array)
    
    # Get all indices where the maximum value occurs
    max_indices = np.flatnonzero(array == max_value)
    
    # Select the largest index among these
    largest_index = max_indices[-1]
    
    return largest_index

def mean_around_max(time_series, num_elements):
    """
    Computes the mean of a specified number of elements surrounding the maximum value in a time series.

    Parameters:
        time_series (np.ndarray): The input time series data.
        num_elements (int): The total number of elements to consider around the maximum value.

    Returns:
        tuple: A tuple containing the index of the maximum value and the mean of the specified elements surrounding the maximum value.
    """
    # Find the index of the maximum value
    argmax_position = argmax_with_largest_index_tie_breaking(time_series)
    
    # Calculate half window size
    half_window = num_elements // 2

    # Determine start and end indices ensuring exactly num_elements are selected
    if argmax_position < half_window:
        start_index = 0
        end_index = num_elements
    elif argmax_position > len(time_series) - half_window:
        start_index = len(time_series) - num_elements
        end_index = len(time_series)
    else:
        start_index = argmax_position - half_window
        end_index = argmax_position + half_window

    # Extract surrounding elements
    surrounding_elements = time_series[start_index:end_index]

    # Compute mean
    mean = np.mean(surrounding_elements)

    return argmax_position, mean


def find_max(time_series, max_position, weight):
    """
    Computes a composite score for a time series based on the location of the maximum value 
    and the mean of surrounding values, weighted by a user-specified parameter.

    Parameters:
        time_series (np.ndarray): The input time series data.
        max_position (int): The length of the time series or the maximum possible position.
        weight (float): Weight assigned to the mean of surrounding values 
                                 when computing the composite score.

    Returns:
        float: A composite score that integrates the position of the maximum value and 
               the mean of surrounding values, weighted by the user-specified parameter.
    """
    # Compute the position of the maximum value and the mean of surrounding elements
    argmax_position, mean = mean_around_max(time_series, 100)
    composite_score = argmax_position * (1 - weight) + mean * weight * np.sign(0.5 - argmax_position/max_position)
    return composite_score

def order_genes(df, weight, window_size=10, stride=5):
    """
    Orders genes in a DataFrame based on the maximum location and mean of their expression profiles.
    
    Parameters:
        df (pd.DataFrame): A DataFrame where rows represent genes and columns represent gene expression.
        window_size (int): The size of the window for binning.
        stride (int): The number of steps to move the window after each bin.
        weight (float): The weight to assign to the mean in the composite score.
    
    Returns:
        list: A list of integer row indices (row positions) sorted by the composite score.
    """
    max_indices = []
    
    for row_idx, expression in df.iterrows():
        binned_averages = bin_and_average_time_series(expression.values, window_size, stride)
        max_position = len(binned_averages)
        composite_score = find_max(binned_averages, max_position, weight=weight)
        max_indices.append((row_idx, composite_score))
    
    # Sort by the composite score and extract row positions
    sorted_indices = sorted(max_indices, key=lambda x: x[1], reverse=False)
    sorted_row_numbers = [df.index.get_loc(row_idx) for row_idx, _ in sorted_indices]
    
    return sorted_row_numbers


def process_gene_data(scores_df, gene_curve, cell_colors, cutoff_points,
                      MI_threshold=0, GCV_threshold=1, AIC_threshold=np.Inf, n_clusters=3,
                      label_names=None, n_init=5, hierarchical=False, weight=0.5):
    # Filter genes based on thresholds
    is_filtered_genes = scores_df.gene[(scores_df.MI > MI_threshold) & (scores_df.GCV < GCV_threshold) & (scores_df.AIC < AIC_threshold)]
    aligned_score = gene_curve.aligned_score
    gene_curve = gene_curve.loc[:, is_filtered_genes]
    
    # Normalize gene curve data
    zscore_scaler = StandardScaler()
    normalized_gene_curve = pd.DataFrame(zscore_scaler.fit_transform(gene_curve), columns=gene_curve.columns, index=gene_curve.index)
    
    # Perform clustering
    if hierarchical:
        clusters = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(normalized_gene_curve.values.T)
    else:
        clusters = KShape(n_clusters=n_clusters, random_state=0, verbose=True, n_init=n_init).fit(normalized_gene_curve.values.T).labels_
    
    # Reorder data based on weighted pseudotime
    combined_order = order_genes(normalized_gene_curve.T, weight=weight)
    sorted_gene_curve = normalized_gene_curve.T.iloc[combined_order,:]
    labels = clusters[combined_order]
    
    # Reorder labels based on their average position
    label_orders = order_categories_by_average_position(labels)
    labels = pd.Series(labels).map({old_label: new_label for new_label, old_label in enumerate(label_orders)})
    
    # Map labels to names if provided
    if label_names is None:
        category = labels
    else:
        category = [label_names[x] for x in labels]
    
    # Prepare row colors and column colors
    row_colors = pd.DataFrame(category)[0]
    col_cols = categorize_values(aligned_score, cutoff_points)
    
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
