import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import scipy.stats as stats
from scipy.optimize import brentq


def split_by_cutpoints(df, cutpoints, score_col):
    """
    Split a DataFrame into segments based on cutpoints in a specified column.

    Args:
        df (pd.DataFrame): DataFrame to be split.
        cutpoints (list): List of cutpoints to split the DataFrame.
        score_col (str): Column name in the DataFrame to split based on.

    Returns:
        list: List of DataFrames, each representing a segment split by the cutpoints.
    """
    segments = [[] for _ in range(len(cutpoints) + 1)]
    
    for _, row in df.iterrows():
        value = row[score_col]
        for i, cutoff in enumerate(cutpoints):
            if value < cutoff:
                segments[i].append(row)
                break
        else:
            segments[-1].append(row)
    
    segments = [pd.DataFrame(segment) for segment in segments]
    return segments


def compute_gmm_cutpoints(X, n_components):
    """
    Compute cutoff points between clusters using Gaussian Mixture Models.
    
    Args:
        X: array-like of shape (n_samples, 2)
           First column contains scores, second column contains numeric labels
        n_components: int
           Number of components/clusters to fit
           
    Returns:
        list: Sorted cutoff points between adjacent Gaussian components
    """
    # Fit GMM
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    
    # Sort components by their means
    means = gmm.means_[:, 0]  # Get means of the score dimension
    sorted_indices = np.argsort(means)
    
    # Calculate intersection points between adjacent Gaussians
    cutoffs = []
    for i in range(len(sorted_indices)-1):
        idx1, idx2 = sorted_indices[i], sorted_indices[i+1]
        mu1, sigma1 = means[idx1], np.sqrt(gmm.covariances_[idx1][0,0])
        mu2, sigma2 = means[idx2], np.sqrt(gmm.covariances_[idx2][0,0])
        
        # Find intersection point of the two Gaussians
        def gaussian_diff(x):
            return (stats.norm.pdf(x, mu1, sigma1) * gmm.weights_[idx1] - 
                   stats.norm.pdf(x, mu2, sigma2) * gmm.weights_[idx2])
        
        # Search for zero crossing between the means
        cutoff = brentq(gaussian_diff, mu1, mu2)
        cutoffs.append(cutoff)
    
    return cutoffs
