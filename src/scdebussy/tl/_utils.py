import logging

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import brentq
from sklearn.mixture import GaussianMixture

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def split_by_cutpoints(df, cutpoints, score_cols):
    """
    Split a DataFrame into segments based on consensus cutpoints across specified columns.

    Args:
        df (pd.DataFrame): DataFrame to be split.
        cutpoints (list of lists): List of cutpoint lists for each dimension.
            Each inner list contains cutpoints for the corresponding column.
            All inner lists must have the same length.
        score_cols (str or list): Column name(s) in the DataFrame to split based on.
            If string, treated as single column. If list, must match length of cutpoints.

    Returns
    -------
        list: List of DataFrames, each representing a segment split by the consensus cutpoints.
    """
    # Convert single column to list for consistent handling
    if isinstance(score_cols, str):
        score_cols = [score_cols]

    if len(score_cols) != len(cutpoints):
        raise ValueError("Number of score columns must match number of cutpoint lists")

    # Verify all cutpoint lists have the same length
    n_cuts = len(cutpoints[0])
    if not all(len(cuts) == n_cuts for cuts in cutpoints):
        raise ValueError("All dimensions must have the same number of cutpoints")

    segments = [[] for _ in range(n_cuts + 1)]

    for _, row in df.iterrows():
        # Find the first cutpoint index where the point exceeds the threshold in any dimension
        for i in range(n_cuts):
            # Check if point is below all cutpoints at index i
            if all(row[col] < cuts[i] for col, cuts in zip(score_cols, cutpoints, strict=False)):
                segments[i].append(row)
                break
        else:
            # If no cutpoint was lower in all dimensions, point goes to last segment
            segments[-1].append(row)

    segments = [pd.DataFrame(segment) for segment in segments]
    return segments


def compute_gmm_cutpoints(X, n_components):
    """
    Compute cutoff points between clusters using Gaussian Mixture Models for all dimensions.

    Args:
        X: array-like of shape (n_samples, n_dimensions)
           Input data matrix where each column represents a dimension
        n_components: int
           Number of components/clusters to fit

    Returns
    -------
        list of lists: Each inner list contains sorted cutoff points between adjacent
                      Gaussian components for one dimension
    """
    # Fit GMM
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)

    n_dimensions = X.shape[1]
    all_cutoffs = [[] for _ in range(n_dimensions)]
    n_cutoffs = n_components - 1

    # First loop through cutoff points
    for i in range(n_cutoffs):
        # Then through dimensions
        for dim in range(n_dimensions):
            means = gmm.means_[:, dim]
            sorted_indices = np.argsort(means)

            idx1, idx2 = sorted_indices[i], sorted_indices[i + 1]
            mu1, sigma1 = means[idx1], np.sqrt(gmm.covariances_[idx1][dim, dim])
            mu2, sigma2 = means[idx2], np.sqrt(gmm.covariances_[idx2][dim, dim])

            def gaussian_diff(x, mu1, sigma1, mu2, sigma2, idx1, idx2, gmm):
                return (
                    stats.norm.pdf(x, mu1, sigma1) * gmm.weights_[idx1]
                    - stats.norm.pdf(x, mu2, sigma2) * gmm.weights_[idx2]
                )

            def posterior_diff(x, mu1, sigma1, mu2, sigma2, idx1, idx2, gmm):
                p1 = stats.norm.pdf(x, mu1, sigma1) * gmm.weights_[idx1]
                p2 = stats.norm.pdf(x, mu2, sigma2) * gmm.weights_[idx2]
                return p1 / (p1 + p2) - 0.5  # Find where probability is equal between two components

            try:
                cutoff = brentq(gaussian_diff, mu1, mu2, args=(mu1, sigma1, mu2, sigma2, idx1, idx2, gmm))
            except Exception:  # noqa: BLE001
                logger.error(
                    f"Unable to determine the pseudotime transition point from cell type {i} to cell type {i + 1}. To find out which sample has this issue, set verbose=True and please check the pseudotime distribution of that sample."
                )
                logger.info("Using the midpoint as a heuristic fallback.")
                cutoff = (mu1 + mu2) / 2
            all_cutoffs[dim].append(cutoff)

    return all_cutoffs
