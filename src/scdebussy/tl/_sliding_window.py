import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyComplexHeatmap as pch
from gseapy import enrichr
from scipy.spatial.distance import cdist
from tqdm import tqdm


def enrich_until_success(gene_list, gene_sets):
    while True:
        try:
            return enrichr(gene_list=gene_list, gene_sets=gene_sets, organism="Human")
        except Exception:  # noqa: BLE001
            pass


def perform_sliding_window_enrichr_analysis(
    ordered_genes, window_size=150, stride=50, gene_set_library="CellMarker_2024"
):
    results = []
    for i in tqdm(range(0, len(ordered_genes) - window_size + 1, stride), desc="Processing windows"):
        window_genes = ordered_genes[i : i + window_size]
        enrichment = enrich_until_success(gene_list=window_genes, gene_sets=gene_set_library)
        result = enrichment.results
        result["start_index"] = i
        result["end_index"] = i + window_size
        results.append(result)
    return pd.concat(results)


def pivot_results(results_df, p_threshold, or_threshold, exclude_gene_sets):
    """Pivot the results DataFrame for Adjusted P-value and Odds Ratio."""
    # Pivot for Adjusted P-value
    if exclude_gene_sets is not None:
        results_df = results_df[~results_df.Term.str.contains("|".join(exclude_gene_sets))]
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

    pivot_df_lor["Max_Position"] = pivot_df_lor.apply(get_max_position, axis=1)
    return pivot_df_lor.sort_values(by="Max_Position").drop(columns=["Max_Position"])


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
        cmap="Spectral_r",
        vmin=-2,
        vmax=2,
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


def rbf_kernel(x, y, gamma):
    return np.exp(-gamma * cdist(x[:, None], y[:, None], "sqeuclidean"))


def sliding_window_enrichment(gene_list, gene_set, window_size, gamma, step_size=1):  # Added step_size parameter
    # Binary encoding
    binary_vector = np.array([1 if gene in gene_set else 0 for gene in gene_list])
    n = len(gene_list)

    # Compute kernel matrix
    positions = np.arange(n)
    kernel_matrix = rbf_kernel(positions, positions, gamma)

    # Sliding window convolution
    enrichment_signal = np.zeros(n)
    half_window = window_size // 2
    for i in range(0, n, step_size):  # Updated loop to use step_size
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        enrichment_signal[i] = np.sum(kernel_matrix[i, start:end] * binary_vector[start:end])

    # Create DataFrame with start positions and enrichment signals
    result_df = pd.DataFrame(
        {
            "start_position": np.arange(0, n, step_size),
            "enrichment_signal": enrichment_signal[::step_size],  # Take every step_size-th value
        }
    )

    return result_df


def permutation_test_with_nes(
    enrichment_signal, gene_list, gene_set, window_size=250, step_size=5, gamma=0.01, num_permutations=1000
):
    """
    Perform a permutation test to calculate a normalized enrichment score (NES).

    Parameters
    ----------
        enrichment_signal (np.array): Observed enrichment signal.
        gene_list (list): Ordered list of genes.
        gene_set (set): Set of genes to test for enrichment.
        num_permutations (int): Number of permutations to perform.

    Returns
    -------
        nes (float): Normalized enrichment score.
        p_value (float): P-value for the observed enrichment.
    """
    # Calculate observed maximum enrichment score
    observed_max = np.max(enrichment_signal.enrichment_signal)
    # print(observed_max)

    # Generate null distribution by permuting the gene list
    permuted_max_values = []
    for _ in range(num_permutations):
        # Permute the gene list
        permuted_list = np.random.permutation(gene_list)

        # Simulate enrichment signal using convolution or other smoothing method
        permuted_signal = sliding_window_enrichment(permuted_list, gene_set, window_size, gamma, step_size)

        # Record maximum value from permuted signal
        permuted_max_values.append(np.max(permuted_signal.enrichment_signal))

    # Convert null distribution to numpy array
    permuted_max_values = np.array(permuted_max_values)
    # print(permuted_max_values)
    # Calculate mean and standard deviation of null distribution
    null_mean = np.mean(permuted_max_values)
    null_std = np.std(permuted_max_values)

    # Handle edge case where null_std is zero
    if null_std == 0:
        nes = 0  # Assign NES as 0 if there's no variability in permutations
    else:
        # Calculate normalized enrichment score (NES)
        nes = (observed_max - null_mean) / null_std

    # Calculate p-value
    p_value = np.mean(permuted_max_values >= observed_max)

    return nes, p_value


def benjamini_hochberg(p_values, alpha=0.05):
    """
    Perform Benjamini-Hochberg FDR correction on a list of p-values.

    Parameters
    ----------
        p_values (list or np.array): List of p-values to correct.
        alpha (float): Desired FDR level (default is 0.05).

    Returns
    -------
        np.array: Array of adjusted p-values (q-values).
    """
    p_values = np.array(p_values)
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]

    # Adjust p-values
    adjusted_p_values = np.zeros(n)
    for i in range(n):
        adjusted_p_values[i] = min(1, sorted_p_values[i] * n / (i + 1))

    # Ensure monotonicity of adjusted p-values
    for i in range(n - 2, -1, -1):
        adjusted_p_values[i] = min(adjusted_p_values[i], adjusted_p_values[i + 1])

    # Return adjusted p-values in original order
    return adjusted_p_values[np.argsort(sorted_indices)]


def compute_enrichment_signals(
    gene_set_keys, gene_list, gene_set_dict, window_size=250, gamma=0.01, step_size=5, use_fdr=False, alpha=0.05
):
    """
    Compute enrichment signals for multiple gene sets with optional FDR correction.

    Parameters
    ----------
        gene_set_keys (list): List of keys identifying the gene sets in the dictionary.
        gene_list (list): Ordered list of genes.
        gene_set_dict (dict): Dictionary containing gene sets.
        window_size (int): Size of the sliding window (default: 250).
        gamma (float): Kernel parameter for smoothing (default: 0.05).
        step_size (int): Step size for sliding window (default: 20).
        use_fdr (bool): Whether to apply FDR correction to p-values (default: False).
        alpha (float): Significance level for FDR correction (default: 0.05).

    Returns
    -------
        results (list of dicts): A list of results containing NES, raw p-value, and optionally FDR-adjusted q-value for each gene set.
    """
    results = []
    raw_p_values = []

    # Process each gene set key
    for gene_set_key in gene_set_keys:
        # Extract the gene set
        gene_set = gene_set_dict[gene_set_key]

        # Compute enrichment signal
        enrichment_signal = sliding_window_enrichment(gene_list, gene_set, window_size, gamma, step_size)

        # Perform statistical test
        nes, p_value = permutation_test_with_nes(enrichment_signal, gene_list, gene_set, window_size, step_size, gamma)

        # Print NES and raw p-value
        print(f"Gene Set: {gene_set_key}, NES: {nes}, P-value: {p_value}")

        # Store intermediate results
        results.append(
            {
                "gene_set_key": gene_set_key,
                "NES": nes,
                "p_value": p_value,
                "q_value": None,  # Placeholder for q-value
                "enrichment_signal": enrichment_signal,  # Store signal for plotting
            }
        )

        # Collect raw p-values for FDR correction
        raw_p_values.append(p_value)

    # Apply FDR correction if requested
    if use_fdr:
        q_values = benjamini_hochberg(raw_p_values, alpha=alpha)

        # Update results with q-values
        for i in range(len(results)):
            results[i]["q_value"] = q_values[i]
            print(f"Gene Set: {results[i]['gene_set_key']}, Q-value (FDR-adjusted): {q_values[i]}")

    return results


def plot_enrichment_signals(results, gene_list, gene_set_dict, subset_gene_set_keys=None):
    """
    Plot enrichment signals for a subset of gene sets from the computation results.

    Parameters
    ----------
        results (list of dicts): A list of results containing NES, p-values, q-values, and enrichment signals.
        subset_gene_set_keys (list or None): Subset of gene set keys to plot. If None, plot all available results.

    Returns
    -------
        None
    """
    # Filter results based on subset_gene_set_keys if provided
    if subset_gene_set_keys is not None:
        results_to_plot = [res for res in results if res["gene_set_key"] in subset_gene_set_keys]
    else:
        results_to_plot = results

    # Plot each result
    for result in results_to_plot:
        enrichment_signal = result["enrichment_signal"]

        binary_vector = np.array([1 if gene in gene_set_dict[result["gene_set_key"]] else 0 for gene in gene_list])

        plt.figure(figsize=(6, 4))

        plt.subplot(2, 1, 1)  # Create a subplot for the binary vector
        plt.plot(binary_vector)
        plt.title(f"Binary Vector ({result['gene_set_key']})")

        plt.subplot(2, 1, 2)  # Create a subplot for the enrichment signal
        plt.plot(enrichment_signal["start_position"], enrichment_signal["enrichment_signal"])
        plt.title("Enrichment Signal")

        # Add NES and p-value to the plot
        plt.text(0.3, 0.8, f"NES: {result['NES']:.4f}", ha="center", va="center", transform=plt.gca().transAxes)
        plt.text(0.3, 0.7, f"P-value: {result['p_value']:.4f}", ha="center", va="center", transform=plt.gca().transAxes)

        if result["q_value"] is not None:
            plt.text(
                0.3, 0.6, f"Q-value: {result['q_value']:.4f}", ha="center", va="center", transform=plt.gca().transAxes
            )

        plt.tight_layout()
        plt.show()
