import numpy as np
import pandas as pd
import scanpy as sc
from tslearn.metrics import dtw_path
from tslearn.barycenters import dtw_barycenter_averaging
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class CellAlignDTW:
    def __init__(self, df, cluster_ordering):
        self.df = df
        self.cluster_ordering = cluster_ordering

    def align(self):
        adata = self.add_annotation()
        df = adata.obs
        cutoff_points = self.compute_cutoff_points_sigmoid(df)
        aligned_segments, barycenter = self.align_with_continuous_barycenter(df, cutoff_points, 'sample', 'term_states_fwd_memberships_SCLC-A-NSCLC')
        return aligned_segments, barycenter

    def add_annotation(self):
        
        adata = adata[np.isin(adata.obs.cell_type_final2, self.cluster_ordering)].copy()
        for col in term_states_fwd_memberships.columns:
            print(col)
            adata.obs['term_states_fwd_memberships_' + col.split("_")[0]] = term_states_fwd_memberships.loc[:, col]
        score = adata.obs['term_states_fwd_memberships_SCLC-A'] - adata.obs['term_states_fwd_memberships_NSCLC']
        adata.obs['term_states_fwd_memberships_SCLC-A-NSCLC'] = score
        sorted_indices = adata.obs.sort_values(by='term_states_fwd_memberships_SCLC-A-NSCLC').index
        adata = adata[sorted_indices, :]
        return adata

    @staticmethod
    def constrained_sigmoid(x, x0, k):
        return 1 / (1 + np.exp(-k * (x - x0)))

    def compute_cutoff_points_sigmoid(self, df):
        cutoff_points = {}
        label_mapping = {label: idx for idx, label in enumerate(df['label'].unique())}
        df['numeric_label'] = df['label'].map(label_mapping)
        samples = df['sample'].unique()
        for sample in samples:
            sample_data = df[df['sample'] == sample]
            for prob_col in self.probability_cols:
                x_data = sample_data[prob_col].values
                y_data = sample_data['numeric_label'].values
                initial_guess = [0, 1]
                popt, _ = curve_fit(self.constrained_sigmoid, x_data, y_data, p0=initial_guess)
                x0, _ = popt
                cutoff_point = x0
                cutoff_points[sample + "_" + prob_col] = cutoff_point
        return cutoff_points

    @staticmethod
    def plot_sigmoid_fits(df, cutoff_points):
        samples = df['sample'].unique()
        fig, axes = plt.subplots(nrows=1, ncols=len(samples), figsize=(15, 5), sharey=True)
        for i, sample in enumerate(samples):
            ax = axes[i]
            sample_data = df[df['sample'] == sample]
            for prob_col in sample_data.columns:
                if 'probability' in prob_col:
                    sample_data.loc[:, prob_col] = pd.to_numeric(sample_data[prob_col], errors='coerce')
            for prob_col in sample_data.columns:
                if 'probability' in prob_col:
                    x_data = sample_data[prob_col].values
                    y_data = sample_data['numeric_label'].values
                    cutoff_point = cutoff_points[sample + "_" + prob_col]
                    ax.scatter(x_data, y_data, label='Data', alpha=0.2)
                    ax.axvline(x=cutoff_point, color='green', linestyle='--', label=f'Cutoff at x={cutoff_point:.2f}')
                    ax.set_title(f'Sample: {sample}')
                    ax.set_xlabel('Probability')
                    if i == 0:
                        ax.set_ylabel('Label')
                    ax.legend()
        plt.show()

    def align_with_continuous_barycenter(self, df, cutoff_points, sample_col, probability_col):
        aligned_segments = {}
        all_probabilities = [df[df[sample_col] == sample][probability_col].to_numpy().reshape(-1, 1) for sample in df[sample_col].unique()]

        # Compute a single continuous barycenter for all samples
        continuous_barycenter = dtw_barycenter_averaging(all_probabilities).flatten()

        # Use the first value of the barycenter as the reference cutoff point
        reference_cutoff_value = np.argmin(np.abs(continuous_barycenter))
        barycenter_before, barycenter_after = continuous_barycenter[:reference_cutoff_value], continuous_barycenter[reference_cutoff_value:]

        for sample in df[sample_col].unique():
            sample_data = df[df[sample_col] == sample]
            before_cutoff, after_cutoff = sample_data[sample_data[probability_col] < cutoff_points[sample]], sample_data[sample_data[probability_col] >= cutoff_points[sample]]

            # Convert probabilities to numpy arrays for DTW
            before_probabilities, after_probabilities = before_cutoff[probability_col].to_numpy().reshape(-1, 1), after_cutoff[probability_col].to_numpy().reshape(-1, 1)

            # Perform DTW alignment with respective segments of the barycenter
            path_before, _ = dtw_path(before_probabilities, barycenter_before)
            path_after, _ = dtw_path(after_probabilities, barycenter_after)

            # Extract original indices and aligned probabilities
            before_indices, after_indices = [before_cutoff.index[i] for i, _ in path_before], [after_cutoff.index[i] for i, _ in path_after]
            aligned_before_values, aligned_after_values = [continuous_barycenter[j] for _, j in path_before], [continuous_barycenter[j] for _, j in path_after]

            aligned_segments[sample] = {
                "before": {"original_indices": before_indices, "aligned_values": aligned_before_values},
                "after": {"original_indices": after_indices,"aligned_values": aligned_after_values}
            }

        return aligned_segments, (barycenter_before,barycenter_after)
