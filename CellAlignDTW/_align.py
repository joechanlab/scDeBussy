import numpy as np
import pandas as pd
from tslearn.metrics import dtw_path
from tslearn.barycenters import dtw_barycenter_averaging
from scipy.optimize import curve_fit

class CellAlignDTW:
    def __init__(self, df, cluster_ordering, sample_col, score_col, cell_type_col):
        self.df = df
        self.cluster_ordering = cluster_ordering
        self.sample_col = sample_col
        self.score_col = score_col
        self.cell_type_col = cell_type_col
        self.cutoff_points = None

    def align(self):
        cutoff_points = self.compute_cutoff_points_sigmoid(self.df)
        aligned_segments, barycenter = self.align_with_continuous_barycenter(self.df, 
                                                                             cutoff_points)
        self.create_aligned_dataframe(aligned_segments)

    @staticmethod
    def constrained_sigmoid(x, x0, k):
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-k * (x_clipped - x0)))
    
    def compute_cutoff_points_sigmoid(self):
        cutoff_points = {}
        samples = self.df[self.sample_col].unique()
        num_cutpoints = len(self.cluster_ordering) - 1
        for i in range(num_cutpoints):
            for sample in samples:
                df_cluster = self.df[np.isin(self.df['numeric_label'], [i, i+1])]
                sample_data = df_cluster[df_cluster[self.sample_col] == sample]
                x_data = sample_data[self.score_col].values
                y_data = sample_data['numeric_label'].values - sample_data['numeric_label'].values.min()
                initial_guess = [i/num_cutpoints, 10]
                popt, _ = curve_fit(self.constrained_sigmoid, x_data, y_data, 
                                    p0=initial_guess,
                                    bounds=([i / num_cutpoints, -np.inf], [1, np.inf]))
                x0, _ = popt
                if i == 0:
                    cutoff_points[sample] = [x0]
                else:
                    cutoff_points[sample].append(x0)
        self.cutoff_points = cutoff_points
        return cutoff_points

    def align_with_continuous_barycenter(self):
        aligned_segments = {}
        all_probabilities = [self.df[self.df[self.sample_col] == sample][self.score_col].to_numpy().reshape(-1, 1) for sample in self.df[self.sample_col].unique()]
        continuous_barycenter = dtw_barycenter_averaging(all_probabilities).flatten()
        reference_cutoff_value = np.argmin(np.abs(continuous_barycenter))
        barycenter_before, barycenter_after = continuous_barycenter[:reference_cutoff_value], continuous_barycenter[reference_cutoff_value:]

        for sample in self.df[self.sample_col].unique():
            sample_data = self.df[self.df[self.sample_col] == sample]
            cutoff_values = self.cutoff_points[sample]
            before_cutoff, after_cutoff = sample_data[sample_data[self.score_col] < self.cutoff_points[sample]], sample_data[sample_data[self.score_col] >= self.cutoff_points[sample]]
            before_probabilities, after_probabilities = before_cutoff[self.score_col].to_numpy().reshape(-1, 1), after_cutoff[self.score_col].to_numpy().reshape(-1, 1)

            # Perform DTW alignment with respective segments of the barycenter
            path_before, _ = dtw_path(before_probabilities, barycenter_before)
            path_after, _ = dtw_path(after_probabilities, barycenter_after)
        
            # Extract original indices and aligned probabilities
            before_indices, after_indices = [before_cutoff.index[i] for i, _ in path_before], [after_cutoff.index[i] for i, _ in path_after] # input index
            aligned_before_values, aligned_after_values = [barycenter_before[j] for _, j in path_before], [barycenter_after[j] for _, j in path_after] # reference score

            aligned_segments[sample] = {
                "before": {"original_indices": before_indices, "aligned_score": aligned_before_values},
                "after": {"original_indices": after_indices,"aligned_score": aligned_after_values}
            }
        return aligned_segments, (barycenter_before, barycenter_after)

    def create_aligned_dataframe(self, aligned_segments):
        data = {
            "sample": [],
            "original_index": [],
            "aligned_score": [],
            "segment": []
        }
        
        for sample, segments in aligned_segments.items():
            for segment_name in ["before", "after"]:
                original_indices = segments[segment_name]["original_indices"]
                aligned_values = segments[segment_name]["aligned_score"]
                data["sample"].extend([sample] * len(original_indices))
                data["original_index"].extend(original_indices)
                data["aligned_score"].extend(aligned_values)
                data["segment"].extend([segment_name] * len(original_indices))
        aligned_df = pd.DataFrame(data).drop_duplicates(subset='original_index')
        aligned_df = aligned_df.merge(self.df.reset_index(names="original_index"), 
                                      on=['original_index', 'sample'])
        self.df = aligned_df
