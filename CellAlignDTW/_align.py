import numpy as np
import pandas as pd
from tslearn.metrics import dtw_path
from tslearn.barycenters import dtw_barycenter_averaging
from scipy.optimize import curve_fit

class CellAlignDTW:
    def __init__(self, df, cluster_ordering, sample_col, score_col, cell_id_col, cell_type_col):
        self.df = df
        self.cluster_ordering = cluster_ordering
        self.sample_col = sample_col
        self.score_col = score_col
        self.cell_id_col = cell_id_col
        self.cell_type_col = cell_type_col
        self.cutoff_points = None

    def align(self):
        self.compute_cutoff_points_sigmoid()
        aligned_segments = self.align_with_continuous_barycenter()
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
                y_data = sample_data['numeric_label'].values - i
                initial_guess = [i/num_cutpoints, 1]
                popt, _ = curve_fit(self.constrained_sigmoid, x_data, y_data, 
                                    p0=initial_guess,
                                    bounds=([i / num_cutpoints, -np.inf], [1, np.inf]))
                x0, _ = popt
                if i == 0:
                    cutoff_points[sample] = [x0]
                else:
                    cutoff_points[sample].append(x0)
        self.cutoff_points = cutoff_points

    @staticmethod
    def split_by_cutpoints(df, cutpoints, score_col):
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

    def align_with_continuous_barycenter(self):
        aligned_segments = {}
        all_probabilities = [self.df[self.df[self.sample_col] == sample][self.score_col].to_numpy().reshape(-1, 1) for sample in self.df[self.sample_col].unique()]
        continuous_barycenter = dtw_barycenter_averaging(all_probabilities,
                                                        metric_params={'global_constraint':"sakoe_chiba", 'sakoe_chiba_radius': 1}).flatten()
        num_cutpoints = pd.DataFrame(self.cutoff_points).shape[0]
        all_cutoffs = np.array([self.cutoff_points[sample] for sample in self.df[self.sample_col].unique()])
        reference_cutpoints = all_cutoffs.mean(axis = 0)
        barycenter_segments = self.split_by_cutpoints(pd.DataFrame({self.score_col: continuous_barycenter}), reference_cutpoints, self.score_col)

        for sample in self.df[self.sample_col].unique():
            sample_data = self.df[self.df[self.sample_col] == sample]
            sample_cutoffs = self.cutoff_points[sample]
            data_segments = self.split_by_cutpoints(sample_data, sample_cutoffs, self.score_col)

            # Align each segment with its corresponding barycenter segment
            aligned_sample_segments = []
            for data_segment, barycenter_segment in zip(data_segments, barycenter_segments):                
                path, _ = dtw_path(data_segment[self.score_col], barycenter_segment[self.score_col],
                                   global_constraint="sakoe_chiba", sakoe_chiba_radius = 1)

                original_indices = [data_segment[self.cell_id_col].iloc[i] for i, _ in path]
                aligned_values = [barycenter_segment[self.score_col].iloc[j] for _, j in path]

                aligned_sample_segments.append({
                    "cell_id": original_indices,
                    "aligned_score": aligned_values
                })

            aligned_segments[sample] = aligned_sample_segments
        
        return aligned_segments

    def create_aligned_dataframe(self, aligned_segments):
        aligned_df = pd.DataFrame()
        for sample, segments in aligned_segments.items():
            for segment in segments:
                segment = pd.DataFrame(segment).drop_duplicates(subset="cell_id")
                segment["sample"] = sample
                aligned_df = pd.concat([aligned_df, segment])
        aligned_df = aligned_df.merge(self.df, on=['sample','cell_id'])
        self.df = aligned_df
