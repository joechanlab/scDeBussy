import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import scipy.stats as stats
from scipy.optimize import brentq
from tslearn.metrics import dtw_path
from scipy.optimize import curve_fit
from ._dba import dtw_barycenter_averaging_with_categories
from ._utils import split_by_cutpoints, compute_gmm_cutpoints

class CellAlignDTW:
    def __init__(self, df, cluster_ordering, subject_col, score_col, cell_id_col, cell_type_col, verbose=False):
        self.df = df
        self.cluster_ordering = cluster_ordering
        self.subject_col = subject_col
        self.score_col = score_col
        self.cell_id_col = cell_id_col
        self.cell_type_col = cell_type_col
        self.cutoff_points = None
        self.label_mapping = {label: idx for idx, label in enumerate(cluster_ordering)}
        self.df['numeric_label'] = self.df[cell_type_col].map(self.label_mapping)
        self.verbose = verbose

    def align(self):
        self.compute_cutoff_points_gmm()
        aligned_segments = self.align_with_continuous_barycenter()
        self.create_aligned_dataframe(aligned_segments)

    def compute_cutoff_points_gmm(self):
        cutoff_points = {}
        subjects = self.df[self.subject_col].unique()
        num_clusters = len(self.cluster_ordering)
        
        for subject in subjects:
            subject_data = self.df[self.df[self.subject_col] == subject]
            X = np.column_stack([
                subject_data[self.score_col].values,
                subject_data['numeric_label'].values
            ])
            cutoff_points[subject] = compute_gmm_cutpoints(X, num_clusters)
        
        self.cutoff_points = cutoff_points
        print(cutoff_points)

    def align_with_continuous_barycenter(self):
        aligned_segments = {}
        all_probabilities = [self.df[self.df[self.subject_col] == subject][self.score_col].to_numpy().reshape(-1, 1) for subject in self.df[self.subject_col].unique()]
        all_cell_types = [self.df[self.df[self.subject_col] == subject][self.cell_type_col].tolist() for subject in self.df[self.subject_col].unique()]
        continuous_barycenter, barycenter_categories, _  = dtw_barycenter_averaging_with_categories(all_probabilities, all_cell_types,
                                                        metric_params={'global_constraint':"sakoe_chiba", 'sakoe_chiba_radius': 1}, verbose=self.verbose)
        continuous_barycenter = continuous_barycenter.flatten()
        barycenter_categories = [self.label_mapping[x] for x in barycenter_categories]
        
        X = np.column_stack([
            continuous_barycenter,
            barycenter_categories
        ])
        
        reference_cutpoints = compute_gmm_cutpoints(X, len(self.cluster_ordering))

        barycenter_segments = split_by_cutpoints(pd.DataFrame({self.score_col: continuous_barycenter}), reference_cutpoints, self.score_col)

        for subject in self.df[self.subject_col].unique():
            subject_data = self.df[self.df[self.subject_col] == subject]
            subject_cutoffs = self.cutoff_points[subject]
            data_segments = split_by_cutpoints(subject_data, subject_cutoffs, self.score_col)

            # Align each segment with its corresponding barycenter segment
            aligned_subject_segments = []
            for data_segment, barycenter_segment in zip(data_segments, barycenter_segments):                
                path, _ = dtw_path(data_segment[self.score_col], barycenter_segment[self.score_col],
                                   global_constraint="sakoe_chiba", sakoe_chiba_radius = 1)

                original_indices = [data_segment[self.cell_id_col].iloc[i] for i, _ in path]
                aligned_values = [barycenter_segment[self.score_col].iloc[j] for _, j in path]

                aligned_subject_segments.append({
                    "cell_id": original_indices,
                    "aligned_score": aligned_values
                })

            aligned_segments[subject] = aligned_subject_segments
        
        return aligned_segments

    def create_aligned_dataframe(self, aligned_segments):
        aligned_df = pd.DataFrame()
        for subject, segments in aligned_segments.items():
            for segment in segments:
                segment = pd.DataFrame(segment).drop_duplicates(subset="cell_id")
                segment["subject"] = subject
                aligned_df = pd.concat([aligned_df, segment])
        aligned_df = aligned_df.merge(self.df, on=['subject','cell_id'])
        self.df = aligned_df
