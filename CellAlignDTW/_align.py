import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import scipy.stats as stats
from scipy.optimize import brentq
from tslearn.metrics import dtw_path
from scipy.optimize import curve_fit
from ._dba import dtw_barycenter_averaging_with_categories
from ._utils import split_by_cutpoints

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
        self.compute_cutoff_points_kmeans()
        aligned_segments = self.align_with_continuous_barycenter()
        self.create_aligned_dataframe(aligned_segments)

    def compute_cutoff_points_kmeans(self):
        cutoff_points = {}
        subjects = self.df[self.subject_col].unique()
        num_clusters = len(self.cluster_ordering)
        
        for subject in subjects:
            subject_data = self.df[self.df[self.subject_col] == subject]
            X = np.column_stack([
                subject_data[self.score_col].values,
                subject_data['numeric_label'].values
            ])
            
            # Fit GMM instead of KMeans
            gmm = GaussianMixture(n_components=num_clusters, random_state=42)
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
            
            cutoff_points[subject] = cutoffs
        
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
        
        kmeans = KMeans(n_clusters=len(self.cluster_ordering), random_state=42)
        kmeans.fit(X)
        
        # Sort centers by x-coordinate and calculate midpoints
        centers = sorted(kmeans.cluster_centers_, key=lambda x: x[0])
        reference_cutpoints = [(centers[i][0] + centers[i+1][0])/2 for i in range(len(centers)-1)]
        
        print("Reference cutpoints:", reference_cutpoints)

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
