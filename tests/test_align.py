import numpy as np
from scDeBussy import aligner

def test_scDeBussy_initialization(sample_df, cluster_ordering):
    test_aligner = aligner(
        df=sample_df,
        cluster_ordering=cluster_ordering,
        subject_col='subject',
        score_col='score',
        cell_id_col='cell_id',
        cell_type_col='cell_type'
    )
    
    assert test_aligner.df.equals(sample_df)
    assert test_aligner.cluster_ordering == cluster_ordering
    assert test_aligner.cutoff_points is None

def test_scDeBussy_align(sample_df, cluster_ordering):
    test_aligner = aligner(
        df=sample_df,
        cluster_ordering=cluster_ordering,
        subject_col='subject',
        score_col='score',
        cell_id_col='cell_id',
        cell_type_col='cell_type'
    )
    
    test_aligner.align()
    assert test_aligner.cutoff_points is not None
    assert np.isin('aligned_score', test_aligner.df.columns).any()

