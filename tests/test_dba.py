import numpy as np
import pytest
from CellAlignDTW import dtw_barycenter_averaging_with_categories

def test_dtw_barycenter_averaging_with_categories():
    # Create synthetic test data
    n_subjects = 3
    seq_length = 5
    n_features = 2
    
    # Create numerical time series data (X)
    X = np.array([
        [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],  # First sequence
        [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],  # Second sequence
        [[2, 1], [3, 2], [4, 3], [5, 4], [6, 5]]   # Third sequence
    ])
    
    # Create categorical labels (Y)
    Y = np.array([
        ['A', 'A', 'B', 'B', 'C'],  # Categories for first sequence
        ['A', 'A', 'B', 'C', 'C'],  # Categories for second sequence
        ['A', 'B', 'B', 'C', 'C']   # Categories for third sequence
    ])
    
    # Run the function
    barycenter, categories, aligned_barycenters = dtw_barycenter_averaging_with_categories(
        X=X,
        Y=Y,
        barycenter_size=5,
        max_iter=30,
        tol=1e-5,
        verbose=False
    )
    
    # Test assertions
    assert barycenter.shape == (5, 2), "Barycenter shape should be (5, 2)"
    assert len(categories) == 5, "Categories length should be 5"
    assert len(aligned_barycenters) == 3, "Should have aligned barycenters for each input sequence"
    
    # Test that categories are valid
    assert all(cat in ['A', 'B', 'C'] for cat in categories), "All categories should be A, B, or C"
    
    # Test with numeric categories
    Y_numeric = np.array([
        [0, 0, 1, 1, 2],
        [0, 0, 1, 2, 2],
        [0, 1, 1, 2, 2]
    ])
    
    barycenter2, categories2, aligned_barycenters2 = dtw_barycenter_averaging_with_categories(
        X=X,
        Y=Y_numeric,
        barycenter_size=5,
        max_iter=30,
        tol=1e-5,
        verbose=False
    )
    print(categories2)
    # Test assertions for numeric categories
    assert barycenter2.shape == (5, 2), "Barycenter shape should be (5, 2)"
    assert len(categories2) == 5, "Categories length should be 5"
    assert all(isinstance(cat, (int, np.integer)) for cat in categories2), "All categories should be numeric"
    assert all(cat in [0, 1, 2] for cat in categories2), "All categories should be 0, 1, or 2"

def test_dtw_barycenter_averaging_with_categories_edge_cases():
    # Test with single sequence
    X_single = np.array([[[1, 2], [2, 3], [3, 4]]])
    Y_single = np.array([['A', 'B', 'C']])
    
    barycenter, categories, aligned_barycenters = dtw_barycenter_averaging_with_categories(
        X=X_single,
        Y=Y_single,
        barycenter_size=3
    )
    
    assert barycenter.shape == (3, 2), "Barycenter shape should be (3, 2)"
    assert len(categories) == 3, "Categories length should be 3"
    
    # Test with different tie_strategy options
    X_tie = np.array([
        [[1, 1], [2, 2]],
        [[1, 1], [2, 2]]
    ])
    Y_tie = np.array([
        ['A', 'B'],
        ['B', 'A']
    ])
    
    # Test 'random' tie strategy
    barycenter_random, categories_random, _ = dtw_barycenter_averaging_with_categories(
        X=X_tie,
        Y=Y_tie,
        tie_strategy='random',
        n_init=1
    )
    
    # Test 'weighted_random' tie strategy
    barycenter_weighted, categories_weighted, _ = dtw_barycenter_averaging_with_categories(
        X=X_tie,
        Y=Y_tie,
        tie_strategy='weighted_random',
        n_init=1
    )
    
    assert len(categories_random) == len(categories_weighted) == 2

def test_dtw_barycenter_averaging_with_categories_ragged():
    # Create ragged test data with different sequence lengths
    X_ragged = [
        [[1, 2], [2, 3], [3, 4], [4, 5]],          # 4 time points
        [[1, 1], [2, 2], [3, 3]],                   # 3 time points
        [[2, 1], [3, 2], [4, 3], [5, 4], [6, 5]]   # 5 time points
    ]
    
    Y_ragged = [
        ['A', 'A', 'B', 'C'],          # 4 categories
        ['A', 'B', 'C'],               # 3 categories
        ['A', 'B', 'B', 'C', 'C']      # 5 categories
    ]
    
    # Run the function
    barycenter, categories, aligned_barycenters = dtw_barycenter_averaging_with_categories(
        X=X_ragged,
        Y=Y_ragged,
        barycenter_size=4,  # Choose desired output length
        max_iter=30,
        tol=1e-5,
        verbose=False
    )
    
    # Test assertions
    assert barycenter.shape == (4, 2), "Barycenter shape should be (4, 2)"
    assert len(categories) == 4, "Categories length should be 4"
    assert len(aligned_barycenters) == 3, "Should have aligned barycenters for each input sequence"
    assert all(cat in ['A', 'B', 'C'] for cat in categories), "All categories should be A, B, or C"