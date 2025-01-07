import pytest
import pandas as pd
import numpy as np
from CellAlignDTW._utils import split_by_cutpoints

def test_split_by_cutpoints_single_column():
    # Create sample DataFrame
    df = pd.DataFrame({
        'score': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'id': range(10)
    })
    
    # Define cutpoints
    cutpoints = [[3.5, 7.5]]  # Will create 3 segments: <3.5, 3.5-7.5, >7.5
    
    # Split DataFrame
    segments = split_by_cutpoints(df, cutpoints, 'score')
    
    # Assertions
    assert len(segments) == 3  # Should have 3 segments
    assert len(segments[0]) == 3  # First segment should have values 1, 2, 3
    assert len(segments[1]) == 4  # Second segment should have values 4, 5, 6, 7
    assert len(segments[2]) == 3  # Third segment should have values 8, 9, 10
    
    # Check values in each segment
    assert all(segments[0]['score'] < 3.5)
    assert all((segments[1]['score'] >= 3.5) & (segments[1]['score'] < 7.5))
    assert all(segments[2]['score'] >= 7.5)

def test_split_by_cutpoints_multiple_columns():
    # Create sample DataFrame with multiple columns
    df = pd.DataFrame({
        'score1': [1, 2, 3, 4, 5],
        'score2': [2, 3, 4, 5, 6],
        'id': range(5)
    })
    
    # Define cutpoints for both columns
    cutpoints = [[3], [4]]  # One cutpoint for each column
    
    # Split DataFrame
    segments = split_by_cutpoints(df, cutpoints, ['score1', 'score2'])
    
    # Assertions
    assert len(segments) == 2  # Should have 2 segments
    assert len(segments[0]) == 2  # First segment: points below both cutpoints
    assert len(segments[1]) == 3  # Second segment: points above either cutpoint

def test_split_by_cutpoints_error_cases():
    df = pd.DataFrame({
        'score': [1, 2, 3],
        'id': range(3)
    })
    
    # Test mismatched number of score columns and cutpoint lists
    with pytest.raises(ValueError, match="Number of score columns must match number of cutpoint lists"):
        split_by_cutpoints(df, [[1], [2]], 'score')
    
    # Test inconsistent number of cutpoints
    with pytest.raises(ValueError, match="All dimensions must have the same number of cutpoints"):
        split_by_cutpoints(df, [[1, 2], [1]], ['score', 'id'])

def test_split_by_cutpoints_edge_cases():
    # Create DataFrame with edge cases
    df = pd.DataFrame({
        'score': [1, 1, 10, 10],  # Values at extremes
        'id': range(4)
    })
    
    # Test with cutpoint at extreme value
    segments = split_by_cutpoints(df, [[1]], 'score')
    assert len(segments) == 2
    assert len(segments[0]) == 0  # Values equal to cutpoint go to first segment
    assert len(segments[1]) == 4  # Values above cutpoint
    
    # Test with empty segments
    segments = split_by_cutpoints(df, [[5]], 'score')
    assert len(segments) == 2
    assert len(segments[0]) == 2  # Values below cutpoint
    assert len(segments[1]) == 2  # Values above cutpoint