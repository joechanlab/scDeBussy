import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing"""
    n_samples = 100
    
    data = {
        'sample': ['sample1', 'sample2'] * 50,
        'cell_id': [f'cell_{i}' for i in range(n_samples)],
        'score': list(range(50)) + [x * 2 for x in range(50)],
        'cell_type': (['typeA'] * 25 + ['typeB'] * 25) * 2
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def cluster_ordering():
    """Sample cluster ordering"""
    return ['typeA', 'typeB']
