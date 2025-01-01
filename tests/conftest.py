import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing"""
    n_samples = 150
    
    data = {
        'sample': np.repeat(['sample1', 'sample2', 'sample3'], 50),
        'cell_id': [f'cell_{i}' for i in range(n_samples)],
        'score': np.sort(np.random.random((3, 50))).flatten() * 100,
        'cell_type': np.tile(np.repeat(['typeA', 'typeB'], 25), 3)
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def cluster_ordering():
    """Sample cluster ordering"""
    return ['typeA', 'typeB']
