import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    """Create a example DataFrame for testing"""
    n_subjects = 150

    data = {
        "subject": np.repeat(["subject1", "subject2", "subject3"], 50),
        "cell_id": [f"cell_{i}" for i in range(n_subjects)],
        "score": np.sort(np.random.random((3, 50))).flatten() * 100,
        "cell_type": np.tile(np.repeat(["typeA", "typeB"], 25), 3),
    }

    return pd.DataFrame(data)


@pytest.fixture
def cluster_ordering():
    """Example cluster ordering"""
    return ["typeA", "typeB"]
