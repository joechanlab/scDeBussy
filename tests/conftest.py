import anndata as ad
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


@pytest.fixture
def sample_adata():
    """Create a small AnnData object with two shifted patient trajectories."""
    patient1_pt = np.linspace(0.05, 0.95, 8)
    patient2_pt = np.linspace(0.1, 0.9, 8)
    pseudotime = np.concatenate([patient1_pt, patient2_pt])
    patient = np.array(["patient1"] * len(patient1_pt) + ["patient2"] * len(patient2_pt))
    cell_type = np.array(["early" if value < 0.5 else "late" for value in pseudotime])

    gene1 = pseudotime
    gene2 = 1.0 - pseudotime
    gene3 = np.sin(np.pi * pseudotime)
    X = np.column_stack([gene1, gene2, gene3]).astype(float)
    X[len(patient1_pt) :, 0] += 0.05
    X[len(patient1_pt) :, 2] += 0.03

    return ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {
                "patient": patient,
                "s_local": pseudotime,
                "cell_type": cell_type,
            },
            index=[f"cell_{idx}" for idx in range(len(pseudotime))],
        ),
        var=pd.DataFrame(index=["gene1", "gene2", "gene3"]),
    )
