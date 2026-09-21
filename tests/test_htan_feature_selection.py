"""A patient offset cannot outrank reproducible temporal signal by itself."""

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation/v21/experiment_round"))
from select_trajectory_genes import score_genes


def test_label_free_selection_excludes_offset_and_heldout_gene():
    time = np.tile(np.linspace(0, 1, 100), 4)
    patient = np.repeat(["p0", "p1", "p2", "p3"], 100)
    x = np.column_stack((2 * time, np.repeat([0, 5, 10, 15], 100), 3 * time))
    data = ad.AnnData(x, obs=pd.DataFrame({"patient": patient, "s_local": time}),
                      var=pd.DataFrame(index=["temporal", "offset", "heldout"]))
    genes, score, *_ = score_genes(data, ["heldout"], n_bins=10, min_cells=3)
    scores = dict(zip(genes, score))
    assert np.isfinite(scores["temporal"])
    assert not np.isfinite(scores["offset"])
    assert not np.isfinite(scores["heldout"])
