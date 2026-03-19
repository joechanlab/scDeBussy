import warnings

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def cross_batch_knn_purity(
    adata,
    pseudotime_key="aligned_pseudotime",
    label_key="cell_type",
    batch_key="patient",
    n_neighbors=50,
    key_added="cross_batch_purity",
    inplace=True,
):
    """Calculate cross-batch KNN purity, ignoring cell types unique to a single batch."""
    if not all(k in adata.obs for k in [pseudotime_key, label_key, batch_key]):
        raise ValueError("pseudotime_key, label_key, or batch_key missing from adata.obs.")
    valid_mask = ~adata.obs[pseudotime_key].isna()
    X_1d = adata.obs.loc[valid_mask, pseudotime_key].values.reshape(-1, 1)
    labels = adata.obs.loc[valid_mask, label_key].values
    batches = adata.obs.loc[valid_mask, batch_key].values
    if len(X_1d) < n_neighbors:
        raise ValueError("Number of valid cells is less than n_neighbors.")

    # Singleton cell types that only appear in one batch, as cross-batch purity is undefined for them
    df = pd.DataFrame({label_key: labels, batch_key: batches})
    batch_counts = df.groupby(label_key)[batch_key].nunique()
    singleton_types = batch_counts[batch_counts == 1].index.tolist()
    if singleton_types:
        warnings.warn(
            f"The following cell types only appear in 1 batch and will be excluded "
            f"from the cross-batch purity score: {singleton_types}",
            stacklevel=2,
        )

    # Fit KNN model
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="kd_tree").fit(X_1d)
    _, indices = nbrs.kneighbors(X_1d)
    scores = []
    for i in range(len(X_1d)):
        my_label = labels[i]
        if my_label in singleton_types:
            scores.append(np.nan)
            continue
        my_batch = batches[i]
        neighbor_idx = indices[i]
        neighbor_labels = labels[neighbor_idx]
        neighbor_batches = batches[neighbor_idx]
        cross_batch_mask = neighbor_batches != my_batch
        valid_cross_batch_labels = neighbor_labels[cross_batch_mask]
        if len(valid_cross_batch_labels) == 0:
            scores.append(0.0)
        else:
            match_count = np.sum(valid_cross_batch_labels == my_label)
            scores.append(match_count / len(valid_cross_batch_labels))
    scores = np.array(scores)
    results = {
        "global_cross_batch_purity": float(np.nanmean(scores)),
        "per_class_purity": {},
        "excluded_singleton_types": singleton_types,
        "n_neighbors_queried": n_neighbors,
    }
    unique_labels = np.unique(labels)
    for label in unique_labels:
        class_mask = labels == label
        if label not in singleton_types and class_mask.sum() > 0:
            results["per_class_purity"][label] = float(np.nanmean(scores[class_mask]))
    if inplace:
        adata.uns[key_added] = results
    else:
        return results


def cross_batch_purity_sweep(
    adata,
    k_values=(15, 30, 50, 75, 100),
    pseudotime_key="aligned_pseudotime",
    label_key="cell_type",
    batch_key="patient",
    key_added="purity_sweep",
):
    """Compute cross-batch KNN purity across multiple neighborhood sizes (k)."""
    scores = {}
    for k in k_values:
        res = cross_batch_knn_purity(
            adata, n_neighbors=k, pseudotime_key=pseudotime_key, label_key=label_key, batch_key=batch_key, inplace=False
        )
        scores[k] = res["global_cross_batch_purity"]
    adata.uns[key_added] = scores
