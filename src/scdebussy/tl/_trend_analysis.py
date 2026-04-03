import itertools

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans

from ._metrics import cross_batch_knn_purity


def _require_barycenter_expression(adata, barycenter_key: str = "barycenter") -> np.ndarray:
    if barycenter_key not in adata.uns:
        raise ValueError(f"Missing adata.uns[{barycenter_key!r}]. Run scDeBussy() first.")
    barycenter = adata.uns[barycenter_key]
    if not isinstance(barycenter, dict) or "expression" not in barycenter:
        raise ValueError(f"adata.uns[{barycenter_key!r}] must be a dict containing key 'expression'.")

    expression = np.asarray(barycenter["expression"], dtype=float)
    if expression.ndim != 2:
        raise ValueError("Barycenter expression must be a 2D array with shape (n_bins, n_genes).")
    if expression.shape[1] != adata.n_vars:
        raise ValueError(
            "Barycenter expression gene dimension does not match adata.n_vars. "
            f"Got {expression.shape[1]} vs {adata.n_vars}."
        )
    return expression


def compute_gene_trend_features(adata, barycenter_key: str = "barycenter") -> pd.DataFrame:
    """Compute barycenter-derived per-gene temporal trend features.

    Returns a DataFrame indexed by gene name with columns suitable for clustering
    and deterministic ordering.
    """
    expression = _require_barycenter_expression(adata, barycenter_key=barycenter_key)
    n_bins = expression.shape[0]
    tau = np.linspace(0.0, 1.0, n_bins)

    peak_bin = np.argmax(expression, axis=0)
    peak_tau = tau[peak_bin]
    peak_value = expression[peak_bin, np.arange(expression.shape[1])]
    mean_expression = expression.mean(axis=0)
    std_expression = expression.std(axis=0)

    # Preserve edge-peaking genes: if both trajectory ends sit near the gene's
    # maximum, use the stronger edge as the ordering anchor instead of letting a
    # centroid-style summary collapse the gene toward the middle.
    expr_min = expression.min(axis=0)
    expr_span = peak_value - expr_min
    start_relative_peak = np.divide(
        expression[0, :] - expr_min,
        expr_span,
        out=np.zeros_like(expr_span),
        where=expr_span > 0,
    )
    end_relative_peak = np.divide(
        expression[-1, :] - expr_min,
        expr_span,
        out=np.zeros_like(expr_span),
        where=expr_span > 0,
    )
    edge_bimodal = (start_relative_peak >= 0.9) & (end_relative_peak >= 0.9)
    ordering_anchor_tau = peak_tau.copy()
    ordering_anchor_tau[edge_bimodal] = np.where(
        start_relative_peak[edge_bimodal] >= end_relative_peak[edge_bimodal],
        0.0,
        1.0,
    )

    shifted = expression - expression.min(axis=0, keepdims=True)
    denom = shifted.sum(axis=0)
    temporal_centroid = np.divide(
        (shifted * tau[:, None]).sum(axis=0), denom, out=np.zeros_like(denom), where=denom > 0
    )
    temporal_spread = np.divide(
        (shifted * (tau[:, None] - temporal_centroid[None, :]) ** 2).sum(axis=0),
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )

    rise = np.maximum(0.0, np.diff(expression, axis=0)).sum(axis=0)
    fall = np.maximum(0.0, -np.diff(expression, axis=0)).sum(axis=0)

    return (
        pd.DataFrame(
            {
                "gene": adata.var_names.to_numpy(dtype=str),
                "peak_bin": peak_bin,
                "peak_tau": peak_tau,
                "peak_value": peak_value,
                "mean_expression": mean_expression,
                "std_expression": std_expression,
                "start_relative_peak": start_relative_peak,
                "end_relative_peak": end_relative_peak,
                "edge_bimodal": edge_bimodal,
                "ordering_anchor_tau": ordering_anchor_tau,
                "temporal_centroid": temporal_centroid,
                "temporal_spread": temporal_spread,
                "rise_total": rise,
                "fall_total": fall,
            }
        )
        .set_index("gene", drop=False)
        .rename_axis(None)
    )


def cluster_and_order_genes(
    adata,
    barycenter_key: str = "barycenter",
    n_clusters: int = 3,
    random_state: int = 0,
    peak_weight: float = 0.7,
) -> pd.DataFrame:
    """Cluster genes by trend shape and provide deterministic within-cluster ordering."""
    if not 0.0 <= peak_weight <= 1.0:
        raise ValueError("peak_weight must be in [0, 1].")

    features = compute_gene_trend_features(adata, barycenter_key=barycenter_key).copy()
    n_genes = len(features)
    if n_genes == 0:
        raise ValueError("No genes found in adata.var_names.")

    k = max(1, min(int(n_clusters), n_genes))
    # Cluster primarily by where maximal expression sits along pseudotime. Shape
    # features are retained as low-weight tie-breakers but should not be strong
    # enough to pull edge-peaking genes back toward the middle.
    feature_cols = ["ordering_anchor_tau", "temporal_spread", "rise_total", "fall_total"]
    X = features[feature_cols].to_numpy(dtype=float)
    means = X.mean(axis=0, keepdims=True)
    stds = X.std(axis=0, keepdims=True)
    Xz = np.divide(X - means, stds, out=np.zeros_like(X), where=stds > 0)
    Xz[:, 0] *= 3.0
    Xz[:, 1:] *= 0.35

    labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(Xz)
    features["trend_cluster"] = labels

    # Occurrence score favors robustly expressed genes; ordering is anchored to
    # where the strongest expression sits along pseudotime, with edge-aware
    # handling for begin+end-upregulated genes.
    mean_norm = _minmax(features["mean_expression"].to_numpy(dtype=float))
    anchor_norm = _minmax(features["ordering_anchor_tau"].to_numpy(dtype=float))
    features["occurrence_score"] = mean_norm
    features["ordering_score"] = peak_weight * anchor_norm + (1.0 - peak_weight) * (1.0 - mean_norm)

    cluster_centroids = (
        features.groupby("trend_cluster", observed=True)["ordering_anchor_tau"].median().sort_values().index.to_list()
    )
    cluster_rank = {c: i for i, c in enumerate(cluster_centroids)}
    features["cluster_order"] = features["trend_cluster"].map(cluster_rank)

    ordered = features.sort_values(
        by=["cluster_order", "ordering_anchor_tau", "ordering_score", "occurrence_score", "gene"],
        ascending=[True, True, True, False, True],
    ).copy()
    ordered["global_order_rank"] = np.arange(len(ordered), dtype=int)
    return ordered


def cosegment_ordered_heatmap(
    matrix,
    *,
    n_row_clusters: int = 3,
    n_col_clusters: int = 3,
    min_rows_per_cluster: int = 1,
    min_cols_per_cluster: int = 1,
    max_iter: int = 25,
) -> dict:
    """Contiguous biclustering for an already ordered heatmap.

    Rows and columns are partitioned into consecutive segments by alternating
    dynamic programming updates that minimize within-block squared error.
    """
    X = np.asarray(matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError("matrix must be 2D.")

    n_rows, n_cols = X.shape
    if n_rows == 0 or n_cols == 0:
        raise ValueError("matrix must have at least one row and one column.")

    row_k = max(1, min(int(n_row_clusters), n_rows))
    col_k = max(1, min(int(n_col_clusters), n_cols))
    min_rows = max(1, min(int(min_rows_per_cluster), max(1, n_rows // row_k)))
    min_cols = max(1, min(int(min_cols_per_cluster), max(1, n_cols // col_k)))

    if row_k * min_rows > n_rows:
        raise ValueError("n_row_clusters * min_rows_per_cluster exceeds number of rows.")
    if col_k * min_cols > n_cols:
        raise ValueError("n_col_clusters * min_cols_per_cluster exceeds number of columns.")

    prefix = np.pad(np.cumsum(np.cumsum(X, axis=0), axis=1), ((1, 0), (1, 0)))
    prefix_sq = np.pad(np.cumsum(np.cumsum(X**2, axis=0), axis=1), ((1, 0), (1, 0)))

    row_segments = _equal_size_segments(n_rows, row_k)
    col_segments = _equal_size_segments(n_cols, col_k)
    previous_objective = np.inf

    for _ in range(int(max_iter)):
        row_segments, row_obj = _fit_axis_segments(
            n_items=n_rows,
            n_segments=row_k,
            min_size=min_rows,
            interval_cost=lambda left, right, col_segments=col_segments: _row_interval_cost(
                left, right, col_segments, prefix, prefix_sq
            ),
        )
        col_segments, col_obj = _fit_axis_segments(
            n_items=n_cols,
            n_segments=col_k,
            min_size=min_cols,
            interval_cost=lambda left, right, row_segments=row_segments: _col_interval_cost(
                left, right, row_segments, prefix, prefix_sq
            ),
        )

        objective = row_obj + col_obj
        if abs(previous_objective - objective) <= 1e-9:
            break
        previous_objective = objective

    block_means = np.empty((len(row_segments), len(col_segments)), dtype=float)
    for row_idx, (r0, r1) in enumerate(row_segments):
        for col_idx, (c0, c1) in enumerate(col_segments):
            block_sum, _ = _rect_stats(prefix, prefix_sq, r0, r1, c0, c1)
            block_means[row_idx, col_idx] = block_sum / max((r1 - r0) * (c1 - c0), 1)

    return {
        "row_segments": row_segments,
        "col_segments": col_segments,
        "block_means": block_means,
        "objective": float(previous_objective),
    }


def compute_trend_recurrence_score(
    adata,
    aligned_pseudotime_key: str = "aligned_pseudotime",
    patient_key: str = "patient",
    n_bins: int = 50,
) -> dict:
    """Estimate recurrence as mean pairwise trajectory correlation across patients."""
    if aligned_pseudotime_key not in adata.obs or patient_key not in adata.obs:
        raise ValueError("aligned_pseudotime_key and patient_key must exist in adata.obs.")

    patient_ids = adata.obs[patient_key].astype(str).unique().tolist()
    if len(patient_ids) < 2:
        return {
            "mean_cross_patient_trajectory_corr": float("nan"),
            "n_pairs": 0,
            "pairwise_correlations": {},
        }

    grid = np.linspace(0.0, 1.0, int(n_bins))
    mats = {}
    for pid in patient_ids:
        mask = adata.obs[patient_key].astype(str) == pid
        pt = adata.obs.loc[mask, aligned_pseudotime_key].to_numpy(dtype=float)
        valid = np.isfinite(pt)
        if valid.sum() < 2:
            continue

        X = adata[mask].X
        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=float)
        X = X[valid]
        pt = pt[valid]
        sort_idx = np.argsort(pt)
        pt = pt[sort_idx]
        X = X[sort_idx]

        interp = np.vstack([np.interp(grid, pt, X[:, j]) for j in range(X.shape[1])]).T
        interp_mean = interp.mean(axis=0, keepdims=True)
        interp_std = interp.std(axis=0, keepdims=True)
        interp = np.divide(interp - interp_mean, interp_std, out=np.zeros_like(interp), where=interp_std > 0)
        mats[pid] = interp.ravel()

    pair_corr = {}
    for a, b in itertools.combinations(sorted(mats), 2):
        x = mats[a]
        y = mats[b]
        if x.size == 0 or y.size == 0:
            corr = np.nan
        else:
            corr = float(np.corrcoef(x, y)[0, 1])
        pair_corr[f"{a}__{b}"] = corr

    vals = np.array([v for v in pair_corr.values() if np.isfinite(v)], dtype=float)
    return {
        "mean_cross_patient_trajectory_corr": float(np.mean(vals)) if len(vals) else float("nan"),
        "n_pairs": int(len(pair_corr)),
        "pairwise_correlations": pair_corr,
    }


def _equal_size_segments(n_items: int, n_segments: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, n_items, n_segments + 1).round().astype(int)
    edges[0] = 0
    edges[-1] = n_items
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_segments)]


def _fit_axis_segments(
    n_items: int, n_segments: int, min_size: int, interval_cost
) -> tuple[list[tuple[int, int]], float]:
    dp = np.full((n_segments + 1, n_items + 1), np.inf, dtype=float)
    prev = np.full((n_segments + 1, n_items + 1), -1, dtype=int)
    dp[0, 0] = 0.0

    for segment_idx in range(1, n_segments + 1):
        end_min = segment_idx * min_size
        end_max = n_items - (n_segments - segment_idx) * min_size
        for end in range(end_min, end_max + 1):
            start_min = (segment_idx - 1) * min_size
            start_max = end - min_size
            for start in range(start_min, start_max + 1):
                candidate = dp[segment_idx - 1, start] + interval_cost(start, end)
                if candidate < dp[segment_idx, end]:
                    dp[segment_idx, end] = candidate
                    prev[segment_idx, end] = start

    if not np.isfinite(dp[n_segments, n_items]):
        raise RuntimeError("Failed to fit contiguous segments.")

    segments = []
    end = n_items
    for segment_idx in range(n_segments, 0, -1):
        start = int(prev[segment_idx, end])
        segments.append((start, end))
        end = start
    segments.reverse()
    return segments, float(dp[n_segments, n_items])


def _row_interval_cost(left: int, right: int, col_segments: list[tuple[int, int]], prefix, prefix_sq) -> float:
    return float(sum(_block_sse(prefix, prefix_sq, left, right, c0, c1) for c0, c1 in col_segments))


def _col_interval_cost(left: int, right: int, row_segments: list[tuple[int, int]], prefix, prefix_sq) -> float:
    return float(sum(_block_sse(prefix, prefix_sq, r0, r1, left, right) for r0, r1 in row_segments))


def _block_sse(prefix, prefix_sq, row_start: int, row_end: int, col_start: int, col_end: int) -> float:
    block_sum, block_sum_sq = _rect_stats(prefix, prefix_sq, row_start, row_end, col_start, col_end)
    size = (row_end - row_start) * (col_end - col_start)
    if size <= 0:
        return 0.0
    return float(max(block_sum_sq - (block_sum**2) / size, 0.0))


def _rect_stats(prefix, prefix_sq, row_start: int, row_end: int, col_start: int, col_end: int) -> tuple[float, float]:
    block_sum = (
        prefix[row_end, col_end]
        - prefix[row_start, col_end]
        - prefix[row_end, col_start]
        + prefix[row_start, col_start]
    )
    block_sum_sq = (
        prefix_sq[row_end, col_end]
        - prefix_sq[row_start, col_end]
        - prefix_sq[row_end, col_start]
        + prefix_sq[row_start, col_start]
    )
    return float(block_sum), float(block_sum_sq)


def compute_composite_alignment_score(
    adata,
    *,
    aligned_pseudotime_key: str = "aligned_pseudotime",
    label_key: str = "cell_type",
    patient_key: str = "patient",
    n_neighbors: int = 50,
    weight_global_purity: float = 1.0,
    weight_class_stability: float = 1.0,
    weight_recurrence: float = 1.0,
) -> dict:
    """Compute a composite alignment score from purity and recurrence components."""
    purity = cross_batch_knn_purity(
        adata,
        pseudotime_key=aligned_pseudotime_key,
        label_key=label_key,
        batch_key=patient_key,
        n_neighbors=n_neighbors,
        inplace=False,
    )
    if purity is None:
        raise RuntimeError("cross_batch_knn_purity returned None unexpectedly when inplace=False.")

    per_class = purity.get("per_class_purity", {})
    per_class_vals = np.array(list(per_class.values()), dtype=float) if per_class else np.array([], dtype=float)
    class_stability = float(1.0 - np.nanstd(per_class_vals)) if len(per_class_vals) else float("nan")
    class_stability = float(np.clip(class_stability, 0.0, 1.0)) if np.isfinite(class_stability) else float("nan")

    recurrence = compute_trend_recurrence_score(
        adata,
        aligned_pseudotime_key=aligned_pseudotime_key,
        patient_key=patient_key,
    )
    recurrence_raw = recurrence["mean_cross_patient_trajectory_corr"]
    recurrence_scaled = float((recurrence_raw + 1.0) / 2.0) if np.isfinite(recurrence_raw) else float("nan")

    parts = {
        "global_purity": float(purity["global_cross_batch_purity"]),
        "class_stability": class_stability,
        "recurrence": recurrence_scaled,
    }
    weights = {
        "global_purity": float(weight_global_purity),
        "class_stability": float(weight_class_stability),
        "recurrence": float(weight_recurrence),
    }

    weighted_sum = 0.0
    weight_total = 0.0
    for key, w in weights.items():
        v = parts[key]
        if np.isfinite(v) and w > 0:
            weighted_sum += w * v
            weight_total += w
    composite = float(weighted_sum / weight_total) if weight_total > 0 else float("nan")

    return {
        "composite_score": composite,
        "components": parts,
        "weights": weights,
        "purity": purity,
        "recurrence": recurrence,
    }


def temporal_kernel_gene_set_rankings(
    ordered_genes: list[str],
    gene_sets: dict[str, list[str]],
    sigma: float = 25.0,
) -> pd.DataFrame:
    """Score gene-set density along an ordered gene list using a Gaussian kernel."""
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if not ordered_genes:
        raise ValueError("ordered_genes must not be empty.")

    n = len(ordered_genes)
    index = {g: i for i, g in enumerate(ordered_genes)}
    positions = np.arange(n, dtype=float)

    rows = []
    for set_name, genes in gene_sets.items():
        hits = np.array([index[g] for g in genes if g in index], dtype=int)
        if len(hits) == 0:
            continue
        hit_mask = np.zeros(n, dtype=float)
        hit_mask[hits] = 1.0

        for center in range(n):
            kernel = np.exp(-0.5 * ((positions - center) / sigma) ** 2)
            score = float(np.sum(kernel * hit_mask) / np.sum(kernel))
            rows.append({"gene_set": set_name, "center_rank": center, "kernel_score": score})

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["set_rank"] = out.groupby("center_rank", observed=True)["kernel_score"].rank(method="dense", ascending=False)
    return out.sort_values(["center_rank", "set_rank", "gene_set"]).reset_index(drop=True)


def temporal_kernel_pseudotime_enrichment(
    adata,
    gene_sets: dict[str, list[str]],
    barycenter_key: str = "barycenter",
) -> pd.DataFrame:
    """Compute per-bin enrichment scores for each gene set on barycenter expression."""
    expression = _require_barycenter_expression(adata, barycenter_key=barycenter_key)
    genes = adata.var_names.to_numpy(dtype=str)
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    bg = expression.mean(axis=1)
    rows = []
    for set_name, set_genes in gene_sets.items():
        idx = [gene_to_idx[g] for g in set_genes if g in gene_to_idx]
        if not idx:
            continue
        set_signal = expression[:, idx].mean(axis=1)
        score = set_signal - bg
        for b, s in enumerate(score):
            rows.append({"gene_set": set_name, "bin": int(b), "enrichment_score": float(s)})

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["zscore_within_set"] = out.groupby("gene_set", observed=True)["enrichment_score"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-12)
    )
    return out.sort_values(["gene_set", "bin"]).reset_index(drop=True)


def evaluate_basal_signature_prominence(
    ordered_genes: list[str],
    basal_genes: list[str],
    n_permutations: int = 1000,
    seed: int = 0,
) -> dict:
    """Test whether basal genes are concentrated toward top ranks of an ordered list."""
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1.")
    if not ordered_genes:
        raise ValueError("ordered_genes must not be empty.")

    index = {g: i for i, g in enumerate(ordered_genes)}
    hits = [index[g] for g in basal_genes if g in index]
    if not hits:
        return {
            "n_hits": 0,
            "n_basal_input": int(len(basal_genes)),
            "mean_inverse_rank": float("nan"),
            "permutation_pvalue": float("nan"),
            "effect_size_z": float("nan"),
        }

    n = len(ordered_genes)
    ranks = np.array(hits, dtype=float)
    observed = float(np.mean(1.0 - ranks / max(1, n - 1)))

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=float)
    m = len(hits)
    for i in range(n_permutations):
        sampled = rng.choice(n, size=m, replace=False)
        null[i] = np.mean(1.0 - sampled / max(1, n - 1))

    pvalue = float((1 + np.sum(null >= observed)) / (1 + n_permutations))
    effect = float((observed - null.mean()) / (null.std() + 1e-12))
    return {
        "n_hits": int(len(hits)),
        "n_basal_input": int(len(basal_genes)),
        "mean_inverse_rank": observed,
        "permutation_pvalue": pvalue,
        "effect_size_z": effect,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
    }


def compute_running_enrichment(
    ordered_genes,
    gene_set_dict: dict[str, list[str]],
    window_size: int = 100,
    step_size: int = 20,
) -> pd.DataFrame:
    """Compute running enrichment scores for gene sets along an ordered gene list.

    Uses a hypergeometric sliding-window test to measure how many genes from each
    gene set fall within each window, returning a signed –log10(p-value) score.

    Parameters
    ----------
    ordered_genes : list or array-like of str
        Genes in temporal order (early → late).
    gene_set_dict : dict
        Dictionary mapping gene set names to lists of gene symbols.
    window_size : int, optional
        Number of genes in each sliding window. Defaults to 100.
    step_size : int, optional
        Step between consecutive window positions. Defaults to 20.

    Returns
    -------
    pd.DataFrame
        Columns: ``gene_set``, ``position`` (window centre), ``start``, ``end``,
        ``overlap``, ``expected``, ``enrichment_ratio``, ``pval``, ``score``
        (signed –log10 p-value).
    """
    from scipy.stats import hypergeom

    ordered_genes = list(ordered_genes)
    total_genes = len(ordered_genes)

    rows = []
    for gene_set_name, gene_set in gene_set_dict.items():
        gene_set_present = [g for g in gene_set if g in ordered_genes]
        n_gene_set = len(gene_set_present)
        if n_gene_set < 5:
            continue

        for start in range(0, total_genes - window_size + 1, step_size):
            end = start + window_size
            window_genes = ordered_genes[start:end]
            overlap = len(set(window_genes) & set(gene_set_present))

            # Hypergeometric: M=total, n=set size, N=window, k=overlap
            pval = float(hypergeom.sf(overlap - 1, total_genes, n_gene_set, window_size))
            expected = (window_size * n_gene_set) / total_genes
            enrichment_ratio = overlap / expected if expected > 0 else 0.0

            if pval < 1e-10:
                score = 10.0 * np.sign(overlap - expected)
            else:
                score = -np.log10(pval + 1e-10) * np.sign(overlap - expected)

            rows.append(
                {
                    "gene_set": gene_set_name,
                    "position": (start + end) / 2.0,
                    "start": start,
                    "end": end,
                    "overlap": overlap,
                    "expected": expected,
                    "enrichment_ratio": enrichment_ratio,
                    "pval": pval,
                    "score": score,
                }
            )

    return pd.DataFrame(rows)


def _minmax(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    lo = np.min(v)
    hi = np.max(v)
    if hi <= lo:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


__all__ = [
    "cosegment_ordered_heatmap",
    "compute_gene_trend_features",
    "cluster_and_order_genes",
    "compute_trend_recurrence_score",
    "compute_composite_alignment_score",
    "temporal_kernel_gene_set_rankings",
    "temporal_kernel_pseudotime_enrichment",
    "evaluate_basal_signature_prominence",
    "compute_running_enrichment",
]
