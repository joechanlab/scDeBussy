# Modified based on https://github.com/tslearn-team/tslearn/blob/5568c026db4b4380b99095827e0573a8f55a81f0/tslearn/barycenters/dba.py
import warnings

import numpy as np
import tslearn.barycenters.dba as dba
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder
from tslearn.barycenters.utils import _set_weights
from tslearn.metrics import dtw_path
from tslearn.utils import to_time_series_dataset, ts_size

from ._utils import compute_gmm_cutpoints


def _convert_categorical(Y, verbose=False):
    """Convert categorical variables to numeric if needed and return mapping.

    Parameters
    ----------
    Y : array-like, shape=(n_ts, sz)
        Categorical variables (numeric or non-numeric)

    Returns
    -------
    Y_numeric : numpy.array of shape (n_ts, sz)
        Numerically encoded categorical variables
    label_encoder : LabelEncoder or None
        The encoder used for conversion (None if Y was already numeric)
    """
    # Check if Y is already numeric
    if hasattr(Y, "dtype"):
        if np.issubdtype(Y.dtype, np.number):
            if verbose:
                print("Y is already numeric categorical. No conversion needed.")
            return Y, None
    else:
        # Check if all elements are numeric
        try:
            if all(isinstance(item, int | float) for row in Y for item in row):
                if verbose:
                    print("Y is already numeric categorical. No conversion needed.")
                return Y, None
        except TypeError:  # Handle case where items aren't iterable
            if all(isinstance(item, int | float) for item in Y):
                if verbose:
                    print("Y is already numeric categorical. No conversion needed.")
                return Y, None

    # Convert non-numeric categories to numeric
    label_encoder = LabelEncoder()
    Y_flat = [item for row in Y for item in row]
    label_encoder.fit(Y_flat)
    Y_numeric = [[label_encoder.transform([item])[0] for item in row] for row in Y]

    # Print mapping
    if verbose:
        print("Category mapping:")
        for i, category in enumerate(label_encoder.classes_):
            print(f"  {category} -> {i}")

    return Y_numeric, label_encoder


def _handle_majority_vote(values, weights=None, tie_strategy="first"):
    """Handle majority voting with explicit tie handling.

    Parameters
    ----------
    values : array-like
        Values to vote on
    weights : array-like or None
        Weights for each value (if None, equal weights are used)
    tie_strategy : str
        Strategy to handle ties. Options:
        - 'first': Select the first value that appears
        - 'random': Randomly select among tied values
        - 'weighted_random': Randomly select among tied values with weights
        - 'report': Return all tied values

    Returns
    -------
    result : scalar or array
        The winning value(s)
    is_tie : bool
        Whether there was a tie
    """
    if not values:
        return None, False

    values = np.array(values)
    if weights is None:
        weights = np.ones(len(values))
    weights = np.array(weights)

    # Get unique values and their weighted counts
    unique_vals, counts = np.unique(values, return_counts=True)
    weighted_counts = np.zeros_like(counts, dtype=float)

    for i, val in enumerate(unique_vals):
        mask = np.where(values == val)[0]
        weighted_counts[i] = np.sum(weights[mask])

    # Find maximum count and check for ties
    max_count = np.max(weighted_counts)
    tied_indices = np.where(weighted_counts == max_count)[0]
    is_tie = len(tied_indices) > 1
    tied_values = unique_vals[tied_indices]

    if not is_tie or tie_strategy == "first":
        return tied_values[0], is_tie

    elif tie_strategy == "random":
        return np.random.choice(tied_values), is_tie

    elif tie_strategy == "weighted_random":
        tied_weights = weighted_counts[tied_indices]
        tied_weights = tied_weights / tied_weights.sum()  # Normalize weights
        return np.random.choice(tied_values, p=tied_weights), is_tie

    elif tie_strategy == "report":
        return tied_values, is_tie

    else:
        raise ValueError(f"Unknown tie_strategy: {tie_strategy}")


def _mm_assignment(X, Y, barycenter, weights, metric_params=None):
    """Modified to also track Y alignments"""
    if metric_params is None:
        metric_params = {}
    n = X.shape[0]
    cost = 0.0
    list_p_k = []
    list_y_k = []
    dtw_paths = []
    for i in range(n):
        path, dist_i = dtw_path(barycenter, X[i], **metric_params)
        cost += dist_i**2 * weights[i]
        list_p_k.append(path)
        dtw_paths.append(path)
        y_aligned = [(i, path_point[1]) for path_point in path]
        list_y_k.append(y_aligned)
    cost /= weights.sum()
    return list_p_k, list_y_k, cost, dtw_paths


def _mm_update_barycenter_with_categories(
    X, Y, diag_sum_v_k, list_w_k, list_y_k, weights, tie_strategy="first", verbose=False
):
    """Update barycenter and determine majority categories"""
    d = X.shape[2]
    barycenter_size = diag_sum_v_k.shape[0]

    # Update numerical values (original logic)
    sum_w_x = np.zeros((barycenter_size, d))
    for w_k, x_k in zip(list_w_k, X, strict=False):
        sum_w_x += w_k.dot(x_k[: ts_size(x_k)])
    barycenter = np.diag(1.0 / diag_sum_v_k).dot(sum_w_x)

    # Determine majority categories for each time point
    categories = np.zeros(barycenter_size, dtype=int)
    for t in range(barycenter_size):
        # Collect all Y values aligned to this time point
        y_values = []
        y_weights = []
        for y_aligned in list_y_k:
            for idx, time_idx in y_aligned:
                if time_idx == t:
                    y_values.append(Y[idx][time_idx])
                    y_weights.append(weights[idx])

        # Get majority vote if we have any alignments
        if y_values:
            category, is_tie = _handle_majority_vote(y_values, weights=y_weights, tie_strategy=tie_strategy)

            if verbose and is_tie:
                print(f"Tie detected at time point {t}: {category}")

            # Handle multiple categories if tie_strategy is 'report'
            if isinstance(category, np.ndarray):
                categories[t] = category[0]  # Take first value if array
            else:
                categories[t] = category

    return barycenter, categories


def dtw_barycenter_averaging_with_categories_one_init(
    X,
    Y,
    barycenter_size=None,
    init_barycenter=None,
    max_iter=30,
    tol=1e-5,
    weights=None,
    metric_params=None,
    tie_strategy="first",
    verbose=False,
):
    """Single initialization version of DBA with categories"""
    X_ = to_time_series_dataset(X)
    if barycenter_size is None:
        barycenter_size = X_.shape[1]
    weights = _set_weights(weights, X_.shape[0])
    if init_barycenter is None:
        barycenter = dba._init_avg(X_, barycenter_size)
    else:
        barycenter_size = init_barycenter.shape[0]
        barycenter = init_barycenter

    cost_prev, cost = np.inf, np.inf
    categories = None
    final_dtw_paths = None

    for it in range(max_iter):
        # Modified to include Y in the assignment
        list_p_k, list_y_k, cost, dtw_paths = _mm_assignment(X_, Y, barycenter, weights, metric_params)
        diag_sum_v_k, list_w_k = dba._mm_valence_warping(list_p_k, barycenter_size, weights)
        if verbose:
            print(f"[DBA] epoch {it + 1}, cost: {cost:.3f}")

        # Update both barycenter and categories
        barycenter, categories = _mm_update_barycenter_with_categories(
            X_, Y, diag_sum_v_k, list_w_k, list_y_k, weights=weights, tie_strategy=tie_strategy, verbose=verbose
        )

        if abs(cost_prev - cost) < tol:
            final_dtw_paths = dtw_paths
            break
        elif cost_prev < cost:
            warnings.warn(
                "DBA loss is increasing while it should not be. Stopping optimization.",
                ConvergenceWarning,
                stacklevel=2,
            )
            final_dtw_paths = dtw_paths
            break
        else:
            cost_prev = cost
            final_dtw_paths = dtw_paths
    # Create aligned barycenter values for each series
    aligned_barycenters = []
    for path in final_dtw_paths:
        # Extract barycenter indices from the path
        barycenter_indices = [p[0] for p in path]
        # Get aligned barycenter values
        aligned_barycenter = barycenter[barycenter_indices]
        aligned_barycenters.append(aligned_barycenter)

    return barycenter, categories, cost, aligned_barycenters


def dtw_barycenter_averaging_with_categories(
    X,
    Y,
    barycenter_size=None,
    init_barycenter=None,
    max_iter=30,
    tol=1e-5,
    weights=None,
    metric_params=None,
    verbose=False,
    n_init=1,
    tie_strategy="first",
):
    """Modified DBA to also handle categorical variables.

    Parameters
    ----------
    X : array-like, shape=(n_ts, sz, d)
        Time series dataset (numerical values)

    Y : array-like, shape=(n_ts, sz)
        Categorical variables for each time point

    barycenter_size : int or None (default: None)
        Size of the barycenter to generate. If None, the size of the barycenter
        is that of the data provided at fit
        time or that of the initial barycenter if specified.

    init_barycenter : array or None (default: None)
        Initial barycenter to start from for the optimization process.

    max_iter : int (default: 30)
        Number of iterations of the Expectation-Maximization optimization
        procedure.

    tol : float (default: 1e-5)
        Tolerance to use for early stopping: if the decrease in cost is lower
        than this value, the
        Expectation-Maximization procedure stops.

    weights: None or array
        Weights of each X[i]. Must be the same size as len(X).
        If None, uniform weights are used.

    metric_params: dict or None (default: None)
        DTW constraint parameters to be used.
        See :ref:`tslearn.metrics.dtw_path <fun-tslearn.metrics.dtw_path>` for
        a list of accepted parameters
        If None, no constraint is used for DTW computations.

    verbose : boolean (default: False)
        Whether to print information about the cost at each iteration or not.

    n_init : int (default: 1)
        Number of different initializations to be tried (useful only is
        init_barycenter is set to None, otherwise, all trials will reach the
        same performance)

    tie_strategy : str (default='first')
        Strategy to handle ties in majority voting:
        - 'first': Select the first value that appears
        - 'random': Randomly select among tied values
        - 'weighted_random': Randomly select among tied values with weights
        - 'report': Report all tied values

    Returns
    -------
    tuple (barycenter, categories)
        barycenter : numpy.array of shape (barycenter_size, d)
            DBA barycenter of the provided time series dataset
        categories : numpy.array of shape (barycenter_size,)
            Majority vote categories for each time point
    """
    Y_numeric, label_encoder = _convert_categorical(Y, verbose=verbose)

    best_cost = np.inf
    best_barycenter = None
    best_categories = None

    for i in range(n_init):
        if verbose:
            print(f"Attempt {i + 1}")
        bary, cats, loss, aligned_bary = dtw_barycenter_averaging_with_categories_one_init(
            X=X,
            Y=Y_numeric,
            barycenter_size=barycenter_size,
            init_barycenter=init_barycenter,
            max_iter=max_iter,
            tol=tol,
            weights=weights,
            metric_params=metric_params,
            tie_strategy=tie_strategy,
            verbose=verbose,
        )
        if loss < best_cost:
            best_cost = loss
            best_barycenter = bary
            best_categories = cats
            best_aligned_barycenters = aligned_bary

    if label_encoder is not None:
        best_categories = label_encoder.inverse_transform(best_categories)

    return best_barycenter, best_categories, best_aligned_barycenters


def _initialize_barycenter(X, Y, barycenter_size):
    """Initialize barycenter and categories through interpolation."""
    n_dims = len(X[0][0])

    # Interpolate sequences and categories
    resampled_x = []
    resampled_y = []
    for x, y in zip(X, Y, strict=False):
        # Interpolate X values
        x = np.array(x).reshape(-1, n_dims)
        indices = np.linspace(0, len(x) - 1, barycenter_size)
        resampled_x.append(np.array([np.interp(indices, range(len(x)), x[:, d]) for d in range(n_dims)]).T)
        # Interpolate Y values and round to nearest integer
        resampled_y.append(np.round(np.interp(indices, range(len(y)), y)).astype(int))

    # Initialize barycenter and categories
    barycenter = np.mean(resampled_x, axis=0)
    barycenter_cats = np.round(np.mean(resampled_y, axis=0)).astype(int)

    return barycenter, barycenter_cats


def _compute_all_cutpoints(X, Y_numeric):
    """Compute GMM cutpoints for all sequences."""
    all_cutpoints = []
    for x, y in zip(X, Y_numeric, strict=False):
        # Convert x to numpy array if it's a list
        x = np.array(x) if isinstance(x, list) else x
        # Stack all dimensions with y
        data = np.column_stack([x, y])
        cutpoints = compute_gmm_cutpoints(data, len(np.unique(y)))
        all_cutpoints.append(cutpoints[0:-1])  # Skip last cutpoint
    return all_cutpoints


def dtw_barycenter_averaging_with_segments(
    X, Y, barycenter_size=None, max_iter=30, tol=1e-5, weights=None, metric_params=None, verbose=False, n_init=1
):
    """Main function for DTW barycenter averaging with segments."""
    best_cost = np.inf
    best_barycenter = best_categories = best_aligned = None
    # Set barycenter size if not specified
    if barycenter_size is None:
        barycenter_size = int(np.median([len(x) for x in X]))
    for i in range(n_init):
        if verbose:
            print(f"Initialization {i + 1}/{n_init}")
        # Convert categories and initialize
        Y_numeric, label_encoder = _convert_categorical(Y, verbose=verbose)
        barycenter, barycenter_cats = _initialize_barycenter(X, Y_numeric, barycenter_size)
        all_cutpoints = _compute_all_cutpoints(X, Y_numeric)  # note that this is dimension-first
        prev_cost = np.inf
        # Main iteration loop
        for it in range(max_iter):
            if verbose:
                print(f"Iteration {it + 1}/{max_iter}")
            # Compute reference cutpoints for barycenter
            barycenter_data = np.column_stack([barycenter, barycenter_cats])
            reference_cutpoints = compute_gmm_cutpoints(barycenter_data, len(np.unique(barycenter_cats)))[0:-1]
            # Align sequences
            aligned_barycenters = []
            aligned_categories = []
            total_cost = 0
            for x, y, cutpoints in zip(X, Y_numeric, all_cutpoints, strict=False):
                # Convert x to numpy array if not already
                x = np.array(x)
                # Find split indices for x based on cutpoints
                split_indices = []  # need to convert to cutpoint first, instead of dimension first

                cutpoints = np.array(cutpoints).T
                for cutpoint in cutpoints:
                    # Calculate distances to the cutpoint for each point in x
                    distances = np.linalg.norm(x - cutpoint, axis=1)
                    split_idx = np.argmin(distances)
                    split_indices.append(split_idx)
                split_indices = sorted(split_indices)
                # Find split indices for barycenter based on reference cutpoints
                ref_split_indices = []
                reference_cutpoints = np.array(reference_cutpoints).T
                for ref_cutpoint in reference_cutpoints:
                    distances = np.linalg.norm(barycenter - ref_cutpoint, axis=1)
                    ref_split_idx = np.argmin(distances)
                    ref_split_indices.append(ref_split_idx)
                ref_split_indices = sorted(ref_split_indices)

                # Process segments
                prev_ref_idx = 0
                sequence_aligned_values = []
                sequence_aligned_cats = []
                sequence_cost = 0
                # Process all segments
                for i in range(len(split_indices) + 1):
                    # Get current segment boundaries\
                    curr_ref_idx = ref_split_indices[i] if i < len(ref_split_indices) else len(barycenter)
                    # Align current segment
                    aligned_values, aligned_cats, paths, cost = align_segment(
                        x,
                        y,
                        barycenter,
                        start_idx=prev_ref_idx,
                        end_idx=curr_ref_idx,
                        barycenter_start=prev_ref_idx,
                        barycenter_end=curr_ref_idx,
                        weights=weights,
                        metric_params=metric_params,
                        verbose=verbose,
                    )
                    sequence_aligned_values.extend(aligned_values)
                    sequence_aligned_cats.extend(aligned_cats)
                    sequence_cost += cost
                    # Update indices for next segment
                    prev_ref_idx = curr_ref_idx
                # Store results for this sequence
                aligned_barycenters.extend(sequence_aligned_values)
                aligned_categories.extend(sequence_aligned_cats)
                total_cost += sequence_cost
            # Update barycenter and categories
            print(aligned_barycenters)
            print(aligned_categories)
            barycenter = np.mean(aligned_barycenters, axis=0)
            barycenter_cats = np.round(np.mean(aligned_categories, axis=0)).astype(int)
            # Check convergence
            if abs(prev_cost - total_cost) < tol:
                break
            prev_cost = total_cost
        # Update best results
        if total_cost < best_cost:
            best_cost = total_cost
            best_barycenter = barycenter
            best_categories = barycenter_cats
            best_aligned = aligned_barycenters
    # Convert categories back if needed
    if label_encoder is not None:
        best_categories = label_encoder.inverse_transform(best_categories)
    return best_barycenter, best_categories, best_aligned


def align_segment(
    X,
    Y,
    barycenter,
    start_idx,
    end_idx,
    barycenter_start,
    barycenter_end,
    weights=None,
    metric_params=None,
    tie_strategy="first",
    verbose=False,
):
    """Align a single segment using DTW.

    Parameters
    ----------
    X : array-like, shape=(n_ts, sz, d)
        Segment of time series to align
    Y : array-like, shape=(n_ts, sz)
        Categorical variables for the segment
    barycenter : array-like, shape=(barycenter_sz, d)
        Reference barycenter segment
    start_idx, end_idx : int
        Start and end indices for the input segment
    barycenter_start, barycenter_end : int
        Start and end indices for the barycenter segment
    weights : array-like or None
        Weights for each sequence
    metric_params : dict or None
        DTW parameters
    tie_strategy : str
        Strategy for handling ties in categorical voting
    verbose : bool
        Whether to print progress information

    Returns
    -------
    aligned_values : list
        List of aligned barycenter values for each sequence
    aligned_categories : list
        List of aligned category values for each sequence
    dtw_paths : list
        List of DTW paths for each alignment
    cost : float
        Total alignment cost
    """
    # Extract segments
    X_segment = np.array([X[start_idx:end_idx]])
    Y_segment = np.array([Y[start_idx:end_idx]])
    barycenter_segment = barycenter[barycenter_start:barycenter_end]

    # Convert to time series dataset
    X_segment = to_time_series_dataset(X_segment)
    weights = _set_weights(weights, X_segment.shape[0])

    # Single iteration of alignment
    list_p_k, list_y_k, cost, dtw_paths = _mm_assignment(
        X_segment, Y_segment, barycenter_segment, weights, metric_params
    )

    # Create aligned values and categories
    aligned_values = []
    aligned_categories = []

    for path in dtw_paths:
        # Extract barycenter indices and corresponding values
        barycenter_indices = [p[0] for p in path]
        sequence_indices = [p[1] for p in path]

        # Get aligned barycenter values
        aligned_value = barycenter_segment[barycenter_indices]
        aligned_values.append(aligned_value)

        # Get aligned category values
        y_seq = (
            Y_segment[0, sequence_indices]
            if isinstance(Y_segment, np.ndarray)
            else [Y_segment[0, j] for j in sequence_indices]
        )
        aligned_categories.append(y_seq)

    return aligned_values, aligned_categories, dtw_paths, cost
