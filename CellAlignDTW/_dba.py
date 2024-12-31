# Modified based on https://github.com/tslearn-team/tslearn/blob/5568c026db4b4380b99095827e0573a8f55a81f0/tslearn/barycenters/dba.py
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tslearn.barycenters.dba as dba
from tslearn.metrics import dtw_path
from tslearn.utils import to_time_series_dataset, ts_size
from tslearn.barycenters.utils import _set_weights

def _convert_categorical(Y):
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
    if np.issubdtype(Y.dtype, np.number):
        print("Y is already numeric categorical. No conversion needed.")
        return Y, None
    
    # Convert non-numeric categories to numeric
    label_encoder = LabelEncoder()
    Y_flat = Y.ravel()
    Y_numeric = label_encoder.fit_transform(Y_flat).reshape(Y.shape)
    
    # Print mapping
    print("Category mapping:")
    for i, category in enumerate(label_encoder.classes_):
        print(f"  {category} -> {i}")
    
    return Y_numeric, label_encoder

def _handle_majority_vote(values, weights=None, tie_strategy='first'):
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
    
    if not is_tie or tie_strategy == 'first':
        return tied_values[0], is_tie
    
    elif tie_strategy == 'random':
        return np.random.choice(tied_values), is_tie
    
    elif tie_strategy == 'weighted_random':
        tied_weights = weighted_counts[tied_indices]
        tied_weights = tied_weights / tied_weights.sum()  # Normalize weights
        return np.random.choice(tied_values, p=tied_weights), is_tie
    
    elif tie_strategy == 'report':
        return tied_values, is_tie
    
    else:
        raise ValueError(f"Unknown tie_strategy: {tie_strategy}")


def _mm_assignment(X, Y, barycenter, weights, metric_params=None):
    """Modified to also track Y alignments"""
    if metric_params is None:
        metric_params = {}
    n = X.shape[0]
    cost = 0.
    list_p_k = []
    list_y_k = []
    dtw_paths = []

    for i in range(n):
        path, dist_i = dtw_path(barycenter, X[i], **metric_params)
        cost += dist_i ** 2 * weights[i]
        list_p_k.append(path)
        dtw_paths.append(path)
        y_aligned = [(i, path_point[1]) for path_point in path]
        list_y_k.append(y_aligned)
    cost /= weights.sum()
    return list_p_k, list_y_k, cost, dtw_paths

def _mm_update_barycenter_with_categories(X, Y, diag_sum_v_k, list_w_k, list_y_k,
                                    weights, tie_strategy='first', verbose=False):
    """Update barycenter and determine majority categories"""
    d = X.shape[2]
    barycenter_size = diag_sum_v_k.shape[0]
    
    # Update numerical values (original logic)
    sum_w_x = np.zeros((barycenter_size, d))
    for k, (w_k, x_k) in enumerate(zip(list_w_k, X)):
        sum_w_x += w_k.dot(x_k[:ts_size(x_k)])
    barycenter = np.diag(1. / diag_sum_v_k).dot(sum_w_x)
    
    # Determine majority categories for each time point
    categories = np.zeros(barycenter_size, dtype=Y.dtype)
    for t in range(barycenter_size):
        # Collect all Y values aligned to this time point
        y_values = []
        y_weights = []
        for y_aligned in list_y_k:
            for idx, time_idx in y_aligned:
                if time_idx == t:
                    y_values.append(Y[idx, time_idx])
                    y_weights.append(weights[idx])
                    
        # Get majority vote if we have any alignments
        if y_values:
            category, is_tie = _handle_majority_vote(
                y_values, 
                weights=y_weights,
                tie_strategy=tie_strategy
            )
            
            if verbose and is_tie:
                print(f"Tie detected at time point {t}: {category}")
                
            # Handle multiple categories if tie_strategy is 'report'
            if isinstance(category, np.ndarray):
                categories[t] = category[0]  # Take first value if array
            else:
                categories[t] = category 
            
    return barycenter, categories

def dtw_barycenter_averaging_with_categories_one_init(X, Y, barycenter_size=None,
                                                     init_barycenter=None,
                                                     max_iter=30, tol=1e-5, 
                                                     weights=None,
                                                     metric_params=None,
                                                     tie_strategy='first',
                                                     verbose=False):
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
        list_p_k, list_y_k, cost, dtw_paths = _mm_assignment(X_, Y, barycenter, weights, 
                                                 metric_params)
        diag_sum_v_k, list_w_k = dba._mm_valence_warping(list_p_k, barycenter_size,
                                                     weights)
        if verbose:
            print("[DBA] epoch %d, cost: %.3f" % (it + 1, cost))
            
        # Update both barycenter and categories
        barycenter, categories = _mm_update_barycenter_with_categories(
            X_, Y, diag_sum_v_k, list_w_k, list_y_k, weights=weights, tie_strategy=tie_strategy, verbose=verbose
        )
        
        if abs(cost_prev - cost) < tol:
            final_dtw_paths = dtw_paths
            break
        elif cost_prev < cost:
            warnings.warn("DBA loss is increasing while it should not be. "
                        "Stopping optimization.", ConvergenceWarning)
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

def dtw_barycenter_averaging_with_categories(X, Y, barycenter_size=None, 
                                           init_barycenter=None, max_iter=30, 
                                           tol=1e-5, weights=None,
                                           metric_params=None, verbose=False, 
                                           n_init=1, tie_strategy='first'):
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
    Y_numeric, label_encoder = _convert_categorical(Y)
    
    best_cost = np.inf
    best_barycenter = None
    best_categories = None
    
    for i in range(n_init):
        if verbose:
            print("Attempt {}".format(i + 1))
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
            verbose=verbose
        )
        if loss < best_cost:
            best_cost = loss
            best_barycenter = bary
            best_categories = cats
            best_aligned_barycenters = aligned_bary
    
    if label_encoder is not None:
        best_categories = label_encoder.inverse_transform(best_categories)

    return best_barycenter, best_categories, best_aligned_barycenters
