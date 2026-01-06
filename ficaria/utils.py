import numpy as np
import pandas as pd
from kneed import KneeLocator
from scipy.spatial.distance import cdist


def split_complete_incomplete(X):
    """
    Split the dataset into complete (no missing values) and incomplete (with missing values) objects.

    Parameters
    ----------
    X : pd.DataFrame
        Input data.

    Returns
    -------
    complete : pd.DataFrame
        Rows without missing values.
    incomplete : pd.DataFrame
        Rows containing missing values.
    """
    complete = X.dropna()
    incomplete = X[X.isna().any(axis=1)]
    return complete, incomplete


def check_input_dataset(X, require_numeric=False, allow_nan=True, require_complete_rows=False,
                        no_nan_columns=False):
    """
    Convert input to DataFrame and check the validity of the dataset.

    Parameters
    ----------
    X : pd.DataFrame
        Input data.
    require_numeric : bool
        If True, ensure that only numeric columns are present.
    allow_nan : bool
        If True, allow NaN values in the dataset.
    require_complete_rows : bool
        If True, check that complete rows are present.
    no_nan_columns : bool
        If True, ensure there are no columns with NaN values.

    Returns
    -------
    X : pd.DataFrame
        Converted and validated DataFrame.
    """

    try:
        arr = np.asarray(X)
    except Exception:
        raise TypeError(f"X must be array-like or DataFrame, got {type(X).__name__!r} instead")

    if arr.ndim != 2:
        raise ValueError(f"X must be a 2D array-like structure, got {arr.ndim}D input instead")

    X = pd.DataFrame(X)

    if X.empty:
        raise ValueError("X must contain at least one sample, got an empty dataset instead")

    if require_numeric and not all(pd.api.types.is_numeric_dtype(dt) for dt in X.dtypes):
        non_numeric_cols = X.select_dtypes(exclude=np.number).columns.tolist()
        raise TypeError(f"X must be numeric, got non-numeric columns: {non_numeric_cols} instead")

    if not allow_nan and X.isnull().values.any():
        raise ValueError("X must not contain missing values")

    complete_rows = X.dropna(how="any")
    if require_complete_rows and complete_rows.empty:
        raise ValueError(
            "X must contain at least one row with no missing values")

    if no_nan_columns and X.isna().all().any():
        cols_all_nan = X.isna().all()
        raise ValueError(f"X must not contain columns with all NaNs, got {cols_all_nan.sum()} such columns instead")

    return X


def validate_params(params):
    """
    Validate parameters.

    Parameters
    ----------
    params : dict
        Dictionary of parameter names and values to validate.

    Raises
    ------
    TypeError, ValueError
        If any parameter is invalid.
    """
    if 'max_clusters' in params:
        max_clusters = params['max_clusters']
        if not isinstance(max_clusters, int):
            raise TypeError(f"max_clusters must be int, got {type(max_clusters).__name__} instead")
        if max_clusters < 1:
            raise ValueError(f"max_clusters must be >= 1, got {max_clusters} instead")

    if 'max_iter' in params:
        max_iter = params['max_iter']
        if not isinstance(max_iter, int):
            raise TypeError(f"max_iter must be int, got {type(max_iter).__name__} instead")
        if max_iter <= 1:
            raise ValueError(f"max_iter must be > 1, got {max_iter} instead")

    if 'max_iter_rough_k' in params:
        max_iter_rough_k = params['max_iter_rough_k']
        if not isinstance(max_iter_rough_k, int):
            raise TypeError(f"max_iter_rough_k must be int, got {type(max_iter_rough_k).__name__} instead")
        if max_iter_rough_k <= 1:
            raise ValueError(f"max_iter_rough_k must be > 1, got {max_iter_rough_k} instead")

    if 'max_FCM_iter' in params:
        max_FCM_iter = params['max_FCM_iter']
        if not isinstance(max_FCM_iter, int):
            raise TypeError(f"max_FCM_iter must be int, got {type(max_FCM_iter).__name__} instead")
        if max_FCM_iter <= 1:
            raise ValueError(f"max_FCM_iter must be > 1, got {max_FCM_iter} instead")

    if 'max_II_iter' in params:
        max_II_iter = params['max_II_iter']
        if not isinstance(max_II_iter, int):
            raise TypeError(f"max_II_iter must be int, got {type(max_II_iter).__name__} instead")
        if max_II_iter <= 1:
            raise ValueError(f"max_II_iter must be > 1, got {max_II_iter} instead")

    if 'max_k' in params:
        max_k = params['max_k']
        if not isinstance(max_k, int):
            raise TypeError(f"max_k must be int, got {type(max_k).__name__} instead")
        if max_k < 1:
            raise ValueError(f"max_k must be >= 1, got {max_k} instead")

    if 'random_state' in params:
        rs = params['random_state']
        if rs is not None and not isinstance(rs, int):
            raise TypeError(f"random_state must be int or None, got {type(rs).__name__} instead")

    if 'm' in params:
        m = params['m']
        if not isinstance(m, (int, float)):
            raise TypeError(f"m must be int or float, got {type(m).__name__} instead")
        if m <= 1.0:
            raise ValueError(f"m must be > 1.0, got {m} instead")

    if 'tol' in params:
        tol = params['tol']
        if not isinstance(tol, (int, float)):
            raise TypeError(f"tol must be int or float, got {type(tol).__name__} instead")
        if tol <= 0:
            raise ValueError(f"tol must be > 0, got {tol} instead")

    if 'wl' in params:
        wl = params['wl']
        if not isinstance(wl, (int, float)):
            raise TypeError(f"wl must be int or float, got {type(wl).__name__} instead")
        if wl <= 0 or wl > 1:
            raise ValueError(f"wl must be in range (0, 1], got {wl} instead")

    if 'wb' in params:
        wb = params['wb']
        if not isinstance(wb, (int, float)):
            raise TypeError(f"wb must be int or float, got {type(wb).__name__} instead")
        if wb < 0 or wb > 1:
            raise ValueError(f"wb must be in range [0, 1], got {wb} instead")

    if 'tau' in params:
        tau = params['tau']
        if not isinstance(tau, (int, float)):
            raise TypeError(f"tau must be int or float, got {type(tau).__name__} instead")
        if tau < 0:
            raise ValueError(f"tau must be >= 0, got {tau} instead")

    if 'k' in params:
        k = params['k']
        if not isinstance(k, int):
            raise TypeError(f"k must be int, got {type(k).__name__} instead")
        if k <= 1:
            raise ValueError(f"k must be > 1, got {k} instead")

    if 'n_features' in params:
        n_features = params['n_features']
        if not isinstance(n_features, int):
            raise TypeError(f"n_features must be int, got {type(n_features).__name__} instead")
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features} instead")

    if 'max_features' in params:
        max_features = params['max_features']
        if not isinstance(max_features, int):
            raise TypeError(f"max_features must be int, got {type(max_features).__name__} instead")
        if max_features < 1:
            raise ValueError(f"max_features must be >= 1, got {max_features} instead")

    if 'stop_threshold' in params:
        stop_threshold = params['stop_threshold']
        if not isinstance(stop_threshold, (int, float)):
            raise TypeError(f"stop_threshold must be int or float, got {type(stop_threshold).__name__} instead")
        if stop_threshold < 0:
            raise ValueError(f"stop_threshold must be >= 0, got {stop_threshold} instead")

    if 'min_samples_leaf' in params:
        min_samples_leaf = params['min_samples_leaf']
        if not isinstance(min_samples_leaf, (int, float)):
            raise TypeError(f"min_samples_leaf must be int or float, got {type(min_samples_leaf).__name__} instead")
        if min_samples_leaf <= 0:
            raise ValueError(f"min_samples_leaf must be > 0, got {min_samples_leaf} instead")

    if 'learning_rate' in params:
        learning_rate = params['learning_rate']
        if not isinstance(learning_rate, (int, float)):
            raise TypeError(f"learning_rate must be int or float, got {type(learning_rate).__name__} instead")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate} instead")

    if 'eps' in params:
        eps = params['eps']
        if not isinstance(eps, (int, float)):
            raise TypeError(f"eps must be int or float, got {type(eps).__name__} instead")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps} instead")

    if 'n_jobs' in params:
        n_jobs = params['n_jobs']
        if not isinstance(n_jobs, int):
            raise TypeError(f"n_jobs must be int, got {type(n_jobs).__name__} instead")
        if n_jobs == 0:
            raise ValueError("n_jobs must not be 0; use -1 for all cores or a non-zero integer")


def euclidean_distance(a, b):
    """
    Compute Euclidean distance between two vectors, ignoring NaNs.

    Parameters
    ----------
    a : np.ndarray
        First input vector.
    b : np.ndarray
        Second input vector.

    Returns
    -------
    distance : float
        Euclidean distance between vectors a and b.
    """
    mask = ~np.isnan(a) & ~np.isnan(b)
    return np.linalg.norm(a[mask] - b[mask])


def fuzzy_c_means(X, n_clusters, m=2.0, max_iter=100, tol=1e-5, random_state=None):
    """
    Fuzzy C-Means clustering algorithm.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input data matrix.
    n_clusters : int
        Number of clusters to form.
    v : float
        Fuzziness parameter, must be greater than 1.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance.
    random_state : int
        Seed for random number generator.

    Returns
    -------
    centers : np.ndarray of shape (n_clusters, n_features)
        Computed cluster centers.
    u : np.ndarray of shape (n_samples, n_clusters)
        Membership matrix for each sample.
    """
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    n_samples, n_features = X.shape

    rng = np.random.default_rng(random_state)
    u = rng.random((n_samples, n_clusters))
    u = u / np.sum(u, axis=1, keepdims=True)

    for iteration in range(max_iter):
        u_old = u.copy()

        uv = u ** m
        centers = (uv.T @ X) / np.sum(uv.T, axis=1)[:, None]

        dist = np.zeros((n_samples, n_clusters))
        for j in range(n_clusters):
            dist[:, j] = np.linalg.norm(X - centers[j], axis=1)
        dist = np.fmax(dist, 1e-10)

        u = 1 / np.sum((dist[:, :, None] / dist[:, None, :]) ** (2 / (m - 1)), axis=2)

        if np.linalg.norm(u - u_old) < tol:
            break

    return centers, u


def fcm_predict(X_new, centers, m=2.0):
    """
    Compute fuzzy membership matrix for new data points given cluster centers.

    Parameters
    ----------
    X_new : np.ndarray of shape (n_samples, n_features)
        New data points to classify.
    centers : np.ndarray of shape (n_clusters, n_features)
        Cluster centers obtained from Fuzzy C-Means.
    m : float
        Fuzziness parameter (>1), typically the same as used during training.

    Returns
    -------
    u_new : np.ndarray of shape (n_samples, n_clusters)
        Membership matrix for new samples, where each row sums to 1.
    """
    n_samples = X_new.shape[0]
    n_clusters = centers.shape[0]
    dist = np.zeros((n_samples, n_clusters))
    for j in range(n_clusters):
        dist[:, j] = np.linalg.norm(X_new - centers[j], axis=1)
    dist = np.fmax(dist, 1e-10)

    u_new = 1 / np.sum((dist[:, :, None] / dist[:, None, :]) ** (2 / (m - 1)), axis=2)
    return u_new


def compute_fcm_objective(X, centers, u, m=2):
    """
    Compute the fuzzy C-Means objective function value.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Data points.
    centers : np.ndarray of shape (n_clusters, n_features)
        Cluster centers.
    u : np.ndarray of shape (n_samples, n_clusters)
        Membership matrix.
    m : float, default=2
        Fuzziness parameter.

    Returns
    -------
    objective_value : float
        Value of the fuzzy C-Means objective function.
    """
    dist_sq = cdist(X, centers, metric='sqeuclidean')
    return np.sum((u ** m) * dist_sq)


def find_optimal_clusters_fuzzy(X, min_clusters=2, max_clusters=10, random_state=None, m=2, max_iter=100, tol=1e-5):
    """
    Elbow method for fuzzy C-Means with missing data imputation and objective function calculation.

    Parameters
    ----------
    X : pd.DataFrame
        Input data with missing values.
    min_clusters : int, default=2
        Minimum number of clusters to consider.
    max_clusters : int, default=10
        Maximum number of clusters to consider.
    random_state : int, optional
        Seed for reproducibility.
    m : float, default=2
        Fuzziness parameter for FCM.
    max_iter : int, default=100
        Maximum number of iterations in FCM.
    tol : float, default=1e-5
        Convergence tolerance for FCM.

    Returns
    -------
    optimal_clusters : int or None
        Optimal number of clusters determined by the elbow method.
    """

    objective_values = []
    k_values = list(range(min_clusters, max_clusters + 1))

    sample_size = min(len(X), 10000)
    X_sampled = X.sample(n=sample_size, random_state=random_state).to_numpy()

    for k in k_values:
        centers, u = fuzzy_c_means(X_sampled, n_clusters=k, m=m, random_state=random_state, max_iter=max_iter, tol=tol)

        obj = compute_fcm_objective(X_sampled, centers, u, m)
        objective_values.append(obj)

    kl = KneeLocator(k_values, objective_values, curve="convex", direction="decreasing")
    optimal_k = kl.elbow

    if optimal_k is None:
        return int((max_clusters + min_clusters) // 2)
    return int(optimal_k)
