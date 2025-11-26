from typing import Optional

import numpy as np
import pandas as pd
from gower import gower_matrix
from kneed import KneeLocator


def split_complete_incomplete(X: pd.DataFrame):
    """
    Split the dataset into complete (no missing values) and incomplete (with missing values) objects.
    
    Parameters:
        X (pd.DataFrame): input data
    
    Returns:
        complete (pd.DataFrame), incomplete (pd.DataFrame)
    """
    complete = X.dropna()
    incomplete = X[X.isna().any(axis=1)]
    return complete, incomplete


def check_input_dataset(X, require_numeric=False, allow_nan=True, require_complete_rows=False, no_nan_rows=False,
                        no_nan_columns=False):
    """
    Convert input to DataFrame and check the validity of the dataset

    Parameters:
        X (pd.DataFrame): input data
        require_numeric (bool): check if only numeric columns are present
        allow_nan (bool): allow nan values
        require_complete_rows (bool): check if complete rows are present
        no_nan_rows (bool): check if there are no nan rows are present
        no_nan_columns (bool): check if there are no nan columns are present

    Returns:
        pd.DataFrame: converted data
    """

    try:
        arr = np.asarray(X)
    except Exception:
        raise TypeError(
            "Invalid input: Expected a 2D structure such as a DataFrame, NumPy array, or similar tabular format")

    if arr.ndim != 2:
        raise ValueError("Invalid input: Expected a 2D structure")

    X = pd.DataFrame(X)

    if X.empty:
        raise ValueError("Invalid input: Input dataset is empty")

    complete_rows = X.dropna(how="any")
    if require_complete_rows and complete_rows.empty:
        raise ValueError("Invalid input: Input dataset contains no complete rows.")

    if require_numeric and not all(pd.api.types.is_numeric_dtype(dt) for dt in X.dtypes):
        raise TypeError("Invalid input: Input dataset contains not numeric values")

    if not allow_nan and X.isnull().values.any():
        raise ValueError("Invalid input: Input dataset contains missing values")

    if no_nan_rows:
        rows_all_nan = X.isnull().all(axis=1)
        if rows_all_nan.any():
            raise ValueError("Invalid input: Input dataset contains a row with only NaN values")

    if no_nan_columns and X.isna().all().any():
        raise ValueError("Invalid input: Input dataset contains a column with only NaN values")

    return X


def validate_params(params):
    """
    Validate parameters.

    Parameters:
        params (dict): Dictionary of parameter names and values

    Raises:
        TypeError, ValueError: If any parameter is invalid.
    """

    if 'n_clusters' in params:
        n_clusters = params['n_clusters']
        if not isinstance(n_clusters, int):
            raise TypeError(f"Invalid type for n_clusters: {type(n_clusters).__name__}. Must be int.")
        if n_clusters < 1:
            raise ValueError(f"Invalid value for n_clusters: {n_clusters}. Must be >= 1.")

    if 'max_iter' in params:
        max_iter = params['max_iter']
        if not isinstance(max_iter, int):
            raise TypeError(f"Invalid type for max_iter: {type(max_iter).__name__}. Must be int.")
        if max_iter <= 1:
            raise ValueError(f"Invalid value for max_iter: {max_iter}. Must be >= 1.")

    if 'random_state' in params:
        rs = params['random_state']
        if rs is not None and not isinstance(rs, int):
            raise TypeError(f"Invalid type for random_state: {type(rs).__name__}. Must be int or None.")

    if 'm' in params:
        m = params['m']
        if not isinstance(m, (int, float)):
            raise TypeError(f"Invalid type for m: {type(m).__name__}. Must be float.")
        if m <= 1.0:
            raise ValueError(f"Invalid value for m: {m}. Must be > 1.0.")

    if 'tol' in params:
        tol = params['tol']
        if not isinstance(tol, (int, float)):
            raise TypeError(f"Invalid type for tol: {type(tol).__name__}. Must be float.")
        if tol <= 0:
            raise ValueError(f"Invalid value for tol: {tol}. Must be > 0.")

    if 'wl' in params:
        wl = params['wl']
        if not isinstance(wl, (int, float)):
            raise TypeError(f"Invalid type for wl: {type(wl).__name__}. Must be int or float.")
        if wl <= 0 or wl > 1:
            raise ValueError(f"Invalid value for wl: {wl}. Must be in range (0, 1].")

    if 'wb' in params:
        wb = params['wb']
        if not isinstance(wb, (int, float)):
            raise TypeError(f"Invalid type for wb: {type(wb).__name__}. Must be int or float.")
        if wb < 0 or wb > 1:
            raise ValueError(f"Invalid value for wb: {wb}. Must be in range [0, 1].")

    if 'tau' in params:
        tau = params['tau']
        if not isinstance(tau, (int, float)):
            raise TypeError(f"Invalid type for tau: {type(tau).__name__}. Must be int or float.")
        if tau < 0:
            raise ValueError(f"Invalid value for tau: {tau}. Must be >= 0.")

    if 'n_feature' in params:
        k = params['n_feature']
        if not isinstance(k, int):
            raise TypeError(f"Invalid type for n_feature: {type(k).__name__}. Must be int.")
        if k <= 1:
            raise ValueError(f"Invalid value for n_feature: {k}. Must be > 1.")

    if 'alpha' in params:
        alpha = params['alpha']
        if not isinstance(alpha, (int, float)):
            raise TypeError(f"Invalid type for alpha: {type(alpha).__name__}. Must be int or float.")
        if not (0 < alpha <= 1):
            raise ValueError(f"Invalid value for alpha: {alpha}. Must be in range (0, 1].")

    if 'n_features' in params:
        n_features = params['n_features']
        if not isinstance(n_features, int):
            raise TypeError(f"Invalid type for n_features: {type(n_features).__name__}. Must be int.")
        if n_features < 1:
            raise ValueError(f"Invalid value for n_features: {n_features}. Must be > n_features.")


def euclidean_distance(a: np.ndarray, b: np.ndarray):
    """
    Compute Euclidean distance between two vectors, ignoring NaNs.
    
    Parameters:
        a, b (np.ndarray): input vectors
    
    Returns:
        float: Euclidean distance
    """
    mask = ~np.isnan(a) & ~np.isnan(b)
    return np.linalg.norm(a[mask] - b[mask])


def fuzzy_c_means(X: np.ndarray, n_clusters: int, m: float = 2.0, max_iter: int = 100, tol: float = 1e-5,
                  random_state=None):
    """
    Fuzzy C-Means clustering algorithm.

    Parameters:
        X (np.ndarray): data matrix (n_samples x n_features)
        n_clusters (int): number of clusters
        v (float): fuzziness parameter (>1)
        max_iter (int): maximum number of iterations
        tol (float): convergence tolerance
        random_state (int): random seed

    Returns:
        centers (np.ndarray): cluster centers
        u (np.ndarray): membership matrix (n_samples x n_clusters)
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


def rough_kmeans_from_fcm(X, memberships, center_init, wl=0.6, wb=0.4, tau=0.5, max_iter=100, tol=1e-4):
    """
    Rough K-Means
    Applied after FCM clustering (using its centroids as initialization).
    Each cluster is represented by a lower and an upper approximation, allowing
    samples in boundary regions to belong to multiple clusters when uncertainty exists.
    The algorithm starts from FCM centroids and iteratively updates cluster centers
    using weighted means of lower and boundary regions.

    Parameters:
        X (np.ndarray): data matrix (n_samples x n_features)
        memberships (np.ndarray): Membership matrix from FCM (n_samples, n_clusters)
        center_init (np.ndarray): Initial cluster centers (n_clusters x n_features) - output of FCM
        wl (float): weight for the lower approximation
        wb (float): weight for the boundary region
        tau (float): threshold controlling assignment of samples to lower or boundary regions
        max_iter (int): maximum number of iterations for updating cluster centers
        tol (float): Convergence tolerance; the algorithm stops if the shift in cluster centers is below this threshold.

    Returns:
        list of tuples: Each tuple represents one cluster and contains:
            - lower (np.ndarray): Samples in the lower approximation of the cluster.
            - upper (np.ndarray): Samples in the upper (boundary) approximation.
            - center (np.ndarray): Final cluster center vector.
    """

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    n_samples = X.shape[0]
    n_clusters = center_init.shape[0]
    centers = center_init.copy()

    lower_sets = [[] for _ in range(n_clusters)]
    upper_sets = [[] for _ in range(n_clusters)]

    init_labels = np.argmax(memberships, axis=1)
    for i, lbl in enumerate(init_labels):
        lower_sets[lbl].append(i)
        upper_sets[lbl].append(i)

    for iteration in range(max_iter):

        new_centers = np.zeros_like(centers)
        for k in range(n_clusters):
            lower_idx = lower_sets[k]
            upper_idx = upper_sets[k]
            boundary_idx = list(set(upper_idx) - set(lower_idx))

            if len(lower_idx) == 0:
                new_centers[k] = centers[k]
                continue

            lower_mean = np.mean(X[lower_idx], axis=0)

            if len(boundary_idx) > 0:
                boundary_mean = np.mean(X[boundary_idx], axis=0)
                new_centers[k] = wl * lower_mean + wb * boundary_mean
            else:
                new_centers[k] = lower_mean

        new_lower_sets = [[] for _ in range(n_clusters)]
        new_upper_sets = [[] for _ in range(n_clusters)]

        for i, x in enumerate(X):
            distances = np.array([euclidean_distance(x, c) for c in new_centers])
            h = np.argmin(distances)
            dmin = distances[h]

            new_upper_sets[h].append(i)

            for k in range(n_clusters):
                if k != h and (distances[k] - dmin) <= tau:
                    new_upper_sets[k].append(i)

            count_upper = sum([i in new_upper_sets[k] for k in range(n_clusters)])
            if count_upper == 1:
                new_lower_sets[h].append(i)

        shift = np.linalg.norm(new_centers - centers)

        if shift < tol:
            break

        centers = new_centers
        lower_sets = new_lower_sets
        upper_sets = new_upper_sets

    clusters = []
    for k in range(n_clusters):
        lower = X[lower_sets[k]] if len(lower_sets[k]) > 0 else np.array([])
        upper = X[upper_sets[k]] if len(upper_sets[k]) > 0 else np.array([])
        clusters.append((lower, upper, centers[k]))

    return clusters


def fuzzy_c_means_categorical(X: np.ndarray, n_clusters: int, m: float = 2.0, max_iter: int = 100, tol: float = 1e-5,
                              random_state=None):
    """
    Fuzzy C-Means clustering algorithm for data that contains categorical variables.

    Parameters:
        X (np.ndarray): data matrix (n_samples x n_features)
        n_clusters (int): number of clusters
        m (float): fuzziness parameter (>1)
        max_iter (int): maximum number of iterations
        tol (float): convergence tolerance
        random_state (int): random seed

    Returns:
        centers (np.ndarray): cluster centers
        u (np.ndarray): membership matrix (n_samples x n_clusters)
    """
    X = pd.DataFrame(X)

    n_samples, n_features = X.shape

    rng = np.random.default_rng(random_state)
    u = rng.random((n_samples, n_clusters))
    u = u / np.sum(u, axis=1, keepdims=True)

    is_numeric = X.apply(pd.api.types.is_numeric_dtype)

    for iteration in range(max_iter):
        u_old = u.copy()
        uv = u ** m
        centers = pd.DataFrame(index=range(n_clusters), columns=X.columns)

        for col_name in X.columns:
            col = X[col_name]
            if is_numeric[col_name]:
                for k in range(n_clusters):
                    centers.at[k, col_name] = np.sum(uv[:, k] * col.values) / np.sum(uv[:, k])
            else:
                values, counts = np.unique(col, return_counts=True)
                for k in range(n_clusters):
                    weights = np.array([np.sum(uv[col == val, k]) for val in values])

                    centers.at[k, col_name] = values[np.argmax(weights)]

        combined = pd.concat([X, centers], ignore_index=True)
        dist_matrix = gower_matrix(combined)
        dist = dist_matrix[:n_samples, n_samples:]

        dist = np.fmax(dist, 1e-10)

        u = 1 / np.sum((dist[:, :, None] / dist[:, None, :]) ** (2 / (m - 1)), axis=2)

        if np.linalg.norm(u - u_old) < tol:
            break

    return centers, u


def fcm_predict(X_new, centers, m=2.0):
    """
    Compute fuzzy membership matrix for new data points given cluster centers.

    Parameters:
        X_new (np.ndarray): New data to classify (n_samples x n_features).
        centers (np.ndarray): Cluster centers obtained from Fuzzy C-Means (n_clusters x n_features).
        m (float): Fuzziness parameter (>1), typically same as used in training.

    Returns:
        u_new (np.ndarray): Membership matrix for new samples (n_samples x n_clusters),
                            where each row sums to 1.
    """
    n_samples = X_new.shape[0]
    n_clusters = centers.shape[0]
    dist = np.zeros((n_samples, n_clusters))
    for j in range(n_clusters):
        dist[:, j] = np.linalg.norm(X_new - centers[j], axis=1)
    dist = np.fmax(dist, 1e-10)

    u_new = 1 / np.sum((dist[:, :, None] / dist[:, None, :]) ** (2 / (m - 1)), axis=2)
    return u_new


def compute_fcm_objective(X: np.ndarray, centers: np.ndarray, u: np.ndarray, m: float = 2):
    """
    Compute the fuzzy c-means objective function value.

    Parameters:
        X (np.array): Data points, shape (n_samples, n_features).
        centers (np.array): Cluster centers, shape (n_clusters, n_features).
        u (np.array): Membership matrix, shape (n_samples, n_clusters).
        m (float): Fuzziness parameter (default is 2).

    Returns:
        float: Value of the fuzzy c-means objective function.
    """
    centers = np.array(centers)

    dist_sq = np.zeros((X.shape[0], centers.shape[0]))
    for j in range(centers.shape[0]):
        diff = X - centers[j]
        dist_sq[:, j] = np.sum(diff ** 2, axis=1)

    obj = np.sum((u ** m) * dist_sq)
    return obj


def find_optimal_clusters_fuzzy(X: pd.DataFrame, min_clusters: int = 2, max_clusters: int = 10,
                                random_state: Optional[int] = None, m: float = 2, max_iter: int = 100,
                                tol: float = 1e-5):
    """
    Elbow method for fuzzy C-means with missing data imputation and objective function calculation.

    Parameters:
        X (pd.DataFrame): Input data with missing values.
        min_clusters (int): Minimum number of clusters (default is 2).
        max_clusters (int): Maximum number of clusters (default is 10).
        random_state (int): Seed for reproducibility (default is None).
        m (float): Fuzziness parameter (default is 2).
        max_iter (int): Maximum number of iterations in FCM (default is 100).
        tol (float): Tolerance for FCM (default is 1e-5).

    Returns:
        int or None: Optimal number of clusters found by the elbow method.
    """

    objective_values = []
    k_values = list(range(min_clusters, max_clusters + 1))

    sample_size = min(len(X), 10000)
    X_sampled = X.sample(n=sample_size, random_state=random_state)

    for k in k_values:
        centers, u = fuzzy_c_means(X_sampled.values, n_clusters=k, m=m, random_state=random_state, max_iter=max_iter,
                                   tol=tol)

        obj = compute_fcm_objective(X_sampled.to_numpy(), centers, u, m)
        objective_values.append(obj)

    kl = KneeLocator(k_values, objective_values, curve="convex", direction="decreasing")
    optimal_k = kl.elbow

    if optimal_k is None:
        return int((max_clusters + min_clusters) // 2)
    return int(optimal_k)
