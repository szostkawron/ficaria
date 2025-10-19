import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from typing import Optional


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


def check_input_dataset(X, require_numeric=False, allow_nan=True):
    """
    Convert input to DataFrame and check the validity of the dataset

    Parameters:
        X (pd.DataFrame): input data
        require_numeric (bool): check if only numeric columns are present
        allow_nan (bool): allow nan values

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

    if require_numeric and not all(pd.api.types.is_numeric_dtype(dt) for dt in X.dtypes):
        raise TypeError("Invalid input: Input dataset contains not numeric values")

    if not allow_nan and X.isnull().values.any():
        raise ValueError("Invalid input: Input dataset contains missing values")

    return X


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


def fuzzy_c_means(X: np.ndarray, n_clusters: int, v: float = 2.0, max_iter: int = 100, tol: float = 1e-5,
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
    n_samples, n_features = X.shape

    rng = np.random.default_rng(random_state)
    u = rng.random((n_samples, n_clusters))
    u = u / np.sum(u, axis=1, keepdims=True)

    for iteration in range(max_iter):
        u_old = u.copy()

        uv = u ** v
        centers = (uv.T @ X) / np.sum(uv.T, axis=1)[:, None]

        dist = np.zeros((n_samples, n_clusters))
        for j in range(n_clusters):
            dist[:, j] = np.linalg.norm(X - centers[j], axis=1)
        dist = np.fmax(dist, 1e-10)

        u = 1 / np.sum((dist[:, :, None] / dist[:, None, :]) ** (2 / (v - 1)), axis=2)

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


def compute_lower_upper_approximation(cluster_data, threshold=0.5):
    """
    Compute lower and upper approximation of a cluster.
    Objects with membership >= threshold are in lower approximation,
    others (membership < threshold) are in upper approximation.
    
    Parameters:
        cluster_data (tuple): (data, memberships) for the cluster
        threshold (float): cutoff for lower approximation
    
    Returns:
        lower (np.ndarray): rows in lower approximation
        upper (np.ndarray): rows in upper approximation
    """
    X, memberships = cluster_data
    lower_mask = memberships >= threshold
    lower = X[lower_mask]
    upper = X[~lower_mask]
    return lower, upper


def find_nearest_approximation(obs, lower, upper):
    """
    Determine if the object belongs to lower or upper approximation
    based on distance to mean of each approximation.
    
    Parameters:
        obs (np.ndarray): incomplete object
        lower (np.ndarray)
        upper (np.ndarray)
        
    Returns:
        'lower' or 'upper'
    """
    if lower.shape[0] > 0:
        dist_lower = np.min([euclidean_distance(obs, row) for row in lower])
    else:
        dist_lower = np.inf

    if upper.shape[0] > 0:
        dist_upper = np.min([euclidean_distance(obs, row) for row in upper])
    else:
        dist_upper = np.inf

    if dist_lower <= dist_upper:
        return 'lower'
    else:
        return 'upper'


def get_neighbors(train: list[list[float]], test_row: list[float], k: int) -> list[list[float]]:
    """
    Returns the k closest rows in `train` to `test_row`
    using Euclidean distance (ignores NaNs).

    Parameters:
        train (list[list[float]]): Training data.
        test_row (list[float]): Query point.
        k (int): Number of neighbors to return.
    Returns:
        list: Closest k rows from `train`.
    """
    test = np.array(test_row)
    distances = list()
    for train_row in train:
        dist = np.sqrt(np.nansum((test - np.array(train_row)) ** 2))
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors


def find_best_k(St: pd.DataFrame, random_col: int, original_value: float, max_iter: int = 30) -> int:
    """
    Select the optimal number of neighbors (k) that minimizes RMSE
    when imputing a masked value in a selected column.

    Parameters:
        St (pd.DataFrame): Data with last row partially masked.
        random_col (int): Index of the masked column.
        original_value (float): True value before masking.
        max_iter (int): Maximum number of iterations (default is 30).

    Returns:
        int: Best value of k.
    """
    n = len(St)
    if n <= 1:
        return 1

    xi = St.iloc[-1].to_numpy()
    St_without_xi = St.iloc[:-1].to_numpy()

    distances = [euclidean_distance(xi, row) for row in St_without_xi]
    sorted_indices = np.argsort(distances)
    sorted_rows = St_without_xi[sorted_indices]

    max_k = min(n - 1, max_iter)
    k_values = range(1, max_k + 1)
    rmse_list = []

    for k in k_values:
        top_k_rows = sorted_rows[:k]
        col_values = top_k_rows[:, random_col]
        col_values = col_values[~np.isnan(col_values)]

        if len(col_values) > 0:
            mean_value = np.mean(col_values)
            rmse = np.sqrt((mean_value - original_value) ** 2)
        else:
            rmse = np.inf
        rmse_list.append(rmse)

    best_k = k_values[np.argmin(rmse_list)]
    return best_k


def impute_KI(X: pd.DataFrame, X_train: Optional[pd.DataFrame] = None, np_rng: Optional[np.random.RandomState] = None,
              random_state: int = 42, max_iter: int = 30) -> np.ndarray:
    """
    Impute missing values using the KI method (KNN + Iterative Imputation).

    Parameters:
        X (pd.DataFrame): Data to impute.
        X_train (pd.DataFrame): Optional reference data (default is None).
        np_rng (np.random.RandomState): Random generator for reproducibility (default is None).
        random_state (int): Random state for reproducibility (default is 42).
        max_iter (int): Maximum number of iterations (default is 30).

    Returns:
        np.ndarray: Imputed dataset.
    """
    if np_rng is None:
        np_rng = np.random.RandomState()
    X_incomplete_rows = X.copy()

    X_mis = X_incomplete_rows[X_incomplete_rows.isnull().any(axis=1)]

    if X_train is not None and not X.equals(X_train):
        X_train_safe = X_train.copy()
        X_train_safe.index = pd.RangeIndex(
            start=X.index.max() + 1 if len(X.index) > 0 else 0,
            stop=(X.index.max() + 1 if len(X.index) > 0 else 0) + len(X_train)
        )
        all_data = pd.concat([X, X_train_safe], axis=0)
    else:
        all_data = X.copy()

    imputed_rows = []

    for idx, xi in X_mis.iterrows():
        col_index_missing = []
        A_mis = []
        for j in range(len(X_incomplete_rows.columns)):
            if pd.isnull(xi.iloc[j]):
                col_index_missing.append(j)
                A_mis.append(X_incomplete_rows.columns[j])

        P = all_data.dropna(inplace=False, axis=0, subset=A_mis)
        if P.empty:
            raise ValueError(f"Invalid input: No rows with valid values found in columns: {A_mis}")
        P = pd.concat([P, pd.DataFrame([xi], index=[idx])], axis=0)

        St = P.copy()
        St_Complete_Temp = St.copy()
        if St_Complete_Temp.iloc[-1].isnull().all():
            raise ValueError("Invalid input: Data contains a row with only NaN values")

        A_r = np_rng.randint(0, St_Complete_Temp.shape[1])
        AV = St_Complete_Temp.iloc[len(St.index) - 1, A_r]
        while pd.isnull(AV):
            A_r = np_rng.randint(0, St_Complete_Temp.shape[1])
            AV = St_Complete_Temp.iloc[len(St.index) - 1, A_r]
        St.iloc[len(St.index) - 1, A_r] = np.NaN

        k = find_best_k(St, A_r, AV, max_iter)

        xi_from_Pt = P.iloc[-1].values.tolist()
        Pt_without_xi = P.iloc[:-1].values.tolist()

        neighbors_xi = get_neighbors(Pt_without_xi, xi_from_Pt, k)

        df_neighbors_xi = pd.DataFrame(data=neighbors_xi, columns=P.columns)

        S = pd.concat([df_neighbors_xi, pd.DataFrame([xi], index=[idx])], axis=0)

        S_filled_EM = IterativeImputer(random_state=random_state, max_iter=max_iter).fit_transform(S.values)

        S_filled_EM = pd.DataFrame(data=S_filled_EM, columns=P.columns)
        xi_imputed = S_filled_EM.iloc[len(S_filled_EM.index) - 1]
        imputed_rows.append((idx, xi_imputed))

        all_data.loc[idx] = xi_imputed
        X_incomplete_rows.loc[idx] = xi_imputed

    all_dataset_imputed = X_incomplete_rows.loc[X.index]
    return all_dataset_imputed.to_numpy()


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
                                random_state: Optional[int] = None, m: float = 2):
    """
    Elbow method for fuzzy C-means with missing data imputation and objective function calculation.

    Parameters:
        X (pd.DataFrame): Input data with missing values.
        min_clusters (int): Minimum number of clusters (default is 2).
        max_clusters (int): Maximum number of clusters (default is 10).
        random_state (int): Seed for reproducibility (default is None).
        m (float): Fuzziness parameter (default is 2).

    Returns:
        int or None: Optimal number of clusters found by the elbow method.
    """

    objective_values = []
    k_values = list(range(min_clusters, max_clusters + 1))

    sample_size = min(len(X), 10000)
    X_sampled = X.sample(n=sample_size, random_state=42)

    for k in k_values:
        np.random.seed(random_state)
        centers, u = fuzzy_c_means(X_sampled.values, n_clusters=k, v=m, random_state=random_state)

        obj = compute_fcm_objective(X_sampled.to_numpy(), centers, u, m)
        objective_values.append(obj)

    kl = KneeLocator(k_values, objective_values, curve="convex", direction="decreasing")
    optimal_k = kl.elbow

    if optimal_k is None:
        return int((max_clusters + min_clusters) // 2)
    return int(optimal_k)


def impute_FCKI(X: pd.DataFrame, X_train: pd.DataFrame, centers: np.ndarray, u_train: np.ndarray, c: int,
                imputer: SimpleImputer, m: float = 2, np_rng: Optional[np.random.RandomState] = None,
                random_state: int = 42, max_iter: int = 30) -> np.ndarray:
    """
    Impute missing values using the FCKI method (FCM + KNN + Iterative Imputation).

    Parameters:
        X (pd.DataFrame): Data to impute.
        X_train (pd.DataFrame or None): Optional reference data.
        centers (np.ndarray): Cluster centers obtained from fuzzy c-means (shape: [n_clusters, n_features]).
        u_train (np.ndarray): Membership matrix for the training data (shape: [n_samples_train, n_clusters]).
        c (int): Optimal number of clusters used in fuzzy c-means.
        imputer (SimpleImputer): A fitted simple imputer used for the initial rough imputation
        m (float): Fuzziness parameter used in fuzzy c-means (m > 1) (default is 2).
        np_rng (np.random.RandomState or None): Optional NumPy random generator for reproducibility (default is None).
        random_state (int): Random seed used for KNN-based imputation and reproducibility (default is 42).
        max_iter (int): Maximum number of iterations (default is 30).

    Returns:
        np.ndarray: Imputed dataset.
    """
    X_filled = imputer.transform(X)
    X_filled = pd.DataFrame(data=X_filled, columns=X.columns, index=X.index)
    membership_matrix = fcm_predict(X_filled.values, centers, m)
    fcm_labels_train = u_train.argmax(axis=1)
    fcm_labels_X = membership_matrix.argmax(axis=1)

    all_clusters = pd.DataFrame(columns=X.columns)

    for i in range(c):
        cluster_train_i = X_train[fcm_labels_train == i]
        cluster_X_i = X[fcm_labels_X == i]
        imputed_cluster_X_I = impute_KI(cluster_X_i, cluster_train_i, np_rng, random_state, max_iter)
        imputed_cluster_X_I = pd.DataFrame(imputed_cluster_X_I, columns=X.columns, index=cluster_X_i.index)
        if len(all_clusters) == 0:
            all_clusters = imputed_cluster_X_I
        else:
            all_clusters = pd.concat([all_clusters, imputed_cluster_X_I], axis=0)

    all_clusters = all_clusters.loc[~all_clusters.index.duplicated(keep='last')]

    all_clusters.sort_index(inplace=True)

    return all_clusters.to_numpy()
