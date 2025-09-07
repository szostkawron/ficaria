import numpy as np
import pandas as pd

# Ta funkcja dzieli zbiory danych na kompletne i niekompletne
# Uwaga - z tego co widziałam potrzebne w wielu miejscach!
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


def fuzzy_c_means(X: np.ndarray, n_clusters: int, v: float = 2.0, max_iter: int = 100, tol: float = 1e-5):
    """
    Fuzzy C-Means clustering algorithm.
    
    Parameters:
        X (np.ndarray): data matrix (n_samples x n_features)
        n_clusters (int): number of clusters
        m (float): fuzziness parameter (>1)
        max_iter (int): maximum number of iterations
        tol (float): convergence tolerance
    
    Returns:
        centers (np.ndarray): cluster centers
        u (np.ndarray): membership matrix (n_samples x n_clusters)
    """
    n_samples, n_features = X.shape

    u = np.random.rand(n_samples, n_clusters)
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

