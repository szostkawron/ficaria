from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List

from .utils import *


class FCMCentroidImputer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=3, v=2.0, max_iter=100, tol=1e-5):
        """
        Fuzzy C-Means centroid-based imputer.

        Each missing value is imputed using the value of the nearest cluster centroid,
        based on the Euclidean distance between the incomplete object and cluster centroids.
        
        Parameters:
            n_clusters (int): number of clusters
            m (float): fuzziness parameter
            max_iter (int): maximum number of FCM iterations
            tol (float): convergence tolerance
        """
        self.n_clusters = n_clusters
        self.v = v
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y=None):
        """
        Fit the FCM imputer on complete data only.
        """
        X = check_input_dataset(X)
        complete, _ = split_complete_incomplete(X)
        self.centers_, self.memberships_ = fuzzy_c_means(
            complete.to_numpy(),
            n_clusters=self.n_clusters,
            v=self.v,
            max_iter=self.max_iter,
            tol=self.tol
        )
        return self

    def transform(self, X):
        """
        Impute missing values using nearest cluster centroid.
        """
        X = check_input_dataset(X)
        _, incomplete = split_complete_incomplete(X)

        if incomplete.empty:
            return X

        X_imputed = X.copy()

        for idx, row in incomplete.iterrows():
            distances = [euclidean_distance(row.values, center) for center in self.centers_]
            nearest_idx = np.argmin(distances)
            nearest_center = self.centers_[nearest_idx]

            missing_cols = row[row.isna()].index
            for col in missing_cols:
                X_imputed.at[idx, col] = nearest_center[X.columns.get_loc(col)]

        return X_imputed


class FCMParameterImputer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=3, v=2.0, max_iter=150, tol=1e-5, random_state=None):
        """
        Fuzzy C-Means Parameter-based Imputation.
        
        Each missing value is imputed as a weighted sum of all cluster centroids,
        where weights are given by the membership values of the object.

        Parameters:
            n_clusters (int): number of clusters
            m (float): fuzziness parameter
            max_iter (int): maximum number of FCM iterations
            tol (float): convergence tolerance
        """
        self.n_clusters = n_clusters
        self.v = v
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y=None):
        """
        Fit the FCM imputer on complete data only.
        """
        X = check_input_dataset(X)
        complete, _ = split_complete_incomplete(X)
        self.centers_, self.memberships_ = fuzzy_c_means(
            complete.to_numpy(),
            n_clusters=self.n_clusters,
            v=self.v,
            max_iter=self.max_iter,
            tol=self.tol
        )

        self.feature_names_in_ = complete.columns
        return self

    def transform(self, X):
        """
        Impute missing values using parameter-based FCM method.
        Each missing value is the weighted sum of all centroids
        based on membership values.
        """
        X = check_input_dataset(X).copy()
        _, incomplete = split_complete_incomplete(X)

        if incomplete.empty:
            return X

        X_imputed = X.copy()

        for idx, row in incomplete.iterrows():
            obs = row.to_numpy()

            dist = np.array([euclidean_distance(obs, center) for center in self.centers_])
            dist = np.fmax(dist, 1e-10)

            u = 1 / np.sum((dist[:, None] / dist[None, :]) ** (2 / (self.v - 1)), axis=1)

            missing_cols = row[row.isna()].index
            for col in missing_cols:
                X_imputed.at[idx, col] = np.sum(u * self.centers_[:, X.columns.get_loc(col)])

        return X_imputed


class FCMRoughParameterImputer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=3, v=2.0, max_iter=100, tol=1e-5, lower_threshold=0.5):
        """
        Fuzzy C-Means Rough Parameter-based imputer.
        
        Each missing value is imputed using information from the 
        lower or upper approximation of the nearest fuzzy cluster.
        """
        self.n_clusters = n_clusters
        self.v = v
        self.max_iter = max_iter
        self.tol = tol
        self.lower_threshold = lower_threshold

    def fit(self, X, y=None):
        """
        Fit the imputer on complete data.
        """
        X = check_input_dataset(X)
        complete, _ = split_complete_incomplete(X)
        complete_array = complete.to_numpy()

        self.centers_, self.memberships_ = fuzzy_c_means(
            complete_array,
            n_clusters=self.n_clusters,
            v=self.v,
            max_iter=self.max_iter,
            tol=self.tol
        )

        self.lower_upper_ = []
        for k in range(self.n_clusters):
            cluster_memberships = self.memberships_[:, k]
            cluster_data = (complete_array, cluster_memberships)
            lower, upper = compute_lower_upper_approximation(
                cluster_data, threshold=self.lower_threshold
            )
            self.lower_upper_.append((lower, upper))

        return self

    def transform(self, X):
        """
        Impute missing values using rough parameter-based FCM method.
        """
        X = check_input_dataset(X)
        _, incomplete = split_complete_incomplete(X)

        if incomplete.empty:
            return X

        X_imputed = X.copy()

        for idx, row in incomplete.iterrows():
            obs = row.to_numpy()

            distances = np.array([euclidean_distance(obs, center) for center in self.centers_])
            nearest_idx = np.argmin(distances)

            lower, upper = self.lower_upper_[nearest_idx]
            approx_type = find_nearest_approximation(obs, lower, upper)

            if approx_type == 'lower' and lower.shape[0] > 0:
                approx_data = lower
            elif approx_type == 'upper' and upper.shape[0] > 0:
                approx_data = upper
            else:
                approx_data = np.array([self.centers_[nearest_idx]])

            missing_cols = row[row.isna()].index
            for col in missing_cols:
                col_idx = X.columns.get_loc(col)
                X_imputed.at[idx, col] = np.mean(approx_data[:, col_idx])

        return X_imputed


class KIImputer(BaseEstimator, TransformerMixin):
    """
    KIImputer: Hybrid KNN + Iterative Imputer for Missing Data.

    Implements the KI imputation method, which combines k-nearest neighbors (KNN)
    and iterative imputation to estimate missing values. For each incomplete row,
    the best number of neighbors (k) is selected by minimizing reconstruction error,
    and the imputation is refined using a model-based iterative approach.
    """

    def __init__(self, random_state=None):
        if random_state is not None and not isinstance(random_state, int):
            raise TypeError('Invalid random_state: Expected an integer or None.')
        self.random_state = random_state
        pass

    def fit(self, X, y=None):
        X = check_input_dataset(X)
        self.X_train_ = X.copy()
        self.np_rng_ = np.random.RandomState(self.random_state)
        return self

    def transform(self, X):
        X = check_input_dataset(X)
        X_imputed = impute_KI(X, self.X_train_, np_rng=self.np_rng_)
        return X_imputed


class FCMKIterativeImputer(BaseEstimator, TransformerMixin):
    """
   Hybrid imputer combining fuzzy c-means clustering, k-nearest neighbors, and iterative imputation.

   FCKI improves missing data imputation by first clustering data with fuzzy c-means,
   allowing points to belong to multiple clusters. It then performs KNN-based imputation
   within clusters, followed by iterative imputation for refinement. This two-level
   similarity search enhances accuracy compared to standard KNN imputation.
    """

    def __init__(self, random_state=None, max_clusters=10, m=2):
        if random_state is not None and not isinstance(random_state, int):
            raise TypeError('Invalid random_state: Expected an integer or None.')

        if not isinstance(max_clusters, int) or max_clusters <= 1:
            raise TypeError('Invalid max_clusters: Expected an integer greater than 1.')

        if not isinstance(m, (int, float)) or m <= 1:
            raise TypeError('Invalid m value: Expected a numeric value greater than 1.')

        self.random_state = random_state
        self.max_clusters = max_clusters
        self.m = m
        pass

    def fit(self, X, y=None):
        X = check_input_dataset(X, require_numeric=True)
        self.X_train_ = X.copy()

        self.imputer_ = SimpleImputer(strategy="mean")
        X_filled = self.imputer_.fit_transform(self.X_train_)
        X_filled = pd.DataFrame(data=X_filled, columns=X.columns, index=X.index)

        self.optimal_c_ = find_optimal_clusters_fuzzy(X_filled, min_clusters=1, max_clusters=self.max_clusters,
                                                      m=self.m,
                                                      random_state=self.random_state)
        self.np_rng_ = np.random.RandomState(self.random_state)
        np.random.seed(self.random_state)

        self.centers_, self.u_ = fuzzy_c_means(
            X_filled.values,
            n_clusters=self.optimal_c_,
            v=self.m,
            random_state=self.random_state,
        )

        return self

    def transform(self, X):
        X = check_input_dataset(X, require_numeric=True)

        X_imputed = impute_FCKI(X, self.X_train_, self.centers_, self.u_, self.optimal_c_, self.imputer_, self.m,
                                self.np_rng_, self.random_state)
        return X_imputed


class LinearInterpolationBasedIterativeIntuitionisticFuzzyCMeans(BaseEstimator, TransformerMixin):

    def __init__(self,n_clusters: int = 3,m: float = 2.0,alpha: float = 2.0,max_iter: int = 100,tol: float = 1e-5,max_outer_iter: int = 20,stop_criteria: float = 0.01,sigma: bool = False,random_state: Optional[int] = None,):
        
        """
        Initialize Linear Interpolation Based Iterative Intuitionistic Fuzzy C-Means (LI-IIFCM).
        Parameters
        ----------
        n_clusters : int, default=3
            Number of fuzzy clusters.
        m : float, default=2.0
            Fuzzification parameter controlling the level of fuzziness.
        alpha : float, default=2.0
            Parameter influencing hesitation degree.
        max_iter : int, default=100
            Maximum number of iterations for the internal IIFCM optimization loop.
        tol : float, default=1e-5
            Tolerance value for convergence during IIFCM iterations.
        max_outer_iter : int, default=20
            Maximum number of iterations for the imputation process.
        stop_criteria : float, default=0.01
            Threshold for average relative change in missing values used to stop iteration early.
        sigma : bool, default=False
            If True, applies weighted IFCM-σ distance metric instead of standard Euclidean distance.
        random_state int, Optional
            Controls randomness.
        """

        if not isinstance(n_clusters, int) or n_clusters < 2:
            raise TypeError("Invalid n_clusters: Expected an integer greater than 1.")
        if not isinstance(m, (int, float)) or m <= 1:
            raise TypeError("Invalid m value: Expected a numeric value greater than 1.")
        if not isinstance(alpha, (int, float)) or alpha <= 0:
            raise TypeError("Invalid alpha value: Expected a positive number.")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise TypeError("Invalid max_iter: Expected a positive integer.")
        if not isinstance(tol, (int, float)) or tol <= 0:
            raise TypeError("Invalid tol: Expected a positive float.")
        if not isinstance(max_outer_iter, int) or max_outer_iter <= 0:
            raise TypeError("Invalid max_outer_iter: Expected a positive integer.")
        if not isinstance(stop_criteria, (int, float)) or stop_criteria <= 0:
            raise TypeError("Invalid stop_criteria: Expected a positive float.")
        if not isinstance(sigma, bool):
            raise TypeError("Invalid sigma: Expected a boolean value.")
        if random_state is not None and not isinstance(random_state, int):
            raise TypeError("Invalid random_state: Expected an integer or None.")

        self.n_clusters = n_clusters
        self.m = m
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.max_outer_iter = max_outer_iter
        self.stop_criteria = stop_criteria
        self.sigma = sigma
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "LinearInterpolationBasedIterativeIntuitionisticFuzzyCMeans":
        """
        Fit the LI-IIFCM model on input data.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataset with missing values.
        y : None, optional
            Present for compatibility with sklearn interface.

        Returns
        -------
        self : object
            Fitted instance.
        """

        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if X.empty:
            raise ValueError("Input DataFrame is empty")
        if not all(np.issubdtype(dt, np.number) for dt in X.dtypes):
            raise TypeError("All columns must be numeric")

        X = check_input_dataset(X)
        self.columns_ = X.columns
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using Linear Interpolation and Iterative Intuitionistic Fuzzy C-Means.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataset with missing values.

        Returns
        -------
        pd.DataFrame
            Dataset with missing values filled using LI-IIFCM algorithm.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if X.empty:
            raise ValueError("Input DataFrame is empty")
        if not hasattr(self, "columns_"):
            raise AttributeError("You must call fit before transform")
        if list(X.columns) != list(self.columns_):
            raise ValueError("Columns of input DataFrame differ from those used in fit")



        X = check_input_dataset(X)
        missing_mask = X.isnull().reset_index(drop=True)
        X_filled = X.interpolate(method='linear', limit_direction='both').reset_index(drop=True)

        for _ in range(1, self.max_outer_iter + 1):
            U_star, V_star, _ = self._ifcm(X_filled)

            X_new = X_filled.copy()
            for i in range(X_filled.shape[0]):
                for j in range(X_filled.shape[1]):
                    if missing_mask.iloc[i, j]:
                        X_new.iloc[i, j] = np.sum(U_star[i, :] * V_star[:, j])

            X_new = X_new.clip(0, 1)

            diff = np.abs(X_new - X_filled)
            rel_diff = diff / (np.abs(X_filled) + 1e-10)
            if missing_mask.values.any():
                AvgV = rel_diff[missing_mask].mean().mean()
            else:
                AvgV = 0

            if AvgV <= self.stop_criteria:
                X_filled = X_new
                break

            X_filled = X_new.copy()

        return X_filled

    def _ifcm(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Method implementing Intuitionistic Fuzzy C-Means clustering.
        Optionally applies weighted distance metric (IFCM-σ) if sigma=True.

        Parameters
        ----------
        data : np.ndarray or pd.DataFrame
            Numeric data without missing values.

        Returns
        -------
        U_star : np.ndarray
            Adjusted membership matrix after applying intuitionistic correction.
        V_star : np.ndarray
            Cluster prototype (centroid) matrix after optimization.
        J_history : list of float
            Objective function values for each iteration of the clustering process.
        """
        data = data.to_numpy()
        N, F = data.shape

        rng = np.random.RandomState(self.random_state)
        U = rng.rand(N, self.n_clusters)
        U = U / np.sum(U, axis=1, keepdims=True)
        J_history = []

        for _ in range(self.max_iter):
            eta = 1 - U - (1 - U ** self.alpha) ** (1 / self.alpha)
            U_star = U + eta

            V_star = np.zeros((self.n_clusters, F))
            for j in range(self.n_clusters):
                numerator = np.sum(U_star[:, j][:, None] * data, axis=0)
                denominator = np.sum(U_star[:, j])
                V_star[j] = numerator / (denominator + 1e-10)

            dist = np.zeros((N, self.n_clusters))
            for i in range(N):
                for j in range(self.n_clusters):
                    diff = np.linalg.norm(data[i] - V_star[j])
                    if self.sigma:
                        numerator = np.sum((U_star[:, j][:, None] ** self.m) * (data - V_star[j]) ** 2, axis=0)
                        denominator = np.sum(U_star[:, j] ** self.m)
                        sigma_j = np.sqrt(np.sum(numerator) / (denominator + 1e-10))
                        diff = diff / (sigma_j + 1e-10)
                    dist[i, j] = diff
            dist = np.fmax(dist, 1e-10)

            new_U = np.zeros_like(U)
            for i in range(N):
                for j in range(self.n_clusters):
                    denom = np.sum((dist[i, j] / dist[i, :]) ** (2 / (self.m - 1)))
                    new_U[i, j] = 1.0 / denom if denom != 0 else 0

            J1 = np.sum((U_star ** self.m) * (dist ** 2))
            J2 = np.sum(np.mean(eta, axis=0) * np.exp(1 - np.mean(eta, axis=0)))
            J_ifcm = J1 + J2
            J_history.append(J_ifcm)

            if np.linalg.norm(new_U - U) < self.tol:
                break

            U = new_U.copy()

        return U_star, V_star, J_history

