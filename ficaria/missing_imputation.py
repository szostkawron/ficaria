from collections import defaultdict
from typing import Tuple, List

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

from .utils import *


# --------------------------------------
# FCMCentroidImputer
# --------------------------------------
class FCMCentroidImputer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=3, m=2.0, max_iter=100, tol=1e-5, random_state=None):
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
        validate_params({
            'n_clusters': n_clusters,
            'm': m,
            'max_iter': max_iter,
            'tol': tol,
            'random_state': random_state
        })

        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fit the FCM imputer on complete data only.
        """
        X = check_input_dataset(X, require_numeric=True, require_complete_rows=True)
        complete, _ = split_complete_incomplete(X)
        if self.n_clusters > len(complete):
            raise ValueError("n_clusters cannot be larger than the number of complete rows")

        self.centers_, self.memberships_ = fuzzy_c_means(
            complete.to_numpy(),
            n_clusters=self.n_clusters,
            m=self.m,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state
        )

        self.feature_names_in_ = list(X.columns)

        return self

    def transform(self, X):
        """
        Impute missing values using nearest cluster centroid.
        """

        if not hasattr(self, "centers_") or not hasattr(self, "memberships_"):
            raise AttributeError("fit must be called before transform.")

        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError("Columns in transform do not match columns seen during fit")

        X = check_input_dataset(X, require_numeric=True)
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


# --------------------------------------
# FCMParameterImputer
# --------------------------------------
class FCMParameterImputer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=3, m=2.0, max_iter=150, tol=1e-5, random_state=None):
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
        validate_params({
            'n_clusters': n_clusters,
            'm': m,
            'max_iter': max_iter,
            'tol': tol,
            'random_state': random_state
        })

        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fit the FCM imputer on complete data only.
        """
        X = check_input_dataset(X, require_numeric=True, require_complete_rows=True)
        complete, _ = split_complete_incomplete(X)

        if self.n_clusters > len(complete):
            raise ValueError("n_clusters cannot be larger than the number of complete rows")

        self.centers_, self.memberships_ = fuzzy_c_means(
            complete.to_numpy(),
            n_clusters=self.n_clusters,
            m=self.m,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state
        )

        self.feature_names_in_ = list(X.columns)

        return self

    def transform(self, X):
        """
        Impute missing values using parameter-based FCM method.
        Each missing value is the weighted sum of all centroids
        based on membership values.
        """
        if not hasattr(self, "centers_") or not hasattr(self, "memberships_"):
            raise AttributeError("fit must be called before transform.")

        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError("Columns in transform do not match columns seen during fit")

        X = check_input_dataset(X, require_numeric=True).copy()
        _, incomplete = split_complete_incomplete(X)

        if incomplete.empty:
            return X

        X_imputed = X.copy()

        for idx, row in incomplete.iterrows():
            obs = row.to_numpy()

            dist = np.array([euclidean_distance(obs, center) for center in self.centers_])
            dist = np.fmax(dist, 1e-10)

            u = 1 / np.sum((dist[:, None] / dist[None, :]) ** (2 / (self.m - 1)), axis=1)

            missing_cols = row[row.isna()].index
            for col in missing_cols:
                X_imputed.at[idx, col] = np.sum(u * self.centers_[:, X.columns.get_loc(col)])

        return X_imputed


# --------------------------------------
# FCMRoughParameterImputer
# --------------------------------------
class FCMRoughParameterImputer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=3, m=2.0, max_iter=100, tol=1e-5, wl=0.6, wb=0.4, tau=0.5, random_state=None):
        """
        Fuzzy C-Means Rough Parameter-based imputer.
        
        Each missing value is imputed using information from the 
        lower or upper approximation of the nearest fuzzy cluster.
        """
        validate_params({
            'n_clusters': n_clusters,
            'm': m,
            'max_iter': max_iter,
            'tol': tol,
            'wl': wl,
            'wb': wb,
            'tau': tau,
            'random_state': random_state
        })

        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.wl = wl
        self.wb = wb
        self.tau = tau
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fit the imputer on complete data.
        """
        X = check_input_dataset(X, require_numeric=True, require_complete_rows=True)
        complete, _ = split_complete_incomplete(X)

        if self.n_clusters > len(complete):
            raise ValueError("n_clusters cannot be larger than the number of complete rows")

        complete_array = complete.to_numpy()
        self.centers_, self.memberships_ = fuzzy_c_means(
            complete_array,
            n_clusters=self.n_clusters,
            m=self.m,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state
        )

        self.clusters_ = self._rough_kmeans_from_fcm(
            complete_array,
            self.memberships_,
            self.centers_,
            wl=self.wl,
            wb=self.wb,
            tau=self.tau,
            max_iter=self.max_iter,
            tol=self.tol
        )

        self.feature_names_in_ = list(X.columns)

        return self

    def transform(self, X):
        """
        Impute missing values using rough parameter-based FCM method.
        """
        if not hasattr(self, "centers_") or not hasattr(self, "memberships_") or not hasattr(self, "clusters_"):
            raise AttributeError("fit must be called before transform")

        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError("Columns in transform do not match columns seen during fit")

        X = check_input_dataset(X, require_numeric=True)

        _, incomplete = split_complete_incomplete(X)

        if incomplete.empty:
            return X

        X_imputed = X.copy()

        for idx, row in incomplete.iterrows():
            obs = row.to_numpy()

            distances = np.array([euclidean_distance(obs, center) for center in self.centers_])
            nearest_idx = np.argmin(distances)

            lower, upper, center = self.clusters_[nearest_idx]

            if len(lower) == 0:
                approx_data = upper
            elif len(upper) == 0:
                approx_data = lower
            else:
                dist_to_lower = np.mean([euclidean_distance(obs, x) for x in lower]) if len(lower) > 0 else np.inf
                dist_to_upper = np.mean([euclidean_distance(obs, x) for x in upper]) if len(upper) > 0 else np.inf
                approx_data = lower if dist_to_lower <= dist_to_upper else upper

            missing_cols = row[row.isna()].index
            for col in missing_cols:
                col_idx = X.columns.get_loc(col)

                approx_data = np.atleast_2d(approx_data)
                X_imputed.at[idx, col] = np.mean(approx_data[:, col_idx])

        return X_imputed

    def _rough_kmeans_from_fcm(self, X, memberships, center_init, wl=0.6, wb=0.4, tau=0.5, max_iter=100, tol=1e-4):
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


# --------------------------------------
# FCMKIterativeImputer
# --------------------------------------
class FCMKIterativeImputer(BaseEstimator, TransformerMixin):
    """
   Hybrid imputer combining fuzzy c-means clustering, k-nearest neighbors, and iterative imputation.

   FCKI improves missing data imputation by first clustering data with fuzzy c-means,
   allowing points to belong to multiple clusters. It then performs KNN-based imputation
   within clusters, followed by iterative imputation for refinement. This two-level
   similarity search enhances accuracy compared to standard KNN imputation.
    """

    def __init__(self, random_state: Optional[int] = None, max_clusters: int = 10, m: float = 2, max_iter: int = 30):
        if random_state is not None and not isinstance(random_state, int):
            raise TypeError('Invalid random_state: Expected an integer or None')

        if not isinstance(max_clusters, int) or max_clusters <= 1:
            raise TypeError('Invalid max_clusters: Expected an integer greater than 1')

        if not isinstance(m, (int, float)) or m <= 1:
            raise TypeError('Invalid m value: Expected a numeric value greater than 1')

        if not isinstance(max_iter, int) or max_iter <= 1:
            raise TypeError('Invalid max_iter: Expected a positive integer greater than 1')

        self.random_state = random_state
        self.max_clusters = max_clusters
        self.m = m
        self.max_iter = max_iter
        pass

    def fit(self, X, y=None):
        X = check_input_dataset(X, require_numeric=True, no_nan_columns=True)
        self.X_train_ = X.copy()

        self.imputer_ = SimpleImputer(strategy="mean")
        X_filled = self.imputer_.fit_transform(self.X_train_)
        X_filled = pd.DataFrame(data=X_filled, columns=X.columns, index=X.index)

        self.optimal_c_ = find_optimal_clusters_fuzzy(X_filled, min_clusters=1, max_clusters=self.max_clusters,
                                                      m=self.m, random_state=self.random_state)
        self.np_rng_ = np.random.RandomState(self.random_state)
        np.random.seed(self.random_state)

        self.centers_, self.u_ = fuzzy_c_means(
            X_filled.values,
            n_clusters=self.optimal_c_,
            m=self.m,
            random_state=self.random_state,
        )

        return self

    def transform(self, X):
        X = check_input_dataset(X, require_numeric=True, no_nan_rows=True)
        check_is_fitted(self, attributes=["X_train_", "imputer_", "centers_", "u_", "optimal_c_", "np_rng_"])

        if not X.columns.equals(self.X_train_.columns):
            raise ValueError(
                f"Invalid input: Input dataset columns do not match columns seen during fit"
            )

        X_imputed = self._FCKI_algorithm(X)
        return X_imputed

    def _find_best_k(self, St: pd.DataFrame, random_col: int, original_value: float) -> int:
        """
        Select the optimal number of neighbors (k) that minimizes RMSE
        when imputing a masked value in a selected column.
    
        Parameters:
            St (pd.DataFrame): Data with last row partially masked.
            random_col (int): Index of the masked column.
            original_value (float): True value before masking.

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

        max_k = min(n - 1, self.max_iter)
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

    def _get_neighbors(self, train: list[list[float]], test_row: list[float], k: int) -> list[list[float]]:
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

    def _KI_algorithm(self, X: pd.DataFrame, X_train: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Impute missing values using the KI method (KNN + Iterative Imputation).

        Parameters:
            X (pd.DataFrame): Data to impute.
            X_train (pd.DataFrame): Optional reference data (default is None).

        Returns:
            pd.DataFrame: Imputed dataset (same shape and index as X).
        """

        X_incomplete_rows = X.copy()
        X_mis = X_incomplete_rows[X_incomplete_rows.isnull().any(axis=1)]

        if X_train is not None and not X.equals(X_train):
            X_safe = X.copy()
            X_train_safe = X_train.copy()

            X_safe_reset = X_safe.reset_index(drop=True)
            X_train_safe_reset = X_train_safe.reset_index(drop=True)

            all_data = pd.concat([X_safe_reset, X_train_safe_reset], axis=0, ignore_index=True)
            index_map = dict(zip(X.index, range(len(X))))
        else:
            all_data = X.reset_index(drop=True).copy()
            index_map = dict(zip(X.index, range(len(X))))

        mis_idx = X_mis.index.to_numpy()
        imputed_values = []

        for idx in mis_idx:
            xi = X_incomplete_rows.loc[idx]

            A_mis = [col for col in X.columns if pd.isnull(xi[col])]

            P = all_data.dropna(subset=A_mis)
            if P.empty:
                raise ValueError(f"Invalid input: No rows with valid values found in columns: {A_mis}")

            P_ext = np.vstack([P.to_numpy(), xi.to_numpy()])

            St = P_ext.copy()
            St_Complete_Temp = St.copy()

            A_r = self.np_rng_.randint(0, St_Complete_Temp.shape[1])
            AV = St_Complete_Temp[-1, A_r]
            while np.isnan(AV):
                A_r = self.np_rng_.randint(0, St_Complete_Temp.shape[1])
                AV = St_Complete_Temp[-1, A_r]
            St[-1, A_r] = np.NaN

            k = self._find_best_k(pd.DataFrame(St, columns=X.columns), A_r, AV)

            xi_from_Pt = P_ext[-1, :].tolist()
            Pt_without_xi = P_ext[:-1, :].tolist()

            neighbors_xi = self._get_neighbors(Pt_without_xi, xi_from_Pt, k)
            S = np.vstack([neighbors_xi, xi.to_numpy()])

            imputer = IterativeImputer(random_state=self.random_state, max_iter=self.max_iter)
            S_filled_EM = imputer.fit_transform(S)

            xi_imputed = S_filled_EM[-1, :]
            imputed_values.append(xi_imputed)

            if idx in index_map:
                all_data.iloc[index_map[idx]] = xi_imputed

        if imputed_values:
            X_incomplete_rows.loc[mis_idx, :] = np.vstack(imputed_values)

        return X_incomplete_rows

    def _FCKI_algorithm(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using the FCKI method (FCM + KNN + Iterative Imputation).

        Parameters:
            X (pd.DataFrame): Data to impute.

        Returns:
            np.ndarray: Imputed dataset.
        """
        X_filled = self.imputer_.transform(X)
        X_filled = pd.DataFrame(data=X_filled, columns=X.columns, index=X.index)
        membership_matrix = fcm_predict(X_filled.values, self.centers_, self.m)
        fcm_labels_train = self.u_.argmax(axis=1)
        fcm_labels_X = membership_matrix.argmax(axis=1)

        all_clusters = pd.DataFrame(columns=X.columns)

        for i in range(self.optimal_c_):
            cluster_train_i = self.X_train_[fcm_labels_train == i]
            cluster_X_i = X[fcm_labels_X == i]
            imputed_cluster_X_I = self._KI_algorithm(cluster_X_i, cluster_train_i)
            imputed_cluster_X_I = pd.DataFrame(imputed_cluster_X_I, columns=X.columns, index=cluster_X_i.index)
            if len(all_clusters) == 0:
                all_clusters = imputed_cluster_X_I
            else:
                all_clusters = pd.concat([all_clusters, imputed_cluster_X_I], axis=0)

        all_clusters = all_clusters.loc[~all_clusters.index.duplicated(keep='last')]

        all_clusters.sort_index(inplace=True)

        return all_clusters


# --------------------------------------
# FCMInterpolationIterativeImputer
# --------------------------------------

class FCMInterpolationIterativeImputer(BaseEstimator, TransformerMixin):

    def __init__(self, n_clusters: int = 3, m: float = 2.0, alpha: float = 2.0, max_iter: int = 100, tol: float = 1e-5,
                 max_outer_iter: int = 20, stop_criteria: float = 0.01, sigma: bool = False,
                 random_state: Optional[int] = None, ):

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

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "FCMInterpolationIterativeImputer":
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

        for _ in range(self.max_outer_iter):
            U_star, V_star, _ = self._ifcm(X_filled)

            X_new = X_filled.copy()
            for i in range(X_filled.shape[0]):
                for j in range(X_filled.shape[1]):
                    if missing_mask.iloc[i, j]:
                        X_new.iloc[i, j] = np.sum(U_star[i, :] * V_star[:, j])

            diff = np.abs(X_new - X_filled)
            rel_diff = diff / (np.abs(X_filled) + 1e-10)
            if missing_mask.values.any():
                AvgV = rel_diff[missing_mask].mean().mean()
            else:
                AvgV = 0

            if AvgV <= self.stop_criteria:
                return X_new

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

        sigma_val = 1.0
        if isinstance(self.sigma, (int, float)):
            sigma_val = float(self.sigma)

        for _ in range(self.max_iter):
            eta = 1 - U - (1 - U ** self.alpha) ** (1 / self.alpha)
            eta = np.clip(eta, 0, 1)

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
                        D2 = diff**2
                        sim = np.exp(-D2 / (2 * sigma_val**2))
                        diff = 1 - sim 

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


# --------------------------------------
# FCMDTIterativeImputer
# --------------------------------------

class FCMDTIterativeImputer(BaseEstimator, TransformerMixin):
    """
    An iterative data imputer that combines Decision Trees and Fuzzy C-Means clustering
    to estimate and refine missing values in mixed-type datasets (numerical and categorical).
    """

    def __init__(self, random_state=None, min_samples_leaf=3, learning_rate=0.1, m=2, max_clusters=20, max_iter=100,
                 stop_threshold=1.0,
                 alpha=1.0):
        if random_state is not None and not isinstance(random_state, int):
            raise TypeError('Invalid random_state: Expected an integer or None.')

        if not isinstance(min_samples_leaf, (int, float)) or min_samples_leaf <= 0:
            raise TypeError('Invalid min_samples_leaf value: Expected a numeric value greater than 0.')

        if not isinstance(learning_rate, (float, int)) or (learning_rate <= 0):
            raise TypeError('Invalid learning_rate value: Expected a numeric value greater than 0.')

        if not isinstance(m, (int, float)) or m <= 1:
            raise TypeError('Invalid m value: Expected a numeric value greater than 1.')

        if not isinstance(max_clusters, int) or max_iter <= 1:
            raise TypeError('Invalid max_clusters value: Expected an integer greater than 1.')

        if not isinstance(max_iter, int) or max_iter <= 1:
            raise TypeError('Invalid max_iter value: Expected an integer greater than 1.')

        if not isinstance(stop_threshold, (int, float)) or stop_threshold < 0:
            raise TypeError('Invalid stop_threshold value: Expected a numeric value >= 0.')

        if not isinstance(alpha, (int, float)) or alpha <= 0:
            raise TypeError('Invalid alpha value: Expected a numeric value greater than 0.')

        self.random_state = random_state
        self.min_samples_leaf = min_samples_leaf
        self.learning_rate = learning_rate
        self.m = m
        self.max_clusters = max_clusters
        self.max_iter = max_iter
        self.stop_threshold = stop_threshold
        self.alpha = alpha

    def fit(self, X, y=None):
        X = check_input_dataset(X, require_numeric=False)
        self.X_train_complete_, _ = split_complete_incomplete(X.copy())

        if self.X_train_complete_.empty:
            raise ValueError("Invalid input: Input dataset has no complete records")

        if self.X_train_complete_.shape[1] < 2:
            raise ValueError("Invalid input: Input dataset has only one column")

        self.imputer_ = self._create_mixed_imputer(self.X_train_complete_)
        self.trees_ = {}
        self.leaf_indices_ = {}
        self.encoders_ = {}
        X_for_tree = self.X_train_complete_.copy()
        cat_cols = X_for_tree.select_dtypes(exclude=["number"]).columns

        for col in cat_cols:
            enc = OrdinalEncoder()
            X_for_tree[col] = enc.fit_transform(X_for_tree[[col]])
            self.encoders_[col] = enc

        for j in self.X_train_complete_.columns:
            if is_numeric_dtype(self.X_train_complete_[j]):
                tree = DecisionTreeRegressor(min_samples_leaf=self.min_samples_leaf, random_state=self.random_state)
            else:
                tree = DecisionTreeClassifier(min_samples_leaf=self.min_samples_leaf, random_state=self.random_state)

            X_without_j_column = X_for_tree.drop(columns=[j])
            tree.fit(X_without_j_column, self.X_train_complete_[j])

            self.leaf_indices_[j] = tree.apply(X_without_j_column)
            self.trees_[j] = tree
        return self

    def transform(self, X):
        X = check_input_dataset(X, require_numeric=False)
        check_is_fitted(self, attributes=["X_train_complete_"])

        if not X.columns.equals(self.X_train_complete_.columns):
            raise ValueError(
                f"Invalid input: Input dataset columns do not match columns seen during fit"
            )
        complete_X, incomplete_X = split_complete_incomplete(X.copy())

        if len(incomplete_X) == 0:
            return X

        mask_missing = incomplete_X.isna()
        cols_with_nan = incomplete_X.columns[incomplete_X.isna().any()]

        if X.select_dtypes(exclude=["number"]).empty:
            fcm_function = fuzzy_c_means
        else:
            fcm_function = fuzzy_c_means_categorical

        incomplete_leaf_indices_dict, imputed_X = self._initial_imputation_DT(incomplete_X.copy(), cols_with_nan)

        AV = np.inf
        old_df = imputed_X.copy()
        count_iter = 0

        while AV > self.stop_threshold and count_iter < self.max_iter:

            for j in cols_with_nan:
                leaf_for_j = [(idx, leaf_number) for (idx, j_key), leaf_number in incomplete_leaf_indices_dict.items()
                              if j_key == j]
                leaf_numbers = list(dict.fromkeys([leaf[0] for _, leaf in leaf_for_j]))

                for k in leaf_numbers:
                    imputed_X = self._improve_imputations_in_leaf(k, j, incomplete_leaf_indices_dict, imputed_X,
                                                                  fcm_function)
            new_df = imputed_X.copy()
            AV = self._calculate_AV(new_df, old_df, mask_missing)
            count_iter += 1
            old_df = new_df

        combined = pd.concat([complete_X, imputed_X]).sort_index()
        return combined

    def _determine_optimal_n_clusters_FSI(self, X, fcm_function):
        c_values = list(range(1, min(len(X), self.max_clusters) + 1))

        FSI = []
        for c in c_values:
            centers, u = fcm_function(X.to_numpy(), c, self.m, self.max_iter, random_state=self.random_state)
            FSI.append(self._fuzzy_silhouette(X.to_numpy(), u, self.alpha))

        opt_c = c_values[np.argmax(FSI)]
        return opt_c

    def _fuzzy_silhouette(self, X, U, alpha=1.0):
        X = pd.DataFrame(X)
        only_numeric = all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns)

        if only_numeric:
            D = cdist(X.to_numpy().astype(float), X.to_numpy().astype(float), metric="euclidean")
        else:
            D = gower_matrix(X)

        N, C = U.shape
        s = np.zeros(N)
        cluster_labels = np.argmax(U, axis=1)

        for j in range(N):
            cj = cluster_labels[j]
            in_cluster = (cluster_labels == cj)
            out_clusters = [k for k in range(C) if k != cj]
            a_j = np.mean(D[j, in_cluster]) if np.sum(in_cluster) > 1 else 0
            mean_dists = []
            for k in out_clusters:
                mask = (cluster_labels == k)
                if np.any(mask):
                    mean_dists.append(np.mean(D[j, mask]))

            if len(mean_dists) > 0:
                b_j = np.min(mean_dists)
            else:
                b_j = a_j
            s[j] = (b_j - a_j) / max(a_j, b_j) if max(a_j, b_j) > 0 else 0.0
        sorted_U = np.sort(U, axis=1)
        p = sorted_U[:, -1]
        if U.shape[1] > 1:
            q = sorted_U[:, -2]
        else:
            q = np.zeros_like(p)

        weights = (p - q) ** alpha
        FS = np.sum(weights * s) / np.sum(weights) if np.sum(weights) > 0 else 0.0
        return FS

    def _create_mixed_imputer(self, X):
        numeric_cols = X.select_dtypes(include=["number"]).columns
        categorical_cols = X.select_dtypes(exclude=["number"]).columns

        numeric_imputer = SimpleImputer(strategy="mean")
        categorical_imputer = SimpleImputer(strategy="most_frequent")

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_imputer, numeric_cols),
                ("cat", categorical_imputer, categorical_cols)
            ],
            remainder='passthrough'
        )
        preprocessor.set_output(transform="pandas")
        preprocessor.fit(X)
        return preprocessor

    def _calculate_AV(self, new_df, old_df, mask_missing):
        diffs = []
        for col in new_df.columns:
            mask_col = mask_missing[col]
            if pd.api.types.is_numeric_dtype(new_df[col]):
                diff = (new_df[col] - old_df[col]).abs()
            else:
                diff = new_df[col].ne(old_df[col]).astype(float)
            diff = diff[mask_col]
            diffs.append(diff)
        all_diffs = pd.concat(diffs)
        if len(all_diffs) == 0:
            return 0.0
        AV = all_diffs.mean()
        return AV

    def _initial_imputation_DT(self, incomplete_X, cols_with_nan):
        incomplete_leaf_indices_dict = {}
        imputed_X = incomplete_X.copy()

        for idx, xi in incomplete_X.iterrows():
            count_missing_values = xi.isna().sum()
            for j in cols_with_nan:
                if pd.isnull(xi[j]):
                    xi_filled = pd.DataFrame([xi.copy()], columns=incomplete_X.columns)
                    if count_missing_values > 1:
                        for col in self.X_train_complete_.columns:
                            xi_filled[col] = xi_filled[col].astype(self.X_train_complete_[col].dtype)

                        xi_imputed = self.imputer_.transform(xi_filled)
                        xi_imputed.columns = self.imputer_.get_feature_names_out()
                        xi_imputed.columns = [col.split("__")[-1] for col in xi_imputed.columns]
                        xi_filled = xi_imputed[self.X_train_complete_.columns]
                        count_missing_values -= 1
                    tree = self.trees_[j]

                    xi_without_j = xi_filled.drop(columns=[j])

                    for col, enc in self.encoders_.items():
                        if col in xi_without_j.columns:
                            val = xi_without_j[col].iloc[0]
                            if isinstance(val, np.ndarray):
                                val = val.item() if val.size == 1 else val[0]
                            val_df = pd.DataFrame({col: [val]})
                            xi_without_j[col] = enc.transform(val_df)[0, 0]

                    xi[j] = tree.predict(xi_without_j)
                    leaf_idx = tree.apply(xi_without_j)
                    incomplete_leaf_indices_dict[(idx, j)] = leaf_idx

            imputed_X.loc[idx] = xi
        return incomplete_leaf_indices_dict, imputed_X

    def _improve_imputations_in_leaf(self, k, j, incomplete_leaf_indices_dict, imputed_X, fcm_function):

        complete_records_in_leaf = self.X_train_complete_[self.leaf_indices_[j] == k]

        matching_indices = [idx for (idx, j_key), leaf_number in incomplete_leaf_indices_dict.items() if
                            j_key == j and (hasattr(leaf_number, "__iter__") and k in leaf_number)]
        records_in_leaf = pd.concat([complete_records_in_leaf, imputed_X.loc[matching_indices]])

        n_clusters = self._determine_optimal_n_clusters_FSI(records_in_leaf, fcm_function)
        centers, u = fcm_function(records_in_leaf.to_numpy(), n_clusters, self.m, max_iter=self.max_iter,
                                  random_state=self.random_state)

        col_j_idx = self.X_train_complete_.columns.get_loc(j)
        centers = np.asarray(centers)
        local_indices = {idx: pos for pos, idx in enumerate(records_in_leaf.index)}

        for i in matching_indices:
            i_local = local_indices[i]
            if is_numeric_dtype(self.X_train_complete_[j]):
                correction = self.learning_rate * (
                        u[i_local, :] @ centers[:, col_j_idx] - imputed_X.loc[i, j])
                imputed_X.loc[i, j] = imputed_X.loc[i, j] + correction
            else:
                u_i = u[i_local, :]
                sums = defaultdict(float)
                for k, cat in enumerate(centers[:, col_j_idx]):
                    sums[cat] += u_i[k]
                new_value = max(sums, key=sums.get)
                imputed_X.loc[i, j] = new_value

        return imputed_X
