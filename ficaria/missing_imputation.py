import warnings
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

from .utils import *


# --------------------------------------
# FCMCentroidImputer
# --------------------------------------
class FCMCentroidImputer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=5, m=2.0, max_iter=100, tol=1e-5, random_state=None):
        """"
        Fuzzy C-Means centroid-based imputer.

        Missing values are imputed using the centroid of the closest fuzzy cluster,
        where the nearest cluster is determined by Euclidean distance between an
        incomplete object and all FCM centroids.

        Parameters
        ----------
        n_clusters : int, default=5
            Number of fuzzy clusters used by the IIFCM algorithm.
            Must be an integer >= 1.

        m : {int, float}, default=2.0
            Fuzzification exponent controlling cluster softness.
            Must be > 1.

        max_iter : int, default=100
            Maximum number of iterations used by the FCM clustering algorithm.
            Must be > 1.

        tol : {int, float}, default=1e-5
            Convergence tolerance for stopping IIFCM updates.
            Must be > 0.

        random_state : {int, None}, default=None
            Seed for reproducibility of internal stochastic components.
            If None, randomness is not fixed.
        """

        validate_params({
            'm': m,
            'max_iter': max_iter,
            'tol': tol,
            'random_state': random_state
        })

        if not isinstance(n_clusters, int):
            raise TypeError(f"n_clusters must be int, got {type(n_clusters).__name__} instead")
        if isinstance(n_clusters, int) and n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters} instead")

        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fit the imputer by applying FCM on complete rows only.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input dataset to fit the imputer.

        y : None, default=None
            Ignored. Included for compatibility with scikit-learn.

        Returns
        -------
        self : FCMCentroidImputer
            Fitted imputer ready for transformation.
        """
        X = check_input_dataset(X, require_numeric=True, require_complete_rows=True)
        complete, _ = split_complete_incomplete(X)

        if self.n_clusters > len(complete):
            raise ValueError(
                f"n_clusters must be ≤ the number of complete rows ({len(complete)}), got {self.n_clusters} instead")

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
        Impute missing values by assigning each incomplete observation to the nearest FCM centroid.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Dataset to impute.

        Returns
        -------
        X_imputed : pandas DataFrame of shape (n_samples, n_features)
            Fully imputed dataset containing no missing values.
        """

        check_is_fitted(self, attributes=["centers_", "memberships_", "feature_names_in_"])
        X = check_input_dataset(X, require_numeric=True)

        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError(
                f"X.columns must match the columns seen during fit {list(self.feature_names_in_)}, got {list(X.columns)} instead")

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
    def __init__(self, n_clusters=5, m=2.0, max_iter=100, tol=1e-5, random_state=None):
        """
        Fuzzy C-Means parameter-based imputer.

        Each missing value is imputed using a membership-weighted linear combination
        of all cluster centroids, computed from FCM membership degrees.

        Parameters
        ----------
        n_clusters : int, default=5
            Number of fuzzy clusters used by the IIFCM algorithm.
            Must be an integer >= 1.

        m : {int, float}, default=2.0
            Fuzzification exponent controlling cluster softness.
            Must be > 1.

        max_iter : int, default=100
            Maximum number of iterations used by the FCM clustering algorithm.
            Must be > 1.

        tol : {int, float}, default=1e-5
            Convergence tolerance for stopping IIFCM updates.
            Must be > 0.

        random_state : {int, None}, default=None
            Seed for reproducibility of internal stochastic components.
            If None, randomness is not fixed.
        """
        validate_params({
            'm': m,
            'max_iter': max_iter,
            'tol': tol,
            'random_state': random_state
        })

        if not isinstance(n_clusters, int):
            raise TypeError(f"n_clusters must be int, got {type(n_clusters).__name__} instead")
        if isinstance(n_clusters, int) and n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters} instead")

        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fit the imputer by performing FCM on complete rows only.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input dataset to fit the imputer.

        y : None, default=None
            Ignored. Included for compatibility with scikit-learn.

        Returns
        -------
        self : FCMParameterImputer
            Fitted imputer ready for transformation.
        """
        X = check_input_dataset(X, require_numeric=True, require_complete_rows=True)
        complete, _ = split_complete_incomplete(X)

        if self.n_clusters > len(complete):
            raise ValueError(
                f"n_clusters must be ≤ the number of complete rows ({len(complete)}), got {self.n_clusters} instead")

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
        Impute missing values using membership-weighted centroid combinations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Dataset to impute.

        Returns
        -------
        X_imputed : pandas DataFrame of shape (n_samples, n_features)
            Fully imputed dataset containing no missing values.
        """

        check_is_fitted(self, attributes=["centers_", "memberships_", "feature_names_in_"])

        X = check_input_dataset(X, require_numeric=True).copy()

        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError(
                f"X.columns must match the columns seen during fit {list(self.feature_names_in_)}, "
                f"got {list(X.columns)} instead")

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
    def __init__(self, n_clusters=5, m=2.0, max_iter=100, max_iter_rough_k=100, tol=1e-5, wl=0.6, wb=0.4, tau=0.5, random_state=None):
        """
        Rough Fuzzy C-Means parameter-based imputer.

        Missing values are imputed using the lower or upper approximation of the
        nearest rough cluster generated from FCM memberships, enabling robust
        handling of objects in boundary regions.

        Parameters
        ----------
        n_clusters : int, default=5
            Number of fuzzy clusters used by the IIFCM algorithm.
            Must be an integer >= 1.

        m : {int, float}, default=2.0
            Fuzzification exponent controlling cluster softness.
            Must be > 1.

        max_iter : int, default=100
            Maximum number of iterations used by the FCM clustering algorithm.
            Must be > 1.

        max_iter_rough_k : int, default=100
            Maximum number of iterations used by the Rough K-Means clustering algorithm.
            Must be > 1.

        tol : {int, float}, default=1e-5
            Convergence tolerance for stopping IIFCM updates.
            Must be > 0.

        wl : {int, float},  default=0.6
            Weight assigned to the lower approximation during centroid updates.
            Must be in range (0, 1].

        wb : {int, float},  default=0.4
            Weight assigned to the boundary region.
            Must be in range [0, 1].

        tau : {int, float},  default=0.5
            Threshold controlling assignment of samples to boundary sets.
            Must be >= 0.

        random_state : {int, None}, default=None
            Seed for reproducibility of internal stochastic components.
            If None, randomness is not fixed.
        """

        validate_params({
            'm': m,
            'max_iter': max_iter,
            'max_iter_rough_k': max_iter_rough_k,
            'tol': tol,
            'wl': wl,
            'wb': wb,
            'tau': tau,
            'random_state': random_state
        })

        if not isinstance(n_clusters, int):
            raise TypeError(f"n_clusters must be int, got {type(n_clusters).__name__} instead")
        if isinstance(n_clusters, int) and n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters} instead")

        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.max_iter_rough_k = max_iter_rough_k
        self.tol = tol
        self.wl = wl
        self.wb = wb
        self.tau = tau
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fit the imputer by computing FCM clusters and refining them
        using Rough K-Means to obtain lower and upper cluster approximations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input dataset to fit the imputer.

        y : None, default=None
            Ignored. Included for compatibility with scikit-learn.

        Returns
        -------
        self : FCMParameterImputer
            Fitted imputer ready for transformation.
        """
        X = check_input_dataset(X, require_numeric=True, require_complete_rows=True)
        complete, _ = split_complete_incomplete(X)

        if self.n_clusters > len(complete):
            raise ValueError(
                f"n_clusters must be ≤ the number of complete rows ({len(complete)}), "
                f"got {self.n_clusters} instead")

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
            max_iter_rough_k=self.max_iter_rough_k,
            tol=self.tol
        )

        self.feature_names_in_ = list(X.columns)

        return self

    def transform(self, X):
        """
        Impute missing values using rough cluster approximations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Dataset to impute.

        Returns
        -------
        X_imputed : pandas DataFrame of shape (n_samples, n_features)
            Fully imputed dataset containing no missing values.
        """

        check_is_fitted(self, attributes=["centers_", "memberships_", "clusters_", "feature_names_in_"])

        X = check_input_dataset(X, require_numeric=True)

        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError(
                f"X.columns must match the columns seen during fit {list(self.feature_names_in_)}, "
                f"got {list(X.columns)} instead")

        _, incomplete = split_complete_incomplete(X)

        if incomplete.empty:
            return X

        X_imputed = X.copy()

        for row in incomplete.itertuples(index=True):
            idx = row.Index
            obs = np.array(row[1:]) 
            
            distances = np.linalg.norm(self.centers_ - obs, axis=1)
            nearest_idx = np.argmin(distances)

            lower, upper, center = self.clusters_[nearest_idx]

            if len(lower) == 0:
                approx_data = upper
            elif len(upper) == 0:
                approx_data = lower
            else:
                lower = np.array(lower)
                upper = np.array(upper)
                dist_to_lower = np.mean(np.linalg.norm(lower - obs, axis=1)) if len(lower) > 0 else np.inf
                dist_to_upper = np.mean(np.linalg.norm(upper - obs, axis=1)) if len(upper) > 0 else np.inf
                approx_data = lower if dist_to_lower <= dist_to_upper else upper

            missing_cols = np.where(np.isnan(obs))[0]
            for col_idx in missing_cols:
                X_imputed.iat[idx, col_idx] = np.mean(approx_data[:, col_idx])


        return X_imputed

    def _rough_kmeans_from_fcm(self, X, memberships, center_init, wl=0.6, wb=0.4, tau=0.5, max_iter_rough_k=100, tol=1e-4):
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

        upper_mask = np.zeros((n_samples, n_clusters), dtype=bool)
        lower_mask = np.zeros((n_samples, n_clusters), dtype=bool)

        init_labels = np.argmax(memberships, axis=1)
        upper_mask[np.arange(n_samples), init_labels] = True
        lower_mask[np.arange(n_samples), init_labels] = True

        for iteration in range(max_iter_rough_k):

            new_centers = np.zeros_like(centers)

            for k in range(n_clusters):
                lower_idx = np.where(lower_mask[:, k])[0]
                upper_idx = np.where(upper_mask[:, k])[0]

                if lower_idx.size == 0:
                    new_centers[k] = centers[k]
                    continue

                boundary_mask = ~np.isin(upper_idx, lower_idx)
                boundary_idx = upper_idx[boundary_mask]

                lower_mean = X[lower_idx].mean(axis=0)

                if boundary_idx.size > 0:
                    boundary_mean = X[boundary_idx].mean(axis=0)
                    new_centers[k] = wl * lower_mean + wb * boundary_mean
                else:
                    new_centers[k] = lower_mean

            distances = np.linalg.norm(X[:, None, :] - new_centers[None, :, :], axis=2)

            winners = np.argmin(distances, axis=1)
            dmin = distances[np.arange(n_samples), winners]

            boundary_mask = (distances - dmin[:, None]) <= tau

            new_upper_mask = boundary_mask.copy()

            only_one = new_upper_mask.sum(axis=1) == 1
            new_lower_mask = np.zeros_like(new_upper_mask)
            new_lower_mask[np.arange(n_samples)[only_one], winners[only_one]] = True

            shift = np.linalg.norm(new_centers - centers)

            if shift < tol:
                break

            centers = new_centers
            upper_mask = new_upper_mask
            lower_mask = new_lower_mask

        clusters = []
        for k in range(n_clusters):
            lower = X[lower_mask[:, k]]
            upper = X[upper_mask[:, k]]
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
   `similarity search enhances accuracy compared to standard KNN imputation.

    Parameters
    ----------
    n_clusters : {int, None}, default=None
        Number of fuzzy clusters used by the FCMKII algorithm.
        Must be an integer >= 1, or None to enable automatic selection.

    max_clusters : int, default=10
        Maximum number of clusters to test when searching for the optimal
        cluster count.
        Must be >= 1.

    m : {int, float}, default=2
        Fuzzifier exponent used in fuzzy c-means. Controls cluster "softness."
        Must be > 1.

    max_FCM_iter : int, default=100
        Maximum number of iterations allowed for the FCM algorithm.
        Must be >= 1.

    max_II_iter : int, default=80
        Maximum number of iterations for the iterative imputer.
        Must be > 1.

    max_k : int, default=20
        Maximum possible number of neighbors evaluated during the adaptive
        KNN step.
        Must be >= 1.

    tol : {int, float}, default=1e-5
        Convergence tolerance for fuzzy c-means.
        Must be > 0.

    random_state : {int, None}, default=None
        Seed for reproducibility of internal stochastic components.
        If None, randomness is not fixed.

    Attributes
    ----------
    X_train_ : pandas DataFrame of shape (n_samples, n_features)
        A validated copy of the training data.

    imputer_ : SimpleImputer
        Mean-imputer used prior to FCM preprocessing and membership prediction.

    centers_ : ndarray of shape (n_clusters, n_features)
        Final cluster centers inferred by fuzzy c-means.

    u_ : ndarray of shape (n_samples, n_clusters)
        Membership matrix output by FCM for training data.

    np_rng_ : numpy.random.RandomState
        Random generator used during adaptive neighbor masking.
    """

    def __init__(self, n_clusters=None, max_clusters=10, m=2, max_FCM_iter=100, max_II_iter=80, max_k=20, tol=1e-5,
                 random_state=None):

        validate_params({
            'n_clusters': n_clusters,
            'max_clusters': max_clusters,
            'm': m,
            'max_FCM_iter': max_FCM_iter,
            'max_II_iter': max_II_iter,
            'max_k': max_k,
            'tol': tol,
            'random_state': random_state
        })

        if n_clusters is not None and not isinstance(n_clusters, int):
            raise TypeError(f"n_clusters must be int or None, got {type(n_clusters).__name__} instead")
        if isinstance(n_clusters, int) and n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters} instead")

        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_clusters = max_clusters
        self.m = m
        self.max_FCM_iter = max_FCM_iter
        self.max_II_iter = max_II_iter
        self.max_k = max_k
        self.tol = tol

    def fit(self, X, y=None):
        """
        Fit the FCMKIterativeImputer to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input dataset to fit the imputer.

        y : None, default=None
            Ignored. Included for compatibility with scikit-learn.

        Returns
        -------
        self : FCMKIterativeImputer
            Fitted estimator.
        """
        X = check_input_dataset(X, require_numeric=True, no_nan_columns=True)
        self.X_train_ = X.copy()

        self.imputer_ = SimpleImputer(strategy="mean")
        X_filled = self.imputer_.fit_transform(self.X_train_)
        X_filled = pd.DataFrame(data=X_filled, columns=X.columns, index=X.index)

        if self.n_clusters is None:
            self.n_clusters = find_optimal_clusters_fuzzy(X_filled, max_clusters=self.max_clusters,
                                                          random_state=self.random_state, m=self.m,
                                                          max_iter=self.max_FCM_iter, tol=self.tol)

        if self.n_clusters > len(X):
            raise ValueError("n_clusters cannot be larger than the number of rows in X")

        self.np_rng_ = np.random.RandomState(self.random_state)
        np.random.seed(self.random_state)

        self.centers_, self.u_ = fuzzy_c_means(X_filled.values, n_clusters=self.n_clusters, m=self.m,
                                               max_iter=self.max_FCM_iter, tol=self.tol, random_state=self.random_state)
        return self

    def transform(self, X):
        """
        Impute missing values in X using the FCM + KNN + iterative imputation pipeline.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Dataset to impute.

        Returns
        -------
        X_imputed : pandas DataFrame of shape (n_samples, n_features)
            Fully imputed dataset containing no missing values.
        """
        X = check_input_dataset(X, require_numeric=True, no_nan_rows=True)
        check_is_fitted(self, attributes=["X_train_", "imputer_", "centers_", "u_", "np_rng_"])

        if not X.columns.equals(self.X_train_.columns):
            raise ValueError(
                f"X.columns must match the columns seen during fit {list(self.X_train_.columns)}, "
                f"got {list(X.columns)} instead")

        X_imputed = self._FCKI_algorithm(X)
        return X_imputed

      def _find_best_k(self, St, random_col, original_value, distances):
        """
        Select the optimal number of neighbors (n_features) that minimizes RMSE
        when imputing a masked value in a selected column.

        Parameters:
            St (np.ndarray): Data with last row partially masked.
            random_col (int): Index of the masked column.
            original_value (float): True value before masking.
            distances (np.ndarray): Distances of xi to all other rows in St

        Returns:
            int: Best value of n_features.
        """
        n = St.shape[0]
        if n <= 1:
            return 1

        St_without_xi = St[:-1, :]

        sorted_indices = np.argsort(distances)
        sorted_rows = St_without_xi[sorted_indices]

        max_k = min(n - 1, self.max_k)
        k_values = np.arange(1, max_k + 1)

        rmses = np.empty_like(k_values, dtype=float)

        for i, k in enumerate(k_values):
            col_vals = sorted_rows[:k, random_col]
            col_vals = col_vals[~np.isnan(col_vals)]

            if col_vals.size == 0:
                rmses[i] = np.inf
            else:
                mean_value = col_vals.mean()
                rmses[i] = np.abs(mean_value - original_value)

        best_k = int(k_values[np.argmin(rmses)])

        knn_idx = sorted_indices[:best_k]

        return knn_idx

    def _KI_algorithm(self, X, X_train=None):
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

        if len(X_mis) == 0:
            return X

        missing_counts = X_mis.isnull().sum(axis=1)
        mis_idx = missing_counts.sort_values().index.to_numpy()

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

        imputed_values = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)

            for idx in mis_idx:
                xi = X_incomplete_rows.loc[idx]

                A_mis = [col for col in X.columns if pd.isnull(xi[col])]

                P = all_data.dropna(subset=A_mis)
                if P.empty:
                    raise ValueError(
                        f"For any row with missing values, there must be at least one row where all "
                        f"those columns are complete, got none for columns {list(A_mis)} instead")

                P_ext = np.vstack([P.to_numpy(), xi.to_numpy()])

                St = P_ext.copy()
                St_Complete_Temp = St.copy()

                A_r = self.np_rng_.randint(0, St_Complete_Temp.shape[1])
                AV = St_Complete_Temp[-1, A_r]
                while np.isnan(AV):
                    A_r = self.np_rng_.randint(0, St_Complete_Temp.shape[1])
                    AV = St_Complete_Temp[-1, A_r]

                St[-1, A_r] = np.NaN

                xi_np = St[-1]
                St_without_xi = St[:-1]

                mask = ~np.isnan(St_without_xi) & ~np.isnan(xi_np)
                diffs = np.where(mask, St_without_xi - xi_np, 0)
                distances = np.sqrt(np.sum(diffs ** 2, axis=1))

                knn_idx = self._find_best_k(St, A_r, AV, distances)
                neighbors_xi = P_ext[knn_idx, :]
                S = np.vstack([neighbors_xi, xi.to_numpy()])

                imputer = IterativeImputer(random_state=self.random_state, max_iter=self.max_II_iter)
                S_filled_EM = imputer.fit_transform(S)

                xi_imputed = S_filled_EM[-1, :]
                imputed_values.append(xi_imputed)

                if idx in index_map:
                    all_data.iloc[index_map[idx]] = xi_imputed

        if imputed_values:
            X_incomplete_rows.loc[mis_idx, :] = np.vstack(imputed_values)

        return X_incomplete_rows

    def _FCKI_algorithm(self, X):
        """
        Impute missing values using the FCKI method (FCM + KNN + Iterative Imputation).

        Parameters:
            X (pd.DataFrame): Data to impute.

        Returns:
            np.ndarray: Imputed dataset.
        """
        X_filled = self.imputer_.transform(X)
        membership_matrix = fcm_predict(X_filled, self.centers_, self.m)

        fcm_labels_train = self.u_.argmax(axis=1)
        fcm_labels_X = membership_matrix.argmax(axis=1)

        imputed_clusters_list = []

        for i in range(self.n_clusters):
            cluster_train_i = self.X_train_[fcm_labels_train == i]
            cluster_X_i = X[fcm_labels_X == i]
            imputed_cluster_X_I = self._KI_algorithm(cluster_X_i, cluster_train_i)

            imputed_cluster_X_I = pd.DataFrame(imputed_cluster_X_I, columns=X.columns, index=cluster_X_i.index)
            imputed_clusters_list.append(imputed_cluster_X_I)

        if imputed_clusters_list:
            all_clusters = pd.concat(imputed_clusters_list, axis=0)
            all_clusters = all_clusters.loc[~all_clusters.index.duplicated(keep='last')]
            all_clusters.sort_index(inplace=True)
        else:
            all_clusters = X.copy()

        return all_clusters


# --------------------------------------
# FCMInterpolationIterativeImputer
# --------------------------------------

class FCMInterpolationIterativeImputer(BaseEstimator, TransformerMixin):

    def __init__(self, n_clusters=5, m=2.0, max_iter=100, alpha=0.85, tol=1e-5,
                 sigma=False, random_state=None):

        """
        Linear-Interpolation Intuitionistic Fuzzy C-Means Iterative Imputer (LI-IIFCM).

        Implements an iterative imputation algorithm that combines linear interpolation
        with intuitionistic fuzzy C-means (IIFCM). The method repeatedly estimates
        missing values using cluster prototypes weighted by intuitionistic membership.
        An optional IFCM-σ distance variant is supported to incorporate adaptive,
        cluster-specific variance scaling.

        Parameters
        ----------
        n_clusters : int, default=5
            Number of fuzzy clusters used by the IIFCM algorithm.
            Must be an integer >= 1.

        m : {int, float}, default=2.0
            Fuzzification exponent controlling cluster softness.
            Must be > 1.

        max_iter : int, default=100
            Maximum iteration count for the internal IIFCM optimization loop.
            Must be > 1.

        alpha : {int, float}, default=0.85
            Parameter controlling hesitation in intuitionistic fuzzification.
            Must be > 0.

        tol : {int, float}, default=1e-5
            Convergence tolerance for stopping IIFCM updates.
            Must be > 0.

        sigma : bool, default=False
            If True, applies the IFCM-σ distance metric with adaptive variance scaling.

        random_state : {int, None}, default=None
            Seed for reproducibility of internal stochastic components.
            If None, randomness is not fixed.

        Attributes
        ----------
        columns_ : list of str
            Column names of the fitted input dataset.

        centers_ : ndarray of shape (n_clusters, n_features)
            Learned cluster obtained after convergence of the IIFCM algorithm.

        sigma_ : ndarray of shape (n_clusters, n_features) or None
            Adaptive, feature-wise variance estimates for each cluster,
            used only when `sigma=True`.
        """

        validate_params({
            'm': m,
            'max_iter': max_iter,
            'tol': tol,
            'random_state': random_state
        })

        if not isinstance(n_clusters, int):
            raise TypeError(f"n_clusters must be int, got {type(n_clusters).__name__} instead")
        if isinstance(n_clusters, int) and n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters} instead")

        if not isinstance(alpha, (int, float)):
            raise TypeError(f"alpha must be int or float, got {type(alpha).__name__} instead")
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha} instead")

        if not isinstance(sigma, bool):
            raise TypeError(f"sigma must be bool, got {type(sigma).__name__} instead")

        self.n_clusters = n_clusters
        self.m = m
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.is_sigma = sigma
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fit the LI-IIFCM imputer on the dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input dataset to fit the imputer.

        y : None, default=None
            Ignored. Included for compatibility with scikit-learn.

        Returns
        -------
        self : FCMInterpolationIterativeImputer
            Fitted imputer with stored column metadata.
        """

        X = check_input_dataset(X, require_numeric=True, no_nan_columns=True)
        self.columns_ = X.columns
        complete, incomplete = split_complete_incomplete(X)

        X_filled = X.copy()
        X_filled = X_filled.interpolate(method='linear', axis=0, limit_direction='both')

        data = X_filled.to_numpy().copy()

        missing_mask = X.isna().values
        self.centers_ = self._ifcm(data, incomplete, missing_mask)

        return self

    def transform(self, X):
        """
        Impute missing values using linear interpolation followed by iterative
        intuitionistic fuzzy C-means.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Dataset to impute.

        Returns
        -------
        X_imputed : pandas DataFrame of shape (n_samples, n_features)
            Fully imputed dataset containing no missing values.
        """
        check_is_fitted(self, attributes=["columns_"])

        X = check_input_dataset(X, require_numeric=True)

        if list(X.columns) != list(self.columns_):
            raise ValueError(f"X.columns must match the columns seen during fit {list(self.columns_)}, "
                             f"got {list(X.columns)} instead")

        _, incomplete = split_complete_incomplete(X)

        if incomplete.empty:
            return X

        X_imputed = X.copy()

        for idx, row in incomplete.iterrows():
            distances = []

            for j, center in enumerate(self.centers_):
                mask = ~np.isnan(row.values)
                if self.is_sigma:
                    sigma_j = self.sigma_[j][mask]
                    diff = row.values[mask] - center[mask]
                    distances.append(np.sqrt(np.sum((diff ** 2) / (sigma_j + 1e-10))))
                else:
                    distances.append(np.linalg.norm(row.values[mask] - center[mask]))

            nearest_idx = np.argmin(distances)
            nearest_center = self.centers_[nearest_idx]

            missing_cols = row[row.isna()].index
            for col in missing_cols:
                X_imputed.at[idx, col] = nearest_center[X.columns.get_loc(col)]

        return X_imputed

    def _ifcm(self, data, incomplete, missing_mask):
        """
        Method implementing Intuitionistic Fuzzy C-Means clustering.
        Optionally applies weighted distance metric (IFCM-σ) if sigma=True.

        Parameters
        ----------
        data : np.ndarray or pd.DataFrame
            Numeric data without missing values.

        incomplete : pandas.DataFrame
            Subset of the original dataset containing only the rows that had missing
            values before interpolation.

        missing_mask : np.ndarray of shape (n_samples, n_features)
            Boolean mask indicating original missing-value in the dataset.

        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
            Estimated cluster after completing the iterative
            IIFCM optimization loop.
        """

        n_samples, n_features = data.shape

        rng = np.random.default_rng(self.random_state)
        u = rng.random((n_samples, self.n_clusters))
        u = u / np.sum(u, axis=1, keepdims=True)

        for _ in range(self.max_iter):
            n = 1 - u - (1 - u) ** (1 / self.alpha)
            u_star = u + n
            uv = u_star ** self.m

            centers = (uv.T @ data) / np.sum(uv.T, axis=1)[:, None]
            dist = np.zeros((n_samples, self.n_clusters))

            if self.is_sigma:
                sigma = np.zeros((self.n_clusters, n_features))
                for j in range(self.n_clusters):
                    u_m = uv[:, j]
                    diff = data - centers[j]
                    sigma[j] = np.sum(u_m[:, None] * diff ** 2, axis=0) / np.sum(u_m)

            for j in range(self.n_clusters):
                if self.is_sigma:
                    dist[:, j] = np.sqrt(np.sum(((data - centers[j]) ** 2) / (sigma[j] + 1e-10), axis=1))
                else:
                    dist[:, j] = np.linalg.norm(data - centers[j], axis=1)
            dist = np.fmax(dist, 1e-10)

            u = 1 / np.sum((dist[:, :, None] / dist[:, None, :]) ** (2 / (self.m - 1)), axis=2)

            prior = data.copy()
            for idx in incomplete.index:
                missing_cols = np.where(missing_mask[idx])[0]
                for j in missing_cols:
                    data[idx, j] = np.sum(u[idx] * centers[:, j]) / np.sum(u[idx])

            diff = np.abs(data[incomplete.index] - prior[incomplete.index])
            avgV = np.nanmean(diff)

            if avgV <= self.tol:
                break

        self.sigma_ = sigma if self.is_sigma else None
        return centers


# --------------------------------------
# FCMDTIterativeImputer
# --------------------------------------

class FCMDTIterativeImputer(BaseEstimator, TransformerMixin):
    """
    Decision tree–guided fuzzy c-means iterative imputer.

    This estimator performs missing-value imputation by first using decision
    trees to predict missing entries based on complete rows, then iteratively
    refining those predictions using localized fuzzy c-means clustering.
    The model identifies leaf-specific subgroups, applies adaptive fuzzy
    clustering within each subgroup, and updates imputed values with a
    gradient-like correction. This hybrid approach effectively captures
    non-linear relationships and local structure in fully numeric datasets.

    Parameters
    ----------
    max_clusters : int, default=20
        Maximum number of FCM clusters allowed during local clustering.
        Must be >= 1.

    m : {int, float}, default=2
        Fuzzifier exponent for FCM. Controls cluster softness.
        Must be > 1.

    max_iter : int, default=100
        Maximum number of global refinement iterations during transform.
        Must be > 1.

    max_FCM_iter : int, default=100
        Maximum iterations for fuzzy c-means clustering.
        Must be > 1.

    tol : {int, float}, default=1e-5
        Convergence tolerance for fuzzy c-means.
        Must be > 0.

    min_samples_leaf : int, default=40
        Minimum samples per leaf in the decision tree regressors.
        Must be > 0

    learning_rate : {int, float}, default=0.1
        Step size for local correction updates when adjusting imputed values.
        Must be > 0.

    stop_threshold : {int, float}, default=1.0
        Stopping criterion. Iteration ends once the mean absolute update
        across missing entries falls below this value.
        Must be >= 0.

    alpha : {int, float}, default=1.0
        Weighting exponent used in the fuzzy silhouette index when selecting
        the optimal number of clusters.
        Must be > 0.

    random_state : {int, None}, default=None
        Seed for reproducibility of internal stochastic components.
        If None, randomness is not fixed.

    Attributes
    ----------
    X_train_complete_ : pandas DataFrame
        Subset of training rows containing no missing values. Used to train
        all feature-specific decision trees.

    imputer_ : SimpleImputer
        Mean imputer used internally for temporary filling of rows with
        multiple missing values before applying decision-tree predictions.

    trees_ : dict
        Mapping from column name → fitted DecisionTreeRegressor used to
        generate initial imputations.

    leaf_indices_ : dict
        Mapping from column name → array of leaf indices for training rows.
        Used to form leaf-local subgroups during refinement.
    """

    def __init__(self, max_clusters=20, m=2, max_iter=100, max_FCM_iter=100, tol=1e-5, min_samples_leaf=40,
                 learning_rate=0.1, stop_threshold=1.0, alpha=1.0, random_state=None):

        validate_params({
            'max_clusters': max_clusters,
            'm': m,
            'max_iter': max_iter,
            'max_FCM_iter': max_FCM_iter,
            'tol': tol,
            'stop_threshold': stop_threshold,
            'random_state': random_state,
            'min_samples_leaf': min_samples_leaf,
            'learning_rate': learning_rate
        })

        if not isinstance(alpha, (int, float)):
            raise TypeError(f"alpha must be int or float, got {type(alpha).__name__} instead")
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha} instead")

        self.random_state = random_state
        self.min_samples_leaf = min_samples_leaf
        self.learning_rate = learning_rate
        self.m = m
        self.max_clusters = max_clusters
        self.max_iter = max_iter
        self.stop_threshold = stop_threshold
        self.alpha = alpha
        self.max_FCM_iter = max_FCM_iter
        self.tol = tol

    def fit(self, X, y=None):
        """
        Fit the decision-tree and FCM-based imputer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input dataset to fit the imputer.

        y : None, default=None
            Ignored. Included for compatibility with scikit-learn.

        Returns
        -------
        self : FCMDTIterativeImputer
            Fitted estimator.
        """
        X = check_input_dataset(X, require_numeric=True, require_complete_rows=True)
        self.X_train_complete_, _ = split_complete_incomplete(X.copy())

        if self.X_train_complete_.shape[1] < 2:
            raise ValueError(f"X must contain at least 2 columns, "
                             f"got {self.X_train_complete_.shape[1]} column instead")

        self.imputer_ = SimpleImputer(strategy="mean")
        self.imputer_.fit(X)
        self.trees_ = {}
        self.leaf_indices_ = {}
        X_for_tree = self.X_train_complete_.copy()

        for j in self.X_train_complete_.columns:
            tree = DecisionTreeRegressor(min_samples_leaf=self.min_samples_leaf, random_state=self.random_state)

            X_without_j_column = X_for_tree.drop(columns=[j])
            tree.fit(X_without_j_column, self.X_train_complete_[j])

            self.leaf_indices_[j] = tree.apply(X_without_j_column)
            self.trees_[j] = tree
        return self

    def transform(self, X):
        """
        Impute missing values using decision trees followed by iterative
        fuzzy c-means refinement.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input dataset to fit the imputer.

        Returns
        -------
        X_imputed : pandas DataFrame of shape (n_samples, n_features)
            Fully imputed dataset containing no missing values.
        """
        X = check_input_dataset(X, require_numeric=True)
        check_is_fitted(self, attributes=["X_train_complete_", "imputer_", "trees_", "leaf_indices_"])

        if not X.columns.equals(self.X_train_complete_.columns):
            raise ValueError(
                f"X.columns must match the columns seen during fit {list(self.X_train_complete_.columns)}, "
                f"got {list(X.columns)} instead")
        complete_X, incomplete_X = split_complete_incomplete(X.copy())

        if len(incomplete_X) == 0:
            return X

        mask_missing = incomplete_X.isna()
        cols_with_nan = incomplete_X.columns[incomplete_X.isna().any()]

        fcm_function = fuzzy_c_means

        incomplete_leaf_indices_dict, imputed_X = self._initial_imputation_DT(incomplete_X.copy(), cols_with_nan)

        AV = np.inf
        old_df = imputed_X.copy()
        count_iter = 0

        while AV > self.stop_threshold and count_iter < self.max_iter:

            col_to_row_indices = defaultdict(list)

            for (idx, col_idx), leaf in incomplete_leaf_indices_dict.items():
                col_to_row_indices[col_idx].append(idx)

            for j in cols_with_nan:
                leaf_numbers = list(set(col_to_row_indices[j]))

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
        if len(X) < 2:
            return 1
        c_values = list(range(1, min(len(X), self.max_clusters) + 1))

        FSI = []
        for c in c_values:
            centers, u = fcm_function(X.to_numpy(), c, self.m, self.max_FCM_iter, random_state=self.random_state,
                                      tol=self.tol)
            FSI.append(self._fuzzy_silhouette(X.to_numpy(), u, self.alpha))
        opt_c = c_values[np.argmax(FSI)]
        return opt_c

    def _fuzzy_silhouette(self, X, U, alpha=1.0):
        X = np.asarray(X, dtype=float)
        N, C = U.shape

        D = cdist(X, X, metric="euclidean")
        cluster_labels = np.argmax(U, axis=1)

        s = np.zeros(N)
        for j in range(N):
            cj = cluster_labels[j]
            in_cluster = (cluster_labels == cj)
            out_clusters = [k for k in range(C) if k != cj]

            a_j = np.mean(D[j, in_cluster]) if in_cluster.sum() > 1 else 0

            mean_dists = []
            for k in out_clusters:
                mask = (cluster_labels == k)
                if np.any(mask):
                    mean_dists.append(np.mean(D[j, mask]))
            b_j = np.min(mean_dists) if mean_dists else a_j

            s[j] = (b_j - a_j) / max(a_j, b_j) if max(a_j, b_j) > 0 else 0.0

        sorted_U = np.sort(U, axis=1)
        p = sorted_U[:, -1]
        q = sorted_U[:, -2] if C > 1 else np.zeros_like(p)
        weights = (p - q) ** alpha

        FS = (weights * s).sum() / weights.sum() if weights.sum() > 0 else 0.0
        return FS

    def _calculate_AV(self, new_df, old_df, mask_missing):
        diffs = (new_df - old_df).abs()
        masked_diffs = diffs.where(mask_missing)
        AV = masked_diffs.stack().mean()
        return 0.0 if pd.isna(AV) else AV

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
                        xi_filled = pd.DataFrame(data=xi_imputed, columns=self.X_train_complete_.columns)
                        count_missing_values -= 1
                    tree = self.trees_[j]

                    xi_without_j = xi_filled.drop(columns=[j])
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

        if len(records_in_leaf) < 2:
            return imputed_X

        n_clusters = self._determine_optimal_n_clusters_FSI(records_in_leaf, fcm_function)
        centers, u = fcm_function(records_in_leaf.to_numpy(), n_clusters, self.m, max_iter=self.max_FCM_iter,
                                  tol=self.tol, random_state=self.random_state)

        col_j_idx = self.X_train_complete_.columns.get_loc(j)
        centers = np.asarray(centers)
        local_indices = {idx: pos for pos, idx in enumerate(records_in_leaf.index)}

        for i in matching_indices:
            i_local = local_indices[i]
            correction = self.learning_rate * (
                    u[i_local, :] @ centers[:, col_j_idx] - imputed_X.loc[i, j])
            imputed_X.loc[i, j] = imputed_X.loc[i, j] + correction

        return imputed_X
