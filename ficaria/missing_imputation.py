from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
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
        X = self._check_input(X)
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
        X = self._check_input(X)
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

    def _check_input(self, X):
        """
        Convert input to DataFrame if not already.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X
    

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
        X = self._check_input(X)
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
        X = self._check_input(X).copy()
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
    
    
    def _check_input(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X
    

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
        X = self._check_input(X)
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
        X = self._check_input(X)
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

    def _check_input(self, X):
        """
        Convert input to DataFrame if not already.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X
