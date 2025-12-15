from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .utils import *


class FuzzyGranularitySelector(BaseEstimator, TransformerMixin):
    """
    Fuzzy-Implication Granularity Feature Selector (FIGFS).

    Implements a fuzzy-implication-driven information granularity algorithm
    for selecting the most informative feature subset. The method evaluates
    both global and local granularity consistency, fuzzy neighbourhood
    structure, and multi-level implication entropy. Compatible with
    scikit-learn pipelines.

    Parameters
    ----------
    n_features : int, default=3
        Number of features to keep in the transformed dataset.
        Must be >= 1 and `n_features <= max_features`.

    eps : {int, float}, default=0.5
        Normalization factor controlling the fuzzy neighbourhood radius
        for numeric features. Must be strictly positive.
        Must be > 0.

    max_features : int, default=10
        Maximum number of features that FIGFS is allowed to consider
        during the iterative selection process.
        Must be >= 1.

    random_state : {int, None}, default=None
        Seed for reproducibility of internal stochastic components.
        If None, randomness is not fixed.

    Attributes
    ----------
    S_ : list of str
        Ordered list of selected features after fitting. The order
        reflects FIGFS importance ranking.

    U_ : DataFrame
        Internal working dataset combining input `X` and the target,
        used during granularity calculations.

    C_ : dict
        Mapping of feature names to types ('numeric' or 'nominal').

    D_ : dict
        Mapping describing the target variable type.

    similarity_matrices_ : dict of ndarray
        Precomputed fuzzy similarity matrices for each feature.

    fuzzy_adaptive_neighbourhood_radius_ : dict
        Radius values used for fuzzy similarity truncation for numeric features.

    D_partition_ : dict
        Partition of the dataset induced by target values (one subset per class).

    delta_cache_ : dict
        Cache storing granule membership vectors and sizes.

    global_row_tuple_to_index_cache_ : dict
        Cache storing row indexes

    Examples
    --------
    >>> selector = FuzzyGranularitySelector(n_features=5, eps=0.3)
    >>> selector.fit(X_train, y_train)
    >>> X_reduced = selector.transform(X_test)
    """

    def __init__(self, n_features=3, eps=0.5, max_features=10, random_state=None):

        validate_params({
            'n_features': n_features,
            'eps': eps,
            'max_features': max_features,
            'random_state': random_state
        })

        if n_features > max_features:
            raise ValueError(f"n_features must be <= max_features: {max_features}, got {n_features} instead")

        self.k = n_features
        self.eps = float(eps)
        self.d = int(max_features)
        self.random_state = random_state

        self.S_: Optional[List[str]] = None

        self.U_: Optional[pd.DataFrame] = None
        self.delta_cache_: Dict[Any, Any] = {}
        self.D_: Tuple[str, str] = ()
        self.n_: int = 0
        self.m_: int = 0
        self.target_name_: str = "target"
        self.similarity_matrices_: Dict[str, np.ndarray] = {}
        self.D_partition_: Dict[Any, pd.DataFrame] = {}
        self.C_: Dict[str, str] = {}

    def fit(self, X, y=None):
        """
        Fit the FIGFS selector on the input dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input dataset to fit the selector.

        y : array-like of shape (n_samples,), default=None
            Target feature. If None, an unsupervised mode is used where all samples
            are assigned to a single dummy class.

        Returns
        -------
        self : FuzzyGranularitySelector
            Fitted selector with the learned feature ordering available
            in `self.S_`.
        """

        X = check_input_dataset(X, allow_nan=False)

        if y is not None and isinstance(y, (np.ndarray, pd.Series, pd.DataFrame)) and len(y) != len(X):
            raise ValueError(f"y must have the same number of rows as X: ({len(X)}), got {len(y)} instead")

        i = 1
        while self.target_name_ in X.columns:
            self.target_name_ = f"target_{i}"
            i += 1

        if y is None:
            y_ser = pd.Series(np.zeros(len(X), dtype=int), name=self.target_name_)
        elif isinstance(y, pd.DataFrame):
            y_ser = y.iloc[:, 0]
        else:
            y_ser = pd.Series(y).reset_index(drop=True)
            y_ser.name = self.target_name_

        self.U_ = X.copy()
        self.U_[self.target_name_] = y_ser.values
        self.n_ = len(self.U_)

        self.C_ = {col: "numeric" if pd.api.types.is_numeric_dtype(X[col]) else "nominal" for col in X.columns}
        self.m_ = len(self.C_)

        self.D_ = {
            self.target_name_: "numeric" if pd.api.types.is_numeric_dtype(self.U_[self.target_name_]) else "nominal"}

        self.fuzzy_adaptive_neighbourhood_radius_ = {}
        for col_name, col_type in {**self.C_, **self.D_}.items():
            if col_type == "numeric":
                std_val = float(self.U_[col_name].std(ddof=0))
                self.fuzzy_adaptive_neighbourhood_radius_[col_name] = std_val / self.eps if self.eps != 0 else 0.0
            else:
                self.fuzzy_adaptive_neighbourhood_radius_[col_name] = None

        self.delta_cache_ = {}
        self.entropy_cache_ = {}
        self.D_partition_ = self._create_partitions()
        self.global_row_tuple_to_index_cache_ = {}

        for col in self.U_.columns:
            self.similarity_matrices_[col] = self._calculate_similarity_matrix_for_df(col, self.U_)

        global_vals = self.U_.values
        for gi in range(self.n_):
            tup = tuple(global_vals[gi])
            if tup not in self.global_row_tuple_to_index_cache_:
                self.global_row_tuple_to_index_cache_[tup] = gi

        self.S_ = self._FIGFS_algorithm()
        return self

    def transform(self, X):
        """
        Transform the input dataset using the selected FIGFS feature subset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input dataset. Must contain the same columns and feature order
            as the data used during `fit()`.

        Returns
        -------
        X_transformed : DataFrame of shape (n_samples, k)
            Dataset reduced to the selected `k` most informative features.
        """
        check_is_fitted(self, attributes=["U_", "n_", "C_", "m_", "D_", "fuzzy_adaptive_neighbourhood_radius_",
                                          "delta_cache_", "entropy_cache_", "D_partition_",
                                          "global_row_tuple_to_index_cache_", "S_"])

        X = check_input_dataset(X, allow_nan=False)

        if list(X.columns) != list(self.C_.keys()):
            raise ValueError(f"X.columns must match the columns seen during fit {list((self.C_.keys()))}, "
                             f"got {list(X.columns)} instead")

        X_transformed = X.copy()
        final_cols = self.S_[:self.k]

        return X_transformed[final_cols].copy()

    def _calculate_similarity_matrix_for_df(self, colname, df):
        """
        Compute fuzzy similarity matrix for a single column (numeric or categorical),
        working correctly in both global and local contexts.

        Parameters
        ----------
        colname : str
            Column name (can refer to position in df or in self.C).
        df : pd.DataFrame
            DataFrame containing the data (global or local context).

        Returns
        -------
        np.ndarray
            n x n fuzzy similarity matrix.
        """
        if colname in self.C_:
            col_type = self.C_[colname]

        elif colname in self.D_:
            col_type = self.D_[colname]

        vals = df[colname].values
        n = len(df)
        if n == 0:
            return np.zeros((0, 0), dtype=float)

        if col_type == 'numeric':
            if n == 1:
                sd = 0.0
            else:
                sd = float(df[colname].std(ddof=0))
            denom = 1.0 + sd

            radius = self.fuzzy_adaptive_neighbourhood_radius_.get(colname, None)

            diffs = np.abs(vals[:, None] - vals[None, :])
            sim = 1.0 - (diffs / denom)
            np.clip(sim, 0.0, 1.0, out=sim)

            if radius is None:
                return sim
            else:
                thresh = 1.0 - radius
                mask = sim >= thresh
                mat = sim * mask
                return mat
        else:
            codes, _ = pd.factorize(vals, sort=True)
            mat = (codes[:, None] == codes[None, :]).astype(float)
            return mat

    def _calculate_delta_for_B_all_rows(self, B: List[str], df: Optional[pd.DataFrame] = None):
        """
        Calculate the granule matrix and row-wise granule sizes for a subset of columns.

        Parameters
        ----------
        B : List[str]
            List of column names to include in the granule computation.
        df : Optional[pd.DataFrame]
            Local DataFrame context. If None, uses the global dataset (`self.U_`).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            granule_matrix : np.ndarray
                Pairwise similarity matrix of shape (n_rows, n_rows), representing similarities
                between all rows across columns in B.
            sizes : np.ndarray
                Array of row-wise granule sizes (sum of similarities per row).
        """
        use_global = df is None
        if use_global:
            n = self.n_
        else:
            n = len(df)

        if n == 0 or len(B) == 0:
            return np.zeros((n, n), dtype=float), np.zeros((n,), dtype=float)

        first_mat = None
        for colname in B:
            if colname == self.target_name_:
                vals = (self.U_[colname].values if use_global else df[colname].values)
                mat = (vals[:, None] == vals[None, :]).astype(float)
            else:
                if use_global:
                    mat = self.similarity_matrices_.get(colname)
                    if mat is None:
                        mat = self._calculate_similarity_matrix_for_df(colname, self.U_)
                        self.similarity_matrices_[colname] = mat
                else:
                    mat = self._calculate_similarity_matrix_for_df(colname, df)

            if first_mat is None:
                first_mat = mat.copy()
            else:
                np.minimum(first_mat, mat, out=first_mat)

        granule_matrix = first_mat if first_mat is not None else np.zeros((n, n), dtype=float)
        sizes = np.sum(granule_matrix, axis=1).astype(float)

        if use_global:
            for i in range(n):
                self.delta_cache_[i] = (granule_matrix[i, :].astype(float), float(sizes[i]))

        return granule_matrix, sizes

    def _calculate_delta_for_column_subset(self, row_index, B, df=None):
        """
        Calculate granule membership vector and size for a given row and subset of features.

        Parameters
        ----------
        row_index : int
            Row index in the DataFrame.
        B : List[str]
            List of column names representing feature subset.
        df : Optional[pd.DataFrame]
            Local DataFrame context. If None, use global self.U.

        Returns
        -------
        Tuple[np.ndarray, float]
            Tuple containing granule_vector and size
        """
        if df is None:
            if row_index in self.delta_cache_:
                return self.delta_cache_[row_index]
            else:
                _, sizes = self._calculate_delta_for_B_all_rows(list(B), df=None)
                vec, size = self.delta_cache_.get(row_index)
                return vec.astype(float), float(size)
        else:
            if not np.array_equal(df.index.values, np.arange(len(df))):
                df_local = df.reset_index(drop=True)
            else:
                df_local = df
            granule_matrix, sizes = self._calculate_delta_for_B_all_rows(list(B), df=df_local)
            return granule_matrix[row_index, :].astype(float), float(sizes[row_index])

    def _calculate_multi_granularity_fuzzy_implication_entropy(self, B, type='basic', T=None):
        """
        Measure the uncertainty or fuzziness of information granules
        formed by a subset of features B, optionally conditioned on another subset T.

        Parameters
        ----------
        B : List[str]
            Feature subset columns.
        type : str
            Entropy type ('basic', 'conditional', 'joint', 'mutual').
        T : Optional[List[str]]
            Optional secondary feature subset for conditional/mutual entropy.

        Returns
        -------
        float
            Entropy value of the subset.
        """

        B_tuple = tuple(B) if B is not None else ()
        T_tuple = tuple(T) if T is not None else ()

        key = (B_tuple, type, T_tuple)
        if key in self.entropy_cache_:
            return self.entropy_cache_[key]

        if len(B_tuple) == 0:
            self.entropy_cache_[key] = 0.0
            return 0.0

        _, delta_B_sizes = self._calculate_delta_for_B_all_rows(list(B_tuple), df=None)
        if len(T_tuple) > 0:
            _, delta_T_sizes = self._calculate_delta_for_B_all_rows(list(T_tuple), df=None)
        else:
            delta_T_sizes = np.zeros_like(delta_B_sizes)

        n = max(self.n_, 1.0)
        if type == 'basic':
            res_vec = 1.0 - (delta_B_sizes / n)
            out = float(np.sum(res_vec) / n)
        elif type == 'conditional':
            res_vec = np.maximum(delta_B_sizes, delta_T_sizes) - delta_B_sizes
            out = float(np.sum(res_vec) / (self.n_ ** 2 if self.n_ > 0 else 1.0))
        elif type == 'joint':
            res_vec = 1.0 + (np.maximum(delta_B_sizes, delta_T_sizes) / n) - ((delta_B_sizes + delta_T_sizes) / n)
            out = float(np.sum(res_vec) / n)
        else:
            res_vec = 1.0 - (np.maximum(delta_B_sizes, delta_T_sizes) / n)
            out = float(np.sum(res_vec) / n)

        self.entropy_cache_[key] = out
        return out

    def _granular_consistency_of_B_subset(self, B):
        """
        Measure how well a subset of features B preserves the structure of the target variable D in terms of
        fuzzy information granules.

        Parameters
        ----------
        B : list
            List of feature names representing the subset B.

        Returns
        -------
        float
            Granularity consistency score in the range [0,1], where 1 indicates perfect
            consistency (granules align exactly with the target classes) and 0 indicates
            maximum inconsistency.
        """

        if len(B) == 0:
            return 0.0

        granule_matrix, _ = self._calculate_delta_for_B_all_rows(B, df=None)
        y_vals = self.U_[self.target_name_].values
        target_mat = (y_vals[:, None] == y_vals[None, :]).astype(float)
        diff_mat = np.maximum(0.0, granule_matrix - target_mat) + np.maximum(0.0, target_mat - granule_matrix)
        diff_norm = np.sum(diff_mat, axis=1) / self.n_
        score_vec = 1.0 - diff_norm
        return float(np.mean(score_vec))

    def _local_granularity_consistency_of_B_subset(self, B):
        """
        Evaluates how consistent the fuzzy granules of B are within each
        class-specific partition of the dataset.

        Parameters
        ----------
        B : List[str]
            List of feature subset columns.

        Returns
        -------
        float
            Average local granularity consistency across all partitions.
        """

        if len(B) == 0:
            return 0.0

        total = 0.0
        for key, df_part in self.D_partition_.items():
            df_local = df_part.reset_index(drop=True)
            part_n = len(df_local)
            if part_n == 0:
                continue

            _, delta_df_sizes = self._calculate_delta_for_B_all_rows(B, df=df_local)

            local_vals = df_local.values
            ratios = np.empty(part_n, dtype=float)
            for i_local in range(part_n):
                tup = tuple(local_vals[i_local])
                if tup not in self.global_row_tuple_to_index_cache_:
                    ratio = 1.0
                else:
                    global_idx = self.global_row_tuple_to_index_cache_[tup]
                    _, delta_U_size = self._calculate_delta_for_column_subset(int(global_idx), B, df=None)
                    delta_df_size = float(delta_df_sizes[i_local])
                    if delta_U_size == 0.0:
                        ratio = 1.0
                    else:
                        ratio = delta_df_size / delta_U_size
                ratios[i_local] = ratio
            total += float(np.mean(ratios))
        return float(total / len(self.D_partition_)) if len(self.D_partition_) > 0 else 0.0

    def _create_partitions(self):
        """
        Partition the dataset into subsets according to target values.

        Returns
        -------
        Dict[Any, pd.DataFrame]
            Dictionary mapping each target class value to a sub-DataFrame
            containing only the objects belonging to that class.
        """
        partitions = {}
        vals = self.U_[self.target_name_].unique()
        for v in vals:
            partitions[v] = self.U_[self.U_[self.target_name_] == v].reset_index(drop=True).copy()
        return partitions

    def _FIGFS_algorithm(self):
        """
        Execute the Fuzzy Implication Granularity-based Feature Selection (FIGFS) algorithm.

        FIGFS iteratively selects features that maximize granularity consistency
        and minimize redundancy.

        Returns
        -------
        List[str]
            Ordered list of selected feature cnames according to the FIGFS algorithm.
            The order reflects the importance of the features.
        """
        B = list(self.C_.keys())
        S = []
        cor_list = []
        for colname in B:
            cor = self._granular_consistency_of_B_subset([colname]) + self._local_granularity_consistency_of_B_subset(
                [colname])
            cor_list.append((colname, cor))

        c1 = max(cor_list, key=lambda x: x[1])[0]

        S.append(c1)
        B.remove(c1)
        i = 1

        while len(B) > 0 and i < self.d:
            i += 1
            J_list = []
            denom_base = self._calculate_multi_granularity_fuzzy_implication_entropy(S, type='conditional',
                                                                                     T=[self.target_name_]) + 0.01
            for colname in B:
                sim = 0
                for s_colname in S:
                    fimi_d_cv = self._calculate_multi_granularity_fuzzy_implication_entropy([self.target_name_],
                                                                                            type='mutual',
                                                                                            T=[colname])
                    fimi_cv_cu = self._calculate_multi_granularity_fuzzy_implication_entropy([colname],
                                                                                             type='mutual',
                                                                                             T=[s_colname])
                    fimi_cd = self._calculate_multi_granularity_fuzzy_implication_entropy([colname], type='mutual',
                                                                                          T=[self.target_name_,
                                                                                             s_colname])
                    sim += fimi_d_cv + fimi_cv_cu - fimi_cd
                sim = sim / len(S)

                l = S + [colname]
                W = 1 + (self._calculate_multi_granularity_fuzzy_implication_entropy(S, type='conditional', T=[
                    self.target_name_]) - self._calculate_multi_granularity_fuzzy_implication_entropy(S,
                                                                                                      type='conditional',
                                                                                                      T=l)) / denom_base
                cor = self._granular_consistency_of_B_subset(
                    [colname]) + self._local_granularity_consistency_of_B_subset([colname])
                j = W * cor - sim
                J_list.append((colname, j))
            cv = max(J_list, key=lambda x: x[1])[0]

            S.append(cv)
            B.remove(cv)
            print(S)
        return S


# --------------------------------------
# WeightedFuzzyRoughSelector
# --------------------------------------
class WeightedFuzzyRoughSelector(BaseEstimator, TransformerMixin):
    """
    Feature selector based on weighted fuzzy-rough sets and density-based sample selection.

    This estimator computes fuzzy similarity relations for individual features
    and feature subsets using a hybrid distance measure, evaluates relevance
    and redundancy, computes feature weights, and builds a greedy weighted
    feature ranking. It supports numerical, categorical, and mixed data, as
    well as missing values, and incorporates a density-based region H to
    improve robustness to noise and outliers.

    Parameters
    ----------
    n_features : int
        Number of top-ranked features to retain after selection.
        Must be >= 1.

    alpha : {int, float}, default=0.5
        Smoothing parameter used in the fuzzy similarity kernel
        Must be in range (0, 1].

    k : int, default=5
        Number of nearest neighbors used during density estimation.
        Must be > 1.

    Attributes
    ----------
    W_ : ndarray of shape (n_features_total, n_features_total)
        Final diagonal weight matrix computed from normalized feature weights.

    selected_features_ : list of int
        Indices of selected features (after transform).

    feature_importances_ : pd.DataFrame
        Sorted feature ranking containing original feature names and importance scores.

    feature_sequence_ : list of int
        Full greedy ranking of all features produced during fit.

    distance_cache_ : dict
        Cache storing computed pairwise distance matrices to avoid recomputation.

    Rw_ : ndarray of shape (n_features_total, n_features_total)
        Diagonal matrix representing weighted feature relevance after logistic scaling.

    Examples
    --------
    >>> selector = WeightedFuzzyRoughSelector(n_features=5)
    >>> selector.fit(X_train, y_train)
    >>> X_reduced = selector.transform(X_test)
    """

    def __init__(self, n_features, alpha=0.5, k=5):

        validate_params({
            'n_features': n_features,
            'k': k
        })

        if not isinstance(alpha, (int, float)):
            raise TypeError(f"alpha must be int or float, got {type(alpha).__name__} instead")
        if not (0 < alpha <= 1):
            raise ValueError(f"alpha must be in range (0, 1], got {alpha} instead")

        self.n_features = n_features
        self.alpha = alpha
        self.k = k

        self.W_ = None
        self.selected_features_ = None
        self.feature_importances_ = None
        self.feature_sequence_ = None
        self.distance_cache_ = {}
        self.Rw_ = None

    def fit(self, X, y):
        """
        Fit the fuzzy-rough feature selector on the input dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_original)
            Input dataset to fit the selector.

        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : WeightedFuzzyRoughSelector
            Fitted selector ready for feature transformation.
        """

        if self.n_features > X.shape[1]:
            raise ValueError(
                f"n_features must be ≤ number of columns in X: ({X.shape[1]}), got {self.n_features} instead")

        if self.k >= len(X):
            self.k = len(X) - 1

        X = check_input_dataset(X)
        self.feature_names_in_ = list(X.columns)

        if not isinstance(y, (pd.Series, np.ndarray, list)):
            raise TypeError(f"y must be a pandas Series, numpy array, or list, got {type(y).__name__} instead")

        y = pd.Series(y)
        y = y.reset_index(drop=True)

        if y.isna().any():
            raise ValueError("y must not contain missing values")

        if len(y) != len(X):
            raise ValueError(f"y must have the same number of rows as X ({len(X)}), got {len(y)} rows instead")

        H = self._identify_high_density_region(X, y)

        relations_single, relations_pair = self._compute_fuzzy_similarity_relations(X, H)
        relevance = self._compute_relevance(relations_single, y, H)
        redundancy = self._compute_redundancy(y, H, relevance, relations_pair)
        weights = self._compute_feature_weights(relevance, redundancy)
        self.W_ = self._update_weight_matrix(weights, X.shape[1])

        self.feature_sequence_, self.Rw_ = self._build_weighted_feature_sequence(relations_single, relations_pair, X, y,
                                                                                 H)

        self.feature_importances_ = pd.DataFrame({
            'feature': X.columns[self.feature_sequence_],
            'importance': np.diag(self.Rw_)[:len(self.feature_sequence_)]
        }).sort_values(by='importance', ascending=False).reset_index(drop=True)

        return self

    def transform(self, X):
        """
        Reduce the input dataset to the top `n_features` selected during fitting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features_original)
            Input dataset containing the same columns and order as during `fit`.

        Returns
        -------
        X_transformed : pd.DataFrame of shape (n_samples, n_features)
            Dataset restricted to the highest-ranked features.
        """

        check_is_fitted(self,
                        attributes=["feature_names_in_", "W_", "feature_sequence_", "Rw_", "feature_importances_"])

        X = check_input_dataset(X)

        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError(f"X.columns must match the columns seen during fit {list(self.feature_names_in_)}, "
                             f"got {list(X.columns)} instead")

        selected_idx = self.feature_sequence_[:self.n_features]
        return X.iloc[:, selected_idx]

    def _identify_high_density_region(self, X, y):
        """
        Identify high-density regions within a dataset based on local density and class labels.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray or pd.Series
            Class labels corresponding to each row in X.
        """
        distances = self._compute_HEC(X)

        n_samples = len(X)
        y = np.asarray(y)

        knn_indices = np.full((n_samples, self.k), -1, dtype=int)

        for i in range(n_samples):
            same_class_idx = np.where(y == y[i])[0]
            same_class_idx = same_class_idx[same_class_idx != i]

            if len(same_class_idx) == 0:
                continue

            ordered = same_class_idx[np.argsort(distances[i, same_class_idx])]

            take = min(len(ordered), self.k)
            if take > 0:
                knn_indices[i, :take] = ordered[:take]

        rho = self._compute_density(distances, knn_indices)
        LDF = self._compute_LDF(rho, knn_indices)

        H_indices = np.where(LDF <= 1)[0]
        H_neighbors = np.unique(knn_indices[H_indices].flatten())
        return H_neighbors

    def _compute_HEC(self, X1, X2=None, W=None):
        """
        Compute Hybrid Mahalanobis (HM) distance supporting numerical, categorical and mixed features including missing values

        Parameters:
            X1 (pd.DataFrame): input data frame containing mixed feature types
            X2 (pd.DataFrame or None): optional second data frame; if None, X1 is used
            W (np.ndarray or None): diagonal weight matrix applied to features; if None, identity matrix is used
        """

        if X2 is None:
            X2 = X1

        key = (tuple(X1.columns), tuple(X2.columns))
        if key in self.distance_cache_:
            return self.distance_cache_[key]

        num_cols = X1.select_dtypes(include=[np.number]).columns
        cat_cols = X1.select_dtypes(exclude=[np.number]).columns

        X1_num = X1[num_cols].to_numpy(dtype=float, copy=False)
        X2_num = X2[num_cols].to_numpy(dtype=float, copy=False)

        mask1_num = np.isnan(X1_num)
        mask2_num = np.isnan(X2_num)

        n_features = X1.shape[1]
        if W is None:
            W = np.eye(n_features)
        else:
            W = np.asarray(W)

        n1, n2 = X1.shape[0], X2.shape[0]
        distances = np.zeros((n1, n2), dtype=float)

        if len(num_cols) > 0:
            diff_num = np.abs(X1_num[:, None, :] - X2_num[None, :, :])
            missing_mask = mask1_num[:, None, :] | mask2_num[None, :, :]
            diff_num[missing_mask] = 1.0

        if len(cat_cols) > 0:
            X1_cat = X1[cat_cols].astype(str).to_numpy(copy=False)
            X2_cat = X2[cat_cols].astype(str).to_numpy(copy=False)
            diff_cat = (X1_cat[:, None, :] != X2_cat[None, :, :]).astype(float)
            mask_cat_missing = np.logical_or(pd.isna(X1[cat_cols].values[:, None, :]),
                                             pd.isna(X2[cat_cols].values[None, :, :]))
            diff_cat[mask_cat_missing] = 1.0

        else:
            diff_cat = np.zeros((n1, n2, 0))

        diff = np.concatenate([diff_num if len(num_cols) > 0 else np.zeros((n1, n2, 0)),
                               diff_cat], axis=2)

        if np.allclose(W, np.diag(np.diag(W))):
            weights = np.diag(W)
            distances = np.sqrt(np.tensordot(diff ** 2, weights, axes=(2, 0)))
        else:
            tmp = np.tensordot(diff, W, axes=(2, 0))
            distances = np.sqrt(np.sum(tmp * diff, axis=2))

        self.distance_cache_[key] = distances

        return distances

    def _compute_density(self, distances, knn_indices):
        """
        Compute local density rho(x) for each sample based on distances and n_features-nearest neighbors

        Parameters:
            distances (np.ndarray): pairwise distance matrix between all samples
            knn_indices (np.ndarray): matrix of neighbor indices for each sample
        """
        n_samples = distances.shape[0]
        rho = np.zeros(n_samples)
        for i in range(n_samples):
            neighbors = knn_indices[i]
            neighbors = neighbors[neighbors != -1]
            if len(neighbors) == 0:
                rho[i] = 0
            else:
                rho[i] = (1 + len(neighbors)) / (1 + np.sum(distances[i, neighbors]))
        return rho

    def _compute_LDF(self, rho, knn_indices):
        """
        Compute Local Density Factor (LDF) for each sample using density ratios of neighbors

        Parameters:
            rho (np.ndarray): density values for all samples
            knn_indices (np.ndarray): matrix with n_features-nearest neighbor indices
        """

        n_samples = len(rho)
        LDF = np.zeros(n_samples)
        for i in range(n_samples):
            neighbors = knn_indices[i]
            neighbors = neighbors[neighbors != -1]

            if len(neighbors) == 0:
                LDF[i] = np.inf
            else:
                LDF[i] = np.mean(rho[neighbors] / rho[i])
        return LDF

    def _compute_fuzzy_similarity_relations(self, X, H, W=None):
        """
        Compute fuzzy similarity relations RM_B(x,y) for single features and feature pairs with respect to region H

        Parameters:
            X (pd.DataFrame): input dataset with mixed features
            H (array-like): indices of high-density samples
            W (np.ndarray or None): diagonal weight matrix used for weighted distances
        """

        n_samples, n_features = X.shape
        feature_indices = list(range(n_features))
        relations_single = {}
        relations_pair = {}

        X_H = X.iloc[H]

        for a in feature_indices:
            X_a = X.iloc[:, [a]]
            X_H_a = X_H.iloc[:, [a]]
            dist_a = self._compute_HEC(X_a, X_H_a, W=W[np.ix_([a], [a])] if W is not None else None)
            relations_single[a] = np.exp(- (dist_a ** 2) / (2 * self.alpha ** 2))

        for i, a in enumerate(feature_indices):
            for b in feature_indices[i + 1:]:
                X_ab = X.iloc[:, [a, b]]
                X_H_ab = X_H.iloc[:, [a, b]]
                dist_ab = self._compute_HEC(X_ab, X_H_ab, W=W[np.ix_([a, b], [a, b])] if W is not None else None)
                relations_pair[(a, b)] = np.exp(- (dist_ab ** 2) / (2 * self.alpha ** 2))

        return relations_single, relations_pair

    def _compute_relation_for_subset(self, X, H, feature_subset, W=None):
        """
        Compute fuzzy relation for an arbitrary subset of features

        Parameters:
            X (pd.DataFrame): input dataset
            H (array-like): indices of high-density samples
            feature_subset (list): selected feature indices
            W (np.ndarray or None): weight matrix corresponding to the subset
        """

        X_H = X.iloc[H, feature_subset]
        X_sub = X.iloc[:, feature_subset]
        dist = self._compute_HEC(X_sub, X_H, W=W)

        if dist.ndim == 1:
            dist = dist[:, np.newaxis]

        relation = np.exp(- (dist ** 2) / (2 * self.alpha ** 2))
        return relation

    def _compute_POS_NOG_B(self, R_B, y, H):
        """
        Compute POS^B and NOG^B for a fuzzy relation matrix R_B

        Parameters:
            R_B (np.ndarray): fuzzy relation matrix of shape (n × |H|)
            y (array-like): class labels for samples
            H (array-like): indices of high-density region samples
        """

        n = len(y)
        classes = np.unique(y)

        y_arr = np.asarray(y)
        H = np.asarray(H, dtype=int)

        if R_B.shape[1] != len(H):
            R_B = R_B[:, H]

        # Fuzzy decision memberships for all classes (hard labels -> crisp membership)
        # D_i(y) = 1 if y==class_i else 0
        DI = {c: (y_arr[H] == c).astype(float) for c in classes}

        POS = np.zeros(n)
        NOG = np.zeros(n)

        for idx_x in range(n):
            R_xH = R_B[idx_x]

            lower_vals = []
            upper_vals = []

            for c in classes:
                D_i = DI[c]

                lower = np.min(np.maximum(1 - R_xH, D_i))
                upper = np.max(np.minimum(R_xH, D_i))

                lower_vals.append(lower)
                upper_vals.append(upper)

            POS[idx_x] = np.max(lower_vals)
            NOG[idx_x] = np.max(upper_vals)

        return POS, NOG

    def _compute_relevance_B(self, R_B, y, H):
        """
        Compute relevance Rel(B) for a feature subset using POS and NOG distributions

        Parameters:
            R_B (np.ndarray): fuzzy relation matrix for subset B
            y (array-like): class labels
            H (array-like): indices of high-density samples
        """

        POS_B, NOG_B = self._compute_POS_NOG_B(R_B, y, H)
        RelB = float(np.mean(POS_B + NOG_B))
        return RelB, POS_B, NOG_B

    def _compute_relevance(self, relations_single, y, H):
        """
        Compute relevance Rel(a) for each individual feature

        Parameters:
            relations_single (dict): mapping {feature_index: relation_matrix}
            y (array-like): class labels
            H (array-like): indices of high-density samples
        """

        relevance = {}

        for a, R_a in relations_single.items():
            POS_a, NOG_a = self._compute_POS_NOG_B(R_a, y, H)

            gamma_P = POS_a.mean()
            gamma_N = NOG_a.mean()

            relevance[a] = gamma_P + gamma_N

        return relevance

    def _compute_redundancy(self, y, H, relevance, relations_pair, cached_REL_pairs=None):
        """
        Compute redundancy Red(a,b) between feature pairs

        Parameters:
            y (array-like): class labels
            H (array-like): indices of high-density samples
            relevance (dict): relevance values for single features
            relations_pair (dict): fuzzy relations for feature pairs
            cached_REL_pairs (dict or None): optional cache for Rel({a,b})
        """

        if cached_REL_pairs is None:
            cached_REL_pairs = {}

        redundancy = {}
        features = list(relevance.keys())

        for (a, b), rel_matrix in relations_pair.items():
            if a in features and b in features:
                key = (min(a, b), max(a, b))
                if key in cached_REL_pairs:
                    Rel_ab = cached_REL_pairs[key]
                else:
                    Rel_ab, _, _ = self._compute_relevance_B(rel_matrix, y, H)
                    cached_REL_pairs[key] = Rel_ab
                redundancy[(min(a, b), max(a, b))] = relevance[a] + relevance[b] - Rel_ab

        return redundancy

    def _compute_feature_weights(self, relevance, redundancy):
        """
        Compute feature weights using normalized relevance and redundancy

        Parameters:
            relevance (dict): relevance scores Rel(a)
            redundancy (dict): redundancy values Red(a,b)
        """

        features = sorted(list(relevance.keys()))
        m = len(features)

        rel_vals = np.array([relevance[a] for a in features], dtype=float)
        rel_min, rel_max = rel_vals.min(), rel_vals.max()
        denom_rel = (rel_max - rel_min) if (rel_max - rel_min) > 0 else 1.0
        NRel = {a: (relevance[a] - rel_min) / denom_rel for a in features}

        if len(redundancy) > 0:
            red_vals = np.array(list(redundancy.values()), dtype=float)
            red_min, red_max = red_vals.min(), red_vals.max()
            denom_red = (red_max - red_min) if (red_max - red_min) > 0 else 1.0
            NRed = {k: (v - red_min) / denom_red for k, v in redundancy.items()}
        else:
            NRed = {}

        weights = {}
        for a in features:
            sum_NRred = 0.0
            for b in features:
                if b == a:
                    continue
                key = (min(a, b), max(a, b))
                sum_NRred += NRed.get(key, 0.0)
            denom = max(m - 1, 1)
            weights[a] = NRel[a] - (sum_NRred / denom)

        return weights

    def _update_weight_matrix(self, weights, n_total_features):
        """
        Update diagonal weight matrix W for all original features

        Parameters:
            weights (dict): feature weights w(a)
            n_total_features (int): number of total features in X
        """

        W = np.zeros((n_total_features, n_total_features))
        for a, w_a in weights.items():
            W[a, a] = 1 / (1 + np.exp(-w_a)) ** 2
        return W

    def _compute_gamma(self, POS_all, NOG_all, features):
        """
        Compute gamma_P and gamma_N for the current subset of features

        Parameters:
            POS_all (dict): mapping from feature to POS distributions
            NOG_all (dict): mapping from feature to NOG distributions
            features (list): selected feature indices
        """

        POS_mean = np.mean([POS_all[a] for a in features], axis=0)
        NOG_mean = np.mean([NOG_all[a] for a in features], axis=0)
        gamma_P = np.mean(POS_mean)
        gamma_N = np.mean(NOG_mean)
        return gamma_P, gamma_N

    def _compute_separability(self, X, y, H, W, selected_features, remaining_features):
        """
        Compute separability measure sig(a, B, D) for each candidate feature

        Parameters:
            X (pd.DataFrame): input dataset
            y (array-like): class labels
            H (array-like): indices of high-density samples
            W (np.ndarray): global weight matrix
            selected_features (list): features already selected
            remaining_features (list): features remaining to evaluate
        """

        if len(selected_features) == 0:
            Rel_B = 0.0
        else:
            W_sub = self.W_[np.ix_(selected_features, selected_features)]
            R_B = self._compute_relation_for_subset(X, H, selected_features, W=W_sub)
            Rel_B, _, _ = self._compute_relevance_B(R_B, y, H)

        separability = {}
        for a in remaining_features:
            B_union_a = selected_features + [a]
            W_sub_a = self.W_[np.ix_(B_union_a, B_union_a)]
            R_Ba = self._compute_relation_for_subset(X, H, B_union_a, W=W_sub_a)
            Rel_Ba, _, _ = self._compute_relevance_B(R_Ba, y, H)
            separability[a] = Rel_Ba - Rel_B

        return separability

    def _build_weighted_feature_sequence(self, relations_single, relations_pair, X, y, H):
        """
        Build weighted feature ranking using greedy selection based on separability

        Parameters:
            relations_single (dict): fuzzy relations for single features
            relations_pair (dict): fuzzy relations for feature pairs
            X (pd.DataFrame): input dataset
            y (array-like): class labels
            H (array-like): high-density region indices
        """

        n_features = X.shape[1]
        selected_features = []
        remaining = list(range(n_features))
        sequence = []

        while len(remaining) > 0:
            separability = self._compute_separability(
                X=X,
                y=y,
                H=H,
                W=self.W_,
                selected_features=selected_features,
                remaining_features=remaining
            )

            best_feature = max(separability, key=separability.get)
            selected_features.append(best_feature)
            remaining.remove(best_feature)
            sequence.append(best_feature)

        logistic_weights = [self.W_[f, f] for f in sequence]
        Rw = np.diag(logistic_weights)

        return sequence, Rw


__all__ = [
    "FuzzyGranularitySelector",
    "WeightedFuzzyRoughSelector",
]
