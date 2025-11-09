from collections import defaultdict

from pandas.api.types import is_numeric_dtype
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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

        self.clusters_ = rough_kmeans_from_fcm(
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


# --------------------------------------
# KIImputer
# --------------------------------------
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
            m=self.m,
            random_state=self.random_state,
        )

        return self

    def transform(self, X):
        X = check_input_dataset(X, require_numeric=True)

        X_imputed = impute_FCKI(X, self.X_train_, self.centers_, self.u_, self.optimal_c_, self.imputer_, self.m,
                                self.np_rng_, self.random_state)
        return X_imputed


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
