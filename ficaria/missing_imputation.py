from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
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

    def __init__(self, random_state: Optional[int] = None, max_iter: int = 30):
        if random_state is not None and not isinstance(random_state, int):
            raise TypeError('Invalid random_state: Expected an integer or None')
        if not isinstance(max_iter, int) or max_iter <= 1:
            raise TypeError('Invalid max_iter: Expected a positive integer')

        self.random_state = random_state
        self.max_iter = max_iter
        pass

    def fit(self, X, y=None):
        X = check_input_dataset(X)
        self.X_train_ = X.copy()
        self.np_rng_ = np.random.RandomState(self.random_state)
        return self

    def transform(self, X):
        X = check_input_dataset(X)
        check_is_fitted(self, attributes=["X_train_", "np_rng_"])
        if not X.columns.equals(self.X_train_.columns):
            raise ValueError(
                f"Invalid input: Input dataset columns do not match columns seen during fit"
            )

        X_imputed = impute_KI(X, self.X_train_, np_rng=self.np_rng_, max_iter=self.max_iter)
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

    def __init__(self, random_state: Optional[int] = None, max_clusters: int = 10, m: float = 2, max_iter: int = 30):
        if random_state is not None and not isinstance(random_state, int):
            raise TypeError('Invalid random_state: Expected an integer or None')

        if not isinstance(max_clusters, int) or max_clusters <= 1:
            raise TypeError('Invalid max_clusters: Expected an integer greater than 1')

        if not isinstance(m, (int, float)) or m <= 1:
            raise TypeError('Invalid m value: Expected a numeric value greater than 1')

        if max_iter is not None and not isinstance(max_iter, int) or max_iter <= 1:
            raise TypeError('Invalid max_iter: Expected a positive integer greater than 1')

        self.random_state = random_state
        self.max_clusters = max_clusters
        self.m = m
        self.max_iter = max_iter
        pass

    def fit(self, X, y=None):
        X = check_input_dataset(X, require_numeric=True)
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
        X = check_input_dataset(X, require_numeric=True)
        check_is_fitted(self, attributes=["X_train_", "imputer_", "centers_", "u_", "optimal_c_", "np_rng_"])

        if not X.columns.equals(self.X_train_.columns):
            raise ValueError(
                f"Invalid input: Input dataset columns do not match columns seen during fit"
            )

        X_imputed = impute_FCKI(X, self.X_train_, self.centers_, self.u_, self.optimal_c_, self.imputer_, self.m,
                                self.np_rng_, self.random_state, max_iter=self.max_iter)
        return X_imputed