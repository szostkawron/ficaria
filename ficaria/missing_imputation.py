from sklearn.base import BaseEstimator, TransformerMixin
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
