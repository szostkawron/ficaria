import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Tuple, Optional, Any


# --------------------------------------
# FuzzyGranularitySelector
# --------------------------------------
class FuzzyImplicationGranularityFeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self, classifier, eps: float = 0.1, d: int = 100, sigma: int = 3):
        """
        Initialize FIGFS feature selection

        Parameters
        ----------
        classifier : sklearn-like classifier
        eps : float
            Parameter for fuzzy adaptive neighborhood radius.
        d : int
            Maximum number of features to select.
        sigma : int
            Percentile threshold for inclusion in selection.
        """
        self.d = d
        self.sigma = sigma
        self.classifier = classifier
        self.eps = eps

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | pd.DataFrame):
        """
        Fit the FIGFS algorithm on the dataset

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series, np.ndarray or pd.DataFrame
            Target variable.

        Returns
        -------
        self : object
            Fitted instance with selected feature ordering in self.S
        """
        if isinstance(y, pd.DataFrame):
            y_ser = y.iloc[:, 0]
        else:
            y_ser = pd.Series(y).reset_index(drop=True)

        self.U = X.reset_index(drop=True).copy()
        self.target_name = "___target___"
        self.U[self.target_name] = y_ser.values

        self.C = {}
        for idx, col in enumerate(X.columns):
            dtype = 'numeric' if pd.api.types.is_numeric_dtype(X[col]) else 'nominal'
            self.C[idx] = (col, dtype)

        self.D = (len(X.columns), self.target_name) 
        self.n = len(self.U)
        self.m = len(self.C)

        self.fuzzy_adaptive_neighbourhood_radius = {}
        for col_idx, (col_name, col_type) in self.C.items():
            if col_type == 'numeric':
                self.fuzzy_adaptive_neighbourhood_radius[col_idx] = float(self.U[col_name].std(ddof=0)) / self.eps
            else:
                self.fuzzy_adaptive_neighbourhood_radius[col_idx] = None

        self.similarity_matrices = {}
        for col_index in range(self.m): 
            self.similarity_matrices[col_index] = self._calculate_similarity_matrix_for_df(col_index, self.U)

        self._delta_cache = {}
        self._entropy_cache = {}

        self.D_partition = self._create_partitions()
        self.S = self._FIGFS_algorithm()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform input dataset using selected optimal feature subset.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset to transform

        Returns
        -------
        pd.DataFrame
            Dataset restricted to selected optimal features.
        """
        S_opt = None
        best_acc = -np.inf
        self.acc_list = []

        for i in range(1, len(self.S) + 1):
            current_subset = list(self.S[:i])
            cols = [self.C[idx][0] for idx in current_subset]

            X_full = self.U.drop(columns=[self.target_name])
            y_full = self.U[self.target_name]

            X_train, X_test, y_train, y_test = train_test_split(
                X_full[cols], y_full, test_size=0.3, random_state=42, stratify=y_full
            )

            num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
            cat_cols = X_train.select_dtypes(exclude=['int64', 'float64']).columns

            scaler = MinMaxScaler()
            X_train_num = pd.DataFrame(
                scaler.fit_transform(X_train[num_cols]),
                columns=num_cols,
                index=X_train.index
            )
            X_test_num = pd.DataFrame(
                scaler.transform(X_test[num_cols]),
                columns=num_cols,
                index=X_test.index
            )

            X_train_scaled = pd.concat([X_train_num, X_train[cat_cols]], axis=1)
            X_test_scaled = pd.concat([X_test_num, X_test[cat_cols]], axis=1)
            
            self.classifier.fit(X_train_scaled, y_train.values.ravel())
            y_pred = self.classifier.predict(X_test_scaled)

            acc = accuracy_score(y_test, y_pred)
            self.acc_list.append(acc)

            if acc > best_acc:
                best_acc = acc
                S_opt = current_subset

        self.S_opt = S_opt
        final_cols = [self.C[idx][0] for idx in S_opt]
        return self.U[final_cols].copy()


    def _calculate_similarity_matrix_for_df(self, col_index: int, df: pd.DataFrame) -> np.ndarray:
        """
        Compute fuzzy similarity matrix for a single column

        Parameters
        ----------
        col_index : int
            Column index in self.C.
        df : pd.DataFrame
            DataFrame containing values.

        Returns
        -------
        np.ndarray
            Similarity matrix (n x n) for the given column.
        """
        col_name, col_type = self.C[col_index]
        vals = df[col_name].values
        n = len(df)
        mat = np.zeros((n, n), dtype=float)

        if col_type == 'numeric':
            sd = float(df[col_name].std(ddof=0)) if n > 1 else 0.0
            denom = 1.0 + sd
            radius = self.fuzzy_adaptive_neighbourhood_radius.get(col_index, 0.0)
            for i in range(n):
                diff = np.abs(vals[i] - vals)
                sim = 1.0 - (diff / denom)
                if radius is None:
                    mat[i, :] = sim
                else:
                    thresh = 1.0 - radius
                    row = np.where(sim >= thresh, sim, 0.0)
                    mat[i, :] = row
        else:
            for i in range(n):
                mat[i, :] = (vals[i] == vals).astype(float)

        return mat

    def _calculate_delta_for_column_subset(self,row_index: int,B: List[int],df: Optional[pd.DataFrame] = None)-> Tuple[np.ndarray, float]:
        """
        Calculate granule membership vector and size for a given row and subset of features.

        Parameters
        ----------
        row_index : int
            Row index in the DataFrame.
        B : List[int]
            List of column indices representing feature subset.
        df : Optional[pd.DataFrame]
            Local DataFrame context. If None, use global self.U.

        Returns
        -------
        Tuple[np.ndarray, float]
            Tuple containing granule_vector and size
        """
        if df is None:
            df = self.U
            use_global = True
        else:
            df = df.reset_index(drop=True).copy()
            use_global = False

        key = (tuple(B), int(row_index), 'global' if use_global else ('local', id(df)))
        if key in self._delta_cache:
            return self._delta_cache[key]

        mats = []
        for col_index in B:
            if col_index == self.D[0]:
                y_vals = self.U[self.target_name].values
                current_class = y_vals[row_index]
                vec = (y_vals == current_class).astype(float)
            else:
                if use_global:
                    mat = self.similarity_matrices[col_index]
                    vec = mat[row_index, :].astype(float)
                else:
                    local_mat = self._calculate_similarity_matrix_for_df(col_index, df)
                    vec = local_mat[row_index, :].astype(float)
            mats.append(vec)


        if len(mats) == 0:
            granule = np.zeros(len(df), dtype=float)
        else:
            granule = np.minimum.reduce(mats)

        size = float(np.sum(granule))
        self._delta_cache[key] = (granule, size)
        return (granule, size)

    def _calculate_multi_granularity_fuzzy_implication_entropy(self, B: List[int], type: str = 'basic', T: Optional[List[int]] = None)-> float:
        """
        Measure the uncertainty or fuzziness of information granules
        formed by a subset of features B, optionally conditioned on another subset T.

        Parameters
        ----------
        B : List[int]
            Feature subset indices.
        type : str
            Entropy type ('basic', 'conditional', 'joint', 'mutual').
        T : Optional[List[int]]
            Optional secondary feature subset for conditional/mutual entropy.

        Returns
        -------
        float
            Entropy value of the subset.
        """
        B_tuple = tuple(B) if B is not None else ()
        T_tuple = tuple(T) if T is not None else ()

        key = (B_tuple, type, T_tuple)
        if key in self._entropy_cache:
            return self._entropy_cache[key]

        res = 0.0

        if len(B_tuple) == 0:
            return 0.0

        for i in range(self.n):
            delta_B_size = self._calculate_delta_for_column_subset(i, B_tuple)[1]
            delta_T_size = self._calculate_delta_for_column_subset(i, T_tuple)[1] if len(T_tuple) > 0 else 0.0

            if type == 'basic':
                res += (1.0 - delta_B_size / max(self.n, 1.0))
            elif type == 'conditional':
                res += max(delta_B_size, delta_T_size) - delta_B_size
            elif type == 'joint':
                res += 1.0 + max(delta_B_size, delta_T_size) / max(self.n,1.0) - (delta_B_size + delta_T_size) / max(self.n,1.0)
            else:
                res += 1.0 - max(delta_B_size, delta_T_size) / max(self.n,1.0)

        if type == 'conditional':
            out = res / (self.n ** 2 if self.n > 0 else 1.0)
        else:
            out = res / max(self.n, 1.0)

        self._entropy_cache[key] = out
        return out
    
    def _granual_consistency_of_B_subset(self, B: list) -> float:
        """
        Measure how well a subset of features B preserves the structure of the target variable D in terms of fuzzy information granules.

        Parameters
        ----------
        B : list
            List of feature indices representing the subset B.

        Returns
        -------
        float
            Granularity consistency score in the range [0,1], where 1 indicates perfect
            consistency (granules align exactly with the target classes) and 0 indicates
            maximum inconsistency.
        """
        total = 0.0
        y_vals = self.U.iloc[:, self.D[0]].values
        
        for i in range(self.n):
            delta_b_vec = np.array(self._calculate_delta_for_column_subset(i, B)[0])
            
            if np.issubdtype(y_vals.dtype, np.number):
                target_vec = np.zeros(self.n)
                target_vec[i] = 1
            else:
                target_vec = (y_vals == y_vals[i]).astype(float)
            
            delta_B_minus_D = np.maximum(0, delta_b_vec - target_vec)
            D_minus_delta_B = np.maximum(0, target_vec - delta_b_vec)
            
            diff_norm = np.sum(delta_B_minus_D + D_minus_delta_B) / self.n
            score_i = 1.0 - diff_norm
            
            total += score_i
        
        return total / self.n


    def _local_granularity_consistency_of_B_subset(self, B: List[int]) -> float:
        """
        Evaluates how consistent the fuzzy granules of B are within each
        class-specific partition of the dataset.

        Parameters
        ----------
        B : List[int]
            List of feature subset indices.

        Returns
        -------
        float
            Average local granularity consistency across all partitions.
        """
        total = 0.0
        K = len(self.D_partition)

        for key, df_part in self.D_partition.items():
            df_local = df_part.reset_index(drop=True)
            part_n = len(df_local)
            res = 0.0
            for i_local in range(part_n):
                _, delta_df_size = self._calculate_delta_for_column_subset(i_local, B, df=df_local)
                row_series = df_local.iloc[i_local]
                mask = np.all(self.U[df_local.columns].values == row_series.values, axis=1)
                if not np.any(mask):
                    ratio = 1.0
                else:
                    global_idx = np.where(mask)[0][0]
                    _, delta_U_size = self._calculate_delta_for_column_subset(int(global_idx), B, df=None)
                    ratio = delta_df_size / delta_U_size
                res += ratio
            total += (res / part_n)
        return total / K

    def _create_partitions(self) -> Dict[Any, pd.DataFrame]:
        """
        Partition the dataset into subsets according to target values.

        Returns
        -------
        Dict[Any, pd.DataFrame]
            Dictionary mapping each target class value to a sub-DataFrame
            containing only the objects belonging to that class.
        """
        partitions = {}
        target_col = self.target_name
        vals = self.U[target_col].unique()
        for v in vals:
            partitions[v] = self.U[self.U[target_col] == v].reset_index(drop=True).copy()
        return partitions


    def _FIGFS_algorithm(self):
        """
        Execute the Fuzzy Implication Granularity-based Feature Selection (FIGFS) algorithm.

        FIGFS iteratively selects features that maximize granularity consistency
        and minimize redundancy.

        Returns
        -------
        List[int]
            Ordered list of selected feature indices according to the FIGFS algorithm.
            The order reflects the importance of the features.
        """
        B = list(self.C.keys())
        S = []
        cor_list = []
        for col_index in B:
            cor = self._granual_consistency_of_B_subset([col_index]) + self._local_granularity_consistency_of_B_subset([col_index])
            cor_list.append(cor)
        c1 = cor_list.index(np.max(cor_list))
        S.append(c1)
        B.remove(c1)

        if self.m < self.d:
            while len(B) > 0:
                J_list = []
                for col_index in B:
                    sim = 0
                    for s_index in S:
                        fimi_d_cv = self._calculate_multi_granularity_fuzzy_implication_entropy([self.D[0]], type='mutual' , T=[col_index])
                        fimi_cv_cu = self._calculate_multi_granularity_fuzzy_implication_entropy([col_index], type='mutual' , T=[s_index])
                        fimi_cd = self._calculate_multi_granularity_fuzzy_implication_entropy([col_index], type='mutual' , T=[self.D[0], s_index])
                        sim += fimi_d_cv + fimi_cv_cu - fimi_cd
                    sim = sim / len(S)

                    l = S + [col_index]
                    W =  1 + (self._calculate_multi_granularity_fuzzy_implication_entropy(S, type='conditional' , T=[self.D[0]]) - self._calculate_multi_granularity_fuzzy_implication_entropy(S, type='conditional' , T=l)) / (self._calculate_multi_granularity_fuzzy_implication_entropy(S, type='conditional' , T=[self.D[0]]) + 0.01)
                    cor = self._granual_consistency_of_B_subset([col_index]) + self._local_granularity_consistency_of_B_subset([col_index])
                    j = W * cor - sim
                    J_list.append(j)
                arg_max = J_list.index(max(J_list))
                cv = B[arg_max]
                S.append(cv)
                B.remove(cv)
        else:
            FIE_dc = self._calculate_multi_granularity_fuzzy_implication_entropy([self.D[0]], type='conditional' , T=list(self.C.keys()))
            FIE_ds = self._calculate_multi_granularity_fuzzy_implication_entropy([self.D[0]], type='conditional' , T=S)
            while FIE_dc != FIE_ds:
                J_list = []
                W_list = []
                for col_index in B:
                    sim = 0
                    for s_index in S:
                        fimi_d_cv = self._calculate_multi_granularity_fuzzy_implication_entropy([self.D[0]], type='mutual' , T=[col_index])
                        fimi_cv_cu = self._calculate_multi_granularity_fuzzy_implication_entropy([col_index], type='mutual' , T=[s_index])
                        fimi_cd = self._calculate_multi_granularity_fuzzy_implication_entropy([col_index], type='mutual' , T=[self.D[0], s_index])
                        sim += fimi_d_cv + fimi_cv_cu - fimi_cd
                    sim = sim / len(S)

                    l = S + [col_index]
                    W =  1 + (self._calculate_multi_granularity_fuzzy_implication_entropy(S, type='conditional' , T=[self.D[0]]) - self._calculate_multi_granularity_fuzzy_implication_entropy(S, type='conditional' , T=l)) / (self._calculate_multi_granularity_fuzzy_implication_entropy(S, type='conditional' , T=[self.D[0]]) + 0.01)
                    cor = self._granual_consistency_of_B_subset([col_index]) + self._local_granularity_consistency_of_B_subset([col_index])
                    j = W * cor - sim
                    J_list.append(j)
                    W_list.append(W)
                arg_max = J_list.index(max(J_list))
                cv = B[arg_max]

                l = S + [cv]
                W_cv_max =  1 + (self._calculate_multi_granularity_fuzzy_implication_entropy(S, type='conditional' , T=[self.D[0]]) - self._calculate_multi_granularity_fuzzy_implication_entropy(S, type='conditional' , T=l)) / (self._calculate_multi_granularity_fuzzy_implication_entropy(S, type='conditional' , T=[self.D[0]]) + 0.01)
                percen = np.percentile(np.array(W_list), self.sigma)
                if W_cv_max >= percen:
                    S.append(cv)
                    B.remove(cv)
                else:
                    break
                FIE_ds = self._calculate_multi_granularity_fuzzy_implication_entropy([self.D[0]], type='conditional' , T=S)

        return S



# --------------------------------------
# WeightedFuzzyRoughSelector
# --------------------------------------
class WeightedFuzzyRoughSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=10, alpha=1.0, k=5):
        self.n_features = n_features
        self.alpha = alpha
        self.k = k
        self.W = None
        self.selected_features_ = None
        self.feature_importances_ = None
        self.feature_sequence_ = None
        self.Rw_ = None
    
    
    def fit(self, X, y):

        H = self._identify_high_density_region(X)
            
        relations_single, relations_pair = self._compute_fuzzy_similarity_relations(X, H)
        POS, NOG = self._compute_POS_NOG(relations_single, y, H)
        relevance = self._compute_relevance(POS, NOG)
        redundancy = self._compute_redundancy(X, y, H, relevance, relations_pair)
        weights = self._compute_feature_weights(relevance, redundancy)
        self.W = self._update_weight_matrix(weights, X.shape[1])
        
        self.feature_sequence_, self.Rw_ = self._build_weighted_feature_sequence(POS, NOG, weights, X, y, H, relations_single, relations_pair)
        self.selected_features_ = self.feature_sequence_[:self.n_features]


        self.feature_importances_ = pd.DataFrame({
            'feature': X.columns[self.feature_sequence_],
            'importance': np.diag(self.Rw_)[:len(self.feature_sequence_)]
        }).sort_values(by='importance', ascending=False).reset_index(drop=True)

        return self


    def transform(self, X):
        """
        Reduce X to the best n_features selected during fit
        """
        if self.selected_features_ is None:
            raise ValueError("The selector has not been fitted yet.")
        return X.iloc[:, self.selected_features_]
    

    def _identify_high_density_region(self, X):
        distances = self._compute_HEC(X)
        knn_indices = np.argsort(distances, axis=1)[:, 1:self.k+1]
        rho = self._compute_density(distances, knn_indices)
        LDF = self._compute_LDF(rho, knn_indices)

        H_indices = np.where(LDF <= 1)[0]
        H_neighbors = np.unique(knn_indices[H_indices].flatten())
        return H_neighbors
    

    def _compute_HEC(self, X, W=None):
        """
        Compute Hybrid Mahalanobis (HM) distance.
        Supports numerical, categorical and mixed features, including missing values.
        
        Parameters:
        X (pd.DataFrame): Input data with mixed features.
        W (np.ndarray or None): Diagonal weight matrix W_B. If None, uses identity matrix.
        
        Returns:
        distances (np.ndarray) Symmetric matrix of pairwise Hybrid distances (M_B(x,y)).
        """
        X = X.copy()
        n_samples, n_features = X.shape

        num_cols = X.select_dtypes(include=[np.number]).columns
        cat_cols = X.select_dtypes(exclude=[np.number]).columns

        if W is None:
            W = np.eye(n_features)

        distances = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i, n_samples):

                diff_vector = []

                for idx, col in enumerate(X.columns):
                    xi, xj = X.iloc[i][col], X.iloc[j][col]

                    # Handle missing values (NaN or None)
                    if pd.isna(xi) or pd.isna(xj):
                        diff = 1.0

                    # Numerical attribute
                    elif col in num_cols:
                        diff = abs(float(xi) - float(xj))

                    # Categorical attribute
                    else:
                        diff = 0.0 if xi == xj else 1.0

                    diff_vector.append(diff)

                diff_vector = np.array(diff_vector)

                weighted_diff = diff_vector @ W @ diff_vector.T

                distances[i, j] = np.sqrt(weighted_diff)
                distances[j, i] = distances[i, j]

        return distances
    
    
    def _compute_density(self, distances, knn_indices):
        """
        Compute local density rho(x) for each sample.
        """
        n_samples = distances.shape[0]
        rho = np.zeros(n_samples)
        for i in range(n_samples):
            neighbors = knn_indices[i]
            rho[i] = (1 + len(neighbors)) / (1 + np.sum(distances[i, neighbors]))
        return rho
    

    def _compute_LDF(self, rho, knn_indices):
        """
        Compute Local Density Factor (LDF) for each sample.
        """
        n_samples = len(rho)
        LDF = np.zeros(n_samples)
        for i in range(n_samples):
            neighbors = knn_indices[i]
            LDF[i] = np.mean(rho[neighbors] / rho[i])
        return LDF


    def _compute_fuzzy_similarity_relations(self, X, H, W=None):
        """
        Compute fuzzy similarity relations RM_{B}(x, y) for B = {a} and B = {a,b}
        """
        n_samples, n_features = X.shape
        feature_indices = list(range(n_features))
        relations_single = {}   # key: a → matrix [n_samples × len(H)]
        relations_pair = {}     # key: (a,b) → matrix [n_samples × len(H)]
        
        for a in feature_indices:
            # distance for B = {a}
            dist_a = self._compute_HEC(X.iloc[:, [a]], W=W[np.ix_([a],[a])] if W is not None else None)
            relations_single[a] = np.exp(- (dist_a ** 2) / (2 * self.alpha ** 2))
        
        for i, a in enumerate(feature_indices):
            for b in feature_indices[i+1:]:
                # distance for B = {a,b}
                dist_ab = self._compute_HEC(X.iloc[:, [a, b]], W=W[np.ix_([a,b],[a,b])] if W is not None else None)
                relations_pair[(a,b)] = np.exp(- (dist_ab ** 2) / (2 * self.alpha ** 2))
        
        return relations_single, relations_pair
    

    def _compute_relation_for_subset(self, X, H, feature_subset, W=None):
        """
        Compute fuzzy relation for a given feature subset
        """
        # Compute hybrid Mahalanobis distance for B ∪ {a}
        dist = self._compute_HEC(X.iloc[:, feature_subset], 
                                W=W[np.ix_(feature_subset, feature_subset)] if W is not None else None)
        
        # Fuzzy relation from Definition 4 (Gaussian kernel)
        relation = np.exp(- (dist ** 2) / (2 * self.alpha ** 2))
        return relation


    def _compute_POS_NOG(self, relations_single, y, H):
        """
        Compute fuzzy Positive (POS) and Non-negative (NOG) regions.

        Parameters:
        ----------
        relations (np.ndarray): Fuzzy relation matrix
        y (np.ndarray): Class labels.
        H (list[int]): Indices of high-density samples (subset of U)

        Returns:
        POS (np.ndarray): Fuzzy positive region values for each sample.
        NOG (np.ndarray): Fuzzy non-negative region values for each sample.
        """
        n_samples = len(y)
        classes = np.unique(y)
        POS_all = {}
        NOG_all = {}

        H_mask = np.zeros(n_samples, dtype=bool)
        H_mask[H] = True

        for a, rel_matrix in relations_single.items():
            lower_approx = np.zeros((n_samples, len(classes)))
            upper_approx = np.zeros((n_samples, len(classes)))

            for c_idx, c in enumerate(classes):
                # Membership function: X(y) = 1 if y ∈ class D_i
                Xy = (y == c).astype(float)

                R_H = rel_matrix[:, H] 
                Xy_H = Xy[H]

                diag_mask = np.eye(len(H), dtype=bool)

                for i in range(n_samples):
                    mask_i = H_mask.copy()
                    mask_i[i] = False
                    R_xy = rel_matrix[i, mask_i]
                    Xy_masked = Xy[mask_i]

                    lower_approx[i, c_idx] = np.min(np.maximum(1 - R_xy, Xy_masked))
                    upper_approx[i, c_idx] = np.max(np.minimum(R_xy, Xy_masked))

            POS_all[a] = np.max(lower_approx, axis=1) # fuzzy positive region
            NOG_all[a] = np.max(upper_approx, axis=1) # fuzzy non-negative region

        return POS_all, NOG_all


    def _compute_relevance(self, POS_all, NOG_all):
        """
        Compute relevance Rel(a) for each feature
        """
        relevance = {}
        for a in POS_all.keys():
            AM = POS_all[a] + NOG_all[a]
            relevance[a] = np.mean(AM)
        return relevance
    

    def _compute_redundancy(self, X, y, H, relevance, relations_pair):
        """
        Compute redundancy Red(a, b)
        """
        redundancy = {}
        features = list(relevance.keys())
        for (a, b), rel_matrix in relations_pair.items():
            if a in features and b in features:
                POS_ab, NOG_ab = self._compute_POS_NOG({0: rel_matrix}, y, H)
                AM_ab = POS_ab[0] + NOG_ab[0]
                Rel_ab = np.mean(AM_ab)
                redundancy[(a, b)] = relevance[a] + relevance[b] - Rel_ab
        return redundancy
    

    def _compute_feature_weights(self, relevance, redundancy):
        """
        Compute feature weights w(a)        
        """
        features = list(relevance.keys())
        
        rel_values = np.array(list(relevance.values()))
        rel_min, rel_max = rel_values.min(), rel_values.max()
        NR = {a: (relevance[a] - rel_min) / (rel_max - rel_min + 1e-12) for a in features}

        if len(redundancy) > 0:
            red_values = np.array(list(redundancy.values()))
            red_min, red_max = red_values.min(), red_values.max()
            NR_red = {k: (v - red_min) / (red_max - red_min + 1e-12) for k,v in redundancy.items()}
        else:
            NR_red = {}

        weights = {}
        for a in features:
            sum_NR_red = 0
            for b in features:
                if a != b:
                    key = (min(a,b), max(a,b))
                    sum_NR_red += NR_red.get(key, 0)
            weights[a] = NR[a] - sum_NR_red / max(len(features)-1, 1)
        
        return weights
    

    def _update_weight_matrix(self, weights, n_total_features):
        """
        Update W matrix using all feature indices (original indices from X)
        """
        W = np.zeros((n_total_features, n_total_features))
        for a, w_a in weights.items():
            W[a, a] = 1 / (1 + np.exp(-w_a))
        return W


    def _compute_gamma(self, POS_all, NOG_all, features):
        """
        Compute gamma_P and gamma_N for the current subset of features B.
        """
        POS_mean = np.mean([POS_all[a] for a in features], axis=0)
        NOG_mean = np.mean([NOG_all[a] for a in features], axis=0)
        gamma_P = np.mean(POS_mean)
        gamma_N = np.mean(NOG_mean)
        return gamma_P, gamma_N


    def _compute_separability(self, X, y, H, W, selected_features, remaining_features):
        """
        Compute separability sig(a, B, D) for each candidate feature.
        """
        separability = {}
        for a in remaining_features:
            B_union_a = selected_features + [a]
            relation_Ba = self._compute_relation_for_subset(X, H, B_union_a, W=W)
            POS_Ba, NOG_Ba = self._compute_POS_NOG({0: relation_Ba}, y, H)
            gamma_P, gamma_N = self._compute_gamma(POS_Ba, NOG_Ba, [0])
            separability[a] = gamma_P - gamma_N
        return separability


    def _build_weighted_feature_sequence(self, POS_all, NOG_all, weights, X, y, H, relations_single, relations_pair):
        """
        Build the weighted feature sequence according to steps (16–26).
        """
        all_features = list(POS_all.keys())
        selected_features = []
        remaining_features = all_features.copy()
        sequence = []

        cached_POS = {}
        cached_NOG = {}
        
        while len(remaining_features) > 0:
            separability = self._compute_separability(X, y, H, self.W, selected_features, remaining_features)
            best_feature = max(separability, key=separability.get)
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)

            POS_current = {}
            NOG_current = {}
            for f in selected_features:
                if f in cached_POS:
                    POS_current[f] = cached_POS[f]
                    NOG_current[f] = cached_NOG[f]
                else:
                    POS_f, NOG_f = self._compute_POS_NOG({f: relations_single[f]}, y, H)
                    POS_current[f] = POS_f[f]
                    NOG_current[f] = NOG_f[f]
                    cached_POS[f] = POS_current[f]
                    cached_NOG[f] = NOG_current[f]

            relevance = self._compute_relevance(POS_current, NOG_current)
            redundancy = self._compute_redundancy(X, y, H, relevance, relations_pair)
            weights = self._compute_feature_weights(relevance, redundancy)
            self.W = self._update_weight_matrix(weights, X.shape[1])

            sequence.append(best_feature)
        
        logistic_weights = [1 / (1 + np.exp(-weights[a])) for a in sequence]
        Rw = np.diag(logistic_weights)
        return sequence, Rw
