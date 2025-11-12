import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple, Dict, Any

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from .utils import check_input_dataset


class FuzzyGranularitySelector(BaseEstimator, TransformerMixin):
    """
    Fuzzy Implication Granularity Feature Selection (FIGFS).

    Selects an optimal feature subset using a fuzzy-implication-based
    granularity similarity framework. Compatible with scikit-learn
    classifiers.

    Parameters
    ----------
    classifier : sklearn-like estimator
        Base classifier implementing fit/predict.

    eps : float, default=0.5
        Controls fuzzy radius normalization (> 0).

    d : int, default=3
        Maximum number of features to consider (> 0).

    sigma : int, default=10
        Similarity scaling factor (1 <= sigma <= 100).

    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    S : list of int
        Feature ordering after FIGFS fitting.

    S_opt : list of int
        Subset of optimal features chosen after transform().

    acc_list : list of float
        Accuracies per step of feature subset selection.

    _fitted_columns : list
        Column names used during fit().
    """

    def __init__(
        self,
        classifier,
        eps: float = 0.5,
        d: int = 3,
        sigma: int = 10,
        random_state: Optional[int] = None,
    ):
        if not hasattr(classifier, "fit") or not hasattr(classifier, "predict"):
            raise ValueError("Classifier must implement fit() and predict().")
        if not isinstance(eps, (int, float)) or eps <= 0:
            raise ValueError("eps must be a positive number.")
        if not isinstance(d, int) or d <= 0:
            raise ValueError("d must be a positive integer.")
        if not isinstance(sigma, int) or not (1 <= sigma <= 100):
            raise ValueError("sigma must be an integer in [1, 100].")
        if random_state is not None and not isinstance(random_state, int):
            raise ValueError("random_state must be an integer or None.")

        self.classifier = classifier
        self.eps = float(eps)
        self.d = int(d)
        self.sigma = int(sigma)
        self.random_state = random_state

        self.S: Optional[List[int]] = None
        self.S_opt: Optional[List[int]] = None
        self.acc_list: List[float] = []
        self._fitted_columns: Optional[List[str]] = None
        self._label_encoders: Dict[str, LabelEncoder] = {}

        self.C: Dict[int, Tuple[str, str]] = {}
        self.U: Optional[pd.DataFrame] = None
        self.target_name: str = "target"
        self._delta_cache: Dict[Any, Any] = {}
        self._entropy_cache: Dict[Any, Any] = {}
        self.D: Tuple[int, str] = (0, self.target_name)
        self.n: int = 0
        self.m: int = 0
        self.fuzzy_adaptive_neighbourhood_radius: Dict[int, Optional[float]] = {}
        self.similarity_matrices: Dict[int, np.ndarray] = {}
        self.D_partition: Dict[Any, pd.DataFrame] = {}

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray, List[List[Any]]],
        y: Optional[Union[pd.Series, np.ndarray, pd.DataFrame]] = None,
    ):
        """
        Fit the FIGFS algorithm on the dataset and determine the optimal feature subset.

        Parameters
        ----------
        X : DataFrame, ndarray, or list of lists
            Feature matrix.
        y : Series, ndarray, DataFrame, or None, default=None
            Target variable. If None, runs in unsupervised mode.

        Returns
        -------
        self : object
            Fitted instance with selected optimal features in self.S_opt.
        """
        X = check_input_dataset(X, allow_nan=False)

        if y is not None and isinstance(y, (np.ndarray, pd.Series, pd.DataFrame)) and len(y) != len(X):
            raise ValueError("X and y must have the same number of rows.")

        if y is None:
            y_ser = pd.Series(np.zeros(len(X), dtype=int), name=self.target_name)
        elif isinstance(y, pd.DataFrame):
            y_ser = y.iloc[:, 0]
        else:
            y_ser = pd.Series(y).reset_index(drop=True)
            y_ser.name = self.target_name

        self._fitted_columns = list(X.columns)
        self.C = {
            idx: (col, "numeric" if pd.api.types.is_numeric_dtype(X[col]) else "nominal")
            for idx, col in enumerate(X.columns)
        }
        self.m = len(self.C)

        # --- Obliczanie promienia sąsiedztwa fuzzy ---
        self.fuzzy_adaptive_neighbourhood_radius = {}
        for col_idx, (col_name, col_type) in self.C.items():
            if col_type == "numeric":
                std_val = float(X[col_name].std(ddof=0))
                self.fuzzy_adaptive_neighbourhood_radius[col_idx] = std_val / self.eps if self.eps != 0 else 0.0
            else:
                self.fuzzy_adaptive_neighbourhood_radius[col_idx] = None

        # --- Dane wewnętrzne ---
        self.U = X.copy()
        self.U[self.target_name] = y_ser.values
        self.n = len(self.U)

        rng = np.random.default_rng(self.random_state)
        take = min(self.d, len(X.columns))
        perm = list(rng.permutation(len(X.columns)))
        self.S = perm[:take]

        self._delta_cache = {}
        self._entropy_cache = {}
        self.D = (len(X.columns), self.target_name)
        self.D_partition = self._create_partitions()

        # --- Label encoders dla nominalnych cech ---
        self._label_encoders = {}
        X_encoded = X.copy()
        for _, (col, col_type) in self.C.items():
            if col_type == "nominal":
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col])
                self._label_encoders[col] = le

        # --- Wybór optymalnego podzbioru cech (logika przeniesiona z transform) ---
        self.acc_list = []
        best_acc = -np.inf
        S_opt: Optional[List[int]] = None

        for i in range(1, len(self.S) + 1):
            subset = self.S[:i]
            cols = [self.C[j][0] for j in subset]

            model = clone(self.classifier)
            if hasattr(model, "random_state"):
                try:
                    model.set_params(random_state=self.random_state)
                except Exception:
                    pass

            X_sub = X_encoded[cols]
            y_sub = self.U[self.target_name]

            stratify_y = y_sub if len(np.unique(y_sub)) > 1 else None

            X_train, X_test, y_train, y_test = train_test_split(
                X_sub,
                y_sub,
                test_size=0.3,
                random_state=self.random_state,
                stratify=stratify_y,
            )

            scaler = MinMaxScaler()
            num_cols = X_train.select_dtypes(include=["number"]).columns
            if len(num_cols) > 0:
                X_train.loc[:, num_cols] = scaler.fit_transform(X_train[num_cols])
                X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            self.acc_list.append(acc)

            if acc > best_acc:
                best_acc = acc
                S_opt = subset

        self.S_opt = S_opt if S_opt is not None else list(self.S)

        return self


    def transform(self, X: Union[pd.DataFrame, np.ndarray, List[List[Any]]]) -> pd.DataFrame:
        """
        Transform input dataset using selected optimal feature subset.

        Parameters
        ----------
        X : DataFrame, ndarray, or list of lists
            Input data with same structure as used in fit().

        Returns
        -------
        DataFrame
            Reduced dataset with optimal features.
        """

        if self.S is None:
            raise RuntimeError("You must call fit() before transform().")

        X = check_input_dataset(X, allow_nan=False)

        if self._fitted_columns is None:
            raise RuntimeError("fit() must be called before transform().")
        if list(X.columns) != self._fitted_columns:
            raise ValueError("Input X columns differ from those used in fit().")

        X_transformed = X.copy()
        for _, (col, col_type) in self.C.items():
            if col_type == "nominal":
                le = self._label_encoders.get(col)
                if le is None:
                    raise RuntimeError(f"Label encoder for column '{col}' not found. Ensure fit() was called first.")
                X_transformed[col] = le.transform(X_transformed[col])

        selected_indices = self.S_opt if hasattr(self, "S_opt") and self.S_opt is not None else self.S
        final_cols = [self.C[idx][0] for idx in selected_indices]

        return X_transformed[final_cols].copy()



    def _calculate_similarity_matrix_for_df(self, col_index: int, df: pd.DataFrame) -> np.ndarray:
        """
        Compute fuzzy similarity matrix for a single column (numeric or categorical),
        working correctly in both global and local contexts.

        Parameters
        ----------
        col_index : int
            Column index (can refer to position in df or in self.C).
        df : pd.DataFrame
            DataFrame containing the data (global or local context).

        Returns
        -------
        np.ndarray
            n x n fuzzy similarity matrix.
        """
        # Pobierz nazwę i typ kolumny
        if isinstance(self.C, list) and col_index < len(self.C):
            col_name, col_type = self.C[col_index]
        else:
            col_name = df.columns[col_index]
            dtype = df[col_name].dtype
            col_type = 'numeric' if np.issubdtype(dtype, np.number) else 'categorical'

        vals = df[col_name].values
        n = len(df)
        mat = np.zeros((n, n), dtype=float)

        if col_type == 'numeric':
            sd = float(df[col_name].std(ddof=0)) if n > 1 else 0.0
            denom = 1.0 + sd

            radius = getattr(self, "fuzzy_adaptive_neighbourhood_radius", {}).get(col_index, None)

            for i in range(n):
                diff = np.abs(vals[i] - vals)
                sim = 1.0 - (diff / denom)
                sim = np.clip(sim, 0.0, 1.0)

                if radius is None:
                    mat[i, :] = sim
                else:
                    thresh = 1.0 - radius
                    mat[i, :] = np.where(sim >= thresh, sim, 0.0)
        else:  # categorical
            for i in range(n):
                mat[i, :] = (vals[i] == vals).astype(float)

        return mat
    

    def _calculate_delta_for_column_subset(self, row_index: int, B: List[int], df: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, float]:
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
            if df.columns[col_index] == self.D[0]:
                y_vals = self.U[self.target_name].values
                current_class = y_vals[row_index]
                vec = (y_vals == current_class).astype(float)
            else:
                col_name = df.columns[col_index]
                if use_global:
                    mat = self.similarity_matrices.get(col_name)
                    if mat is None:
                        mat = self._calculate_similarity_matrix_for_df(col_index, df)
                        self.similarity_matrices[col_name] = mat
                else:
                    mat = self._calculate_similarity_matrix_for_df(col_index, df)

                vec = mat[row_index, :].astype(float)

            mats.append(vec)

        if len(mats) == 0:
            granule = np.zeros(len(df), dtype=float)
        else:
            granule = np.minimum.reduce(mats)

        size = float(np.sum(granule))
        self._delta_cache[key] = (granule, size)
        return granule, size



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
    
    def _granular_consistency_of_B_subset(self, B: list) -> float:
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
            cor = self._granular_consistency_of_B_subset([col_index]) + self._local_granularity_consistency_of_B_subset([col_index])
            cor_list.append(cor)

        cor_arr = np.asarray(cor_list, dtype=float)
        best = np.where(cor_arr == cor_arr.max())[0]
        c1 = int(best[0])

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
                    cor = self._granular_consistency_of_B_subset([col_index]) + self._local_granularity_consistency_of_B_subset([col_index])
                    j = W * cor - sim
                    J_list.append(j)

                J_arr = np.asarray(J_list, dtype=float)
                best = np.where(J_arr == J_arr.max())[0]
                arg_max = int(best[0])
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
                    cor = self._granular_consistency_of_B_subset([col_index]) + self._local_granularity_consistency_of_B_subset([col_index])
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
