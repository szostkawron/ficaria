from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import clone

import numpy as np
import pandas as pd

class FuzzyImplicationGranularityFeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self, eps=0.01, d=10, sigma=3, classifier=LogisticRegression()):
        self.d = d
        self.sigma = sigma
        self.classifier = classifier
        self.eps = eps

    def fit(self, X, y=None):
        y_name = y.columns[0]
        self.U = X.copy()
        self.U[y_name] = y.values.ravel()

        self.C = {}
        self.D = (list(self.U.columns).index(y_name), y_name)

        for col in X.columns:
            pos = list(self.U.columns).index(col)
            if col in X.select_dtypes(include=['number']).columns:
                self.C[pos] = [col, 'numeric']
            else:
                self.C[pos] = [col, 'nominal']

        self.D_partition = self.create_partitions()
        self.n = len(self.U)
        self.m = len(self.C)

        self.fuzzy_adaptive_neighbourhood_radius = []
        for col in self.U.columns:
            if col in self.U.select_dtypes(include=["number"]).columns:
                self.fuzzy_adaptive_neighbourhood_radius.append(self.U[col].std() / self.eps)
            else:
                self.fuzzy_adaptive_neighbourhood_radius.append(None)

        self.S = self.FIGFS_algorithm()

        return self

    def transform(self, X):
        S_opt = None
        best_acc = -np.inf
        target_col = self.D[0]
        test_size = 0.3
        self.acc_list = []
        
        for i in range(1, len(self.S)+1):
            subset_cols = self.S[:i] + [self.D[0]]
            df_subset = self.U.iloc[:, subset_cols]

            X = df_subset.iloc[:, self.S[:i]].values 
            y = df_subset.iloc[:, target_col].values  

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            clf = clone(self.classifier) 
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            self.acc_list.append(acc)

            if acc > best_acc:
                best_acc = acc
                S_opt = self.S[:i]

        self.S_opt = S_opt

        return pd.DataFrame(self.U.iloc[:, S_opt])



    def FIGFS_algorithm(self):
        B = list(self.C.keys())
        S = []
        cor_list = []
        for col_index in B:
            cor = self.granual_consistency_of_B_subset([col_index]) + self.local_granularity_consistency_of_B_subset([col_index])
            cor_list.append(cor)
        c1 = B[np.argmax(cor_list)]
        S.append(c1)
        B.remove(c1)
        print(c1)

        if self.m < self.d:
            while len(B) > 0:
                J_list = []
                for col_index in B:
                    sim = 0
                    for s_index in S:
                        fimi_d_cv = self.calculate_multi_granularity_fuzzy_implication_entropy([self.D[0]], type='mutual' , T=[col_index])
                        fimi_cv_cu = self.calculate_multi_granularity_fuzzy_implication_entropy([col_index], type='mutual' , T=[s_index])
                        fimi_cd = self.calculate_multi_granularity_fuzzy_implication_entropy([col_index], type='mutual' , T=[self.D[0], s_index])
                        sim += fimi_d_cv + fimi_cv_cu - fimi_cd
                    sim = sim / len(S)

                    l = S + [col_index]
                    W =  1 + (self.calculate_multi_granularity_fuzzy_implication_entropy(S, type='conditional' , T=[self.D[0]]) - self.calculate_multi_granularity_fuzzy_implication_entropy(S, type='conditional' , T=l)) / (self.calculate_multi_granularity_fuzzy_implication_entropy(S, type='conditional' , T=[self.D[0]]) + 0.01)
                    cor = self.granual_consistency_of_B_subset([col_index]) + self.local_granularity_consistency_of_B_subset([col_index])
                    j = W * cor - sim
                    J_list.append(j)
                arg_max = J_list.index(max(J_list))
                cv = B[arg_max]
                S.append(cv)
                B.remove(cv)
        else:
            FIE_dc = self.calculate_multi_granularity_fuzzy_implication_entropy([self.D[0]], type='conditional' , T=list(self.C.keys()))
            FIE_ds = self.calculate_multi_granularity_fuzzy_implication_entropy([self.D[0]], type='conditional' , T=S)
            while FIE_dc != FIE_ds:
                J_list = []
                W_list = []
                for col_index in B:
                    sim = 0
                    for s_index in S:
                        fimi_d_cv = self.calculate_multi_granularity_fuzzy_implication_entropy([self.D[0]], type='mutual' , T=[col_index])
                        fimi_cv_cu = self.calculate_multi_granularity_fuzzy_implication_entropy([col_index], type='mutual' , T=[s_index])
                        fimi_cd = self.calculate_multi_granularity_fuzzy_implication_entropy([col_index], type='mutual' , T=[self.D[0], s_index])
                        sim += fimi_d_cv + fimi_cv_cu - fimi_cd
                    sim = sim / len(S)

                    l = S + [col_index]
                    W =  1 + (self.calculate_multi_granularity_fuzzy_implication_entropy(S, type='conditional' , T=[self.D[0]]) - self.calculate_multi_granularity_fuzzy_implication_entropy(S, type='conditional' , T=l)) / (self.calculate_multi_granularity_fuzzy_implication_entropy(S, type='conditional' , T=[self.D[0]]) + 0.01)
                    cor = self.granual_consistency_of_B_subset([col_index]) + self.local_granularity_consistency_of_B_subset([col_index])
                    j = W * cor - sim
                    J_list.append(j)
                    W_list.append(W)
                arg_max = J_list.index(max(J_list))
                cv = B[arg_max]

                l = S + [cv]
                W_cv_max =  1 + (self.calculate_multi_granularity_fuzzy_implication_entropy(S, type='conditional' , T=[self.D[0]]) - self.calculate_multi_granularity_fuzzy_implication_entropy(S, type='conditional' , T=l)) / (self.calculate_multi_granularity_fuzzy_implication_entropy(S, type='conditional' , T=[self.D[0]]) + 0.01)
                percen = np.percentile(np.array(W_list), self.sigma)
                if W_cv_max >= percen:
                    S.append(cv)
                    B.remove(cv)
                else:
                    break
                FIE_ds = self.calculate_multi_granularity_fuzzy_implication_entropy([self.D[0]], type='conditional' , T=S)

        return S

    def f_mapping(self, row, col, X):
        return X.iloc[row, col]
    
    def calculate_similarity_matrix(self, col_index, X): 
        df = X
        size = len(df)
        
        matrix = []
        col_type = self.C[col_index][1]
        for row1 in range(size):
            res = []
            for row2 in range(size):
                if col_type == 'numeric':
                    calc_R =  1 - (abs(self.f_mapping(row1, col_index, df) - self.f_mapping(row2, col_index, df)) / (1 + np.std(df.iloc[:, col_index])))
                    if float(calc_R) >= float(1 - self.fuzzy_adaptive_neighbourhood_radius[col_index]):
                        res.append(float(calc_R))
                    else:
                        res.append(0)
                else:
                    if self.f_mapping(row1, col_index, df) == self.f_mapping(row2, col_index, df):
                        res.append(1)
                    else:
                        res.append(0)
            matrix.append(res)
        return matrix
    
    def calculate_delta_for_column_subset(self, row, B, X=None): 
        matrix = []
        if X is None:
            df = self.U
        else:
            df = X
        if B is None:
            return (None, None)
        for col_index in B:
            similarity_matrix = self.calculate_similarity_matrix(col_index, df)
            matrix.append(similarity_matrix[row])
        granule = np.min(matrix, axis = 0)
        size = np.sum(granule)
        return (granule, size)

    def calculate_multi_granularity_fuzzy_implication_entropy(self, B, type='basic' , T=None): ## type IN ('basic', 'conditional', 'joint', 'mutual')
        res = 0
        for i in range(self.n):
            delta_B_size = self.calculate_delta_for_column_subset(i, B)[1]
            delta_T_size = self.calculate_delta_for_column_subset(i, T)[1]
            if type == 'basic':
                res += 1 - delta_B_size / self.n 
            elif type == 'conditional':
                res += max(delta_B_size, delta_T_size) - delta_B_size
            elif type == 'joint':
                res += 1 + max(delta_B_size, delta_T_size) / self.n - (delta_B_size + delta_T_size) / self.n
            else:
                res += 1-  max(delta_B_size, delta_T_size) / self.n
        return res / self.n ** 2 if type == 'conditional' else res / self.n
    
    def granual_consistency_of_B_subset(self, B):
        res = 0
        for i in range(self.n):
            delta_bd = np.maximum(np.zeros(self.n), np.array(self.calculate_delta_for_column_subset(i, B)[0]) - np.array(self.U.iloc[:, self.D[0]]))
            delta_db = np.maximum(0, np.array(self.U.iloc[:, self.D[0]]) -  np.array(self.calculate_delta_for_column_subset(i, B)[0]))
            res += 1 - max(np.sum(delta_bd), np.sum(delta_db)) / self.n
        return res / self.n
        
    def local_granularity_consistency_of_B_subset(self, B):
        total = 0
        for key, df in self.D_partition.items():
            partition_size = len(df)
            res = 0
            for i in range(partition_size):
                delta_df_size = self.calculate_delta_for_column_subset(i, B, df)[1]
                delta_U_size = self.calculate_delta_for_column_subset(i, B)[1]
                res += delta_df_size / delta_U_size
            total += res / partition_size
        return total / len(self.D_partition)

    def create_partitions(self):
        partitions = {}
        unique_y = self.U.iloc[:, self.D[0]].unique()
        for i in range(len(unique_y)):
            val = unique_y[i]
            df = self.U[self.U.iloc[:, self.D[0]] == val]
            partitions[val] = df
        return partitions
    
    ## to do later:
    ###     add documentation
    ###     specify data types for all parameters
    ###     handle errors and exceptions
