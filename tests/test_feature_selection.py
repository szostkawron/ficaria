import numpy as np
import pandas as pd
import pytest

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ficaria.feature_selection import *


# ----- WeightedFuzzyRoughSelector -----------------------------------------------

dataframes_list = [
    pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0], 
        "b": [4.0, 5.0, 6.0, 7.0, 6.0],
        "c": ["k", "k", "m", "m", "m"],
        }),

    pd.DataFrame({
        "a": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 
        "b": [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
        }),

    pd.DataFrame({
        "a": np.random.rand(10), 
        "b": np.random.rand(10), 
        "c": np.random.rand(10)
        }),

    pd.DataFrame({
        "a": [1, 2, np.nan, 4, 5, 6, 7],
        "b": [7, 6, 5, np.nan, 3, 2, 1],
        "c": ["x", np.nan, "z", "x", "y", "z", "x"],
        "d": [0.1, 0.2, 0.3, 0.4, 0.5, np.nan, 0.7]
    }),

    pd.DataFrame({
        "a": np.random.randint(0, 100, size=50),
        "b": np.random.rand(50),
        "c": np.random.choice(["red", "blue", "green"], size=50),
        "d": np.random.randn(50),
        "e": np.random.rand(50),
    }),

    pd.DataFrame({
        "a": ["txt", "txt", "txt", "csv", "csv"], 
        "b": ["A", "B", "B", "C", "C"],
        "c": ["k", "k", "m", "m", "m"],
        }),
]

y_list = [
    np.array([0, 1, 0, 1, 1]),
    np.array([1, 1, 1, 0, 0, 0, 0]),
    np.random.randint(0, 2, size=10),
    np.array([0, 1, 0, 1, 1, 0, 1]),
    np.random.randint(0, 3, size=50),
    np.array(['y','n', 'y', 'n', 'y']),
]

selector_params_list = [
    (1, 0.1, 2),
    (1, 0.5, 3),
    (2, 0.6, 4),
    (2, 1.0, 5),
    (3, 0.25, 2),
    (3, 0.75, 6)
]

@pytest.mark.parametrize("n_features,alpha,k", selector_params_list)
def test_weightedfuzzyroughselector_init_parametrized(n_features, alpha, k):
    selector = WeightedFuzzyRoughSelector(n_features=n_features, alpha=alpha, k=k)
    assert selector.n_features == n_features
    assert selector.alpha == alpha
    assert selector.k == k


def test_weightedfuzzyroughselector_transform_raises_if_not_fitted():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    selector = WeightedFuzzyRoughSelector(n_features=1)
    with pytest.raises(AttributeError, match="fit must be called before transform"):
        selector.transform(X)


def test_weightedfuzzyroughselector_fit_raises_if_y_has_missing_values():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    y = pd.Series([1, np.nan, 0])
    selector = WeightedFuzzyRoughSelector(n_features=1, k=2)

    with pytest.raises(ValueError, match="Target variable y contains missing values"):
        selector.fit(X, y)


def test_weightedfuzzyroughselector_fit_raises_if_y_length_mismatch():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    y = pd.Series([0, 1])
    selector = WeightedFuzzyRoughSelector(n_features=1, k=2)

    with pytest.raises(ValueError, match="Length mismatch: X has 3 samples but y has 2 entries"):
        selector.fit(X, y)


def test_weightedfuzzyroughselector_transform_raises_if_columns_differ():
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [4.0, 5.0, 6.0, 7.0, 8.0]})
    X_test = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "c": [7.0, 8.0, 9.0, 10.0, 11.0]})
    y = np.array([0, 1, 0, 1, 0])

    selector = WeightedFuzzyRoughSelector(n_features=1, k=2)
    selector.fit(X_train, y)

    with pytest.raises(ValueError, match="Columns in transform do not match columns seen during fit"):
        selector.transform(X_test)


def test_weightedfuzzyroughselector_fit_creates_attributes():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [4.0, 5.0, 6.0, 7.0, 8.0]})
    y = np.array([0, 1, 0, 1, 0])
    selector = WeightedFuzzyRoughSelector(n_features=1, k=2)
    selector.fit(X, y)
    
    assert hasattr(selector, "feature_sequence_")
    assert hasattr(selector, "feature_importances_")
    assert hasattr(selector, "Rw_")


def test_weightedfuzzyroughselector_transform_returns_reduced_features():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [4.0, 5.0, 6.0, 7.0, 8.0], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
    y = pd.Series([0, 1, 0, 1, 0])
    selector = WeightedFuzzyRoughSelector(n_features=2, k=2)
    selector.fit(X, y)
    
    X_transformed = selector.transform(X)
    
    assert X_transformed.shape[1] == 2
    assert all(col in X.columns for col in X_transformed.columns)


@pytest.mark.parametrize("X, y, params", [
    (X, y, params) 
    for X, y in zip(dataframes_list, y_list) 
    for params in selector_params_list
])
def test_weightedfuzzyroughselector_fit_transform_combinations(X, y, params):
    n_features, alpha, k = params

    n_features, alpha, k = params
    
    y = np.asarray(y)
    
    if not np.issubdtype(y.dtype, np.integer):
        y, uniques = pd.factorize(y)
    min_class_count = np.min(np.bincount(y))
    if k >= min_class_count:
        k = int(min_class_count - 1)
        if k <= 1:
            pytest.skip(f"Skipping k for this dataset, not enough samples in smallest class.")

    if n_features >= X.shape[1]:
        n_features = X.shape[1] 

    selector = WeightedFuzzyRoughSelector(n_features=n_features, alpha=alpha, k=k)
    
    selector.fit(X, y)
    
    X_transformed = selector.transform(X)
    
    assert X_transformed.shape[0] == len(X)
    assert X_transformed.shape[1] == n_features
    assert all(col in X.columns for col in X_transformed.columns)
    assert selector.feature_sequence_ is not None
    assert selector.feature_importances_ is not None


def test_single_feature_dataset():
    X = pd.DataFrame({"a": [1, 2, 1, 2, 1]})
    y = np.array([0, 1, 0, 1, 0])

    selector = WeightedFuzzyRoughSelector(n_features=1, alpha=1.0, k=2)
    
    selector.fit(X, y)
    
    assert selector.feature_sequence_ == [0]    
    assert selector.feature_importances_.shape[0] == 1
    assert selector.feature_importances_['feature'].iloc[0] == "a"
    X_transformed = selector.transform(X)
    assert X_transformed.shape == X.shape
    assert (X_transformed['a'] == X['a']).all()
    assert selector.Rw_.shape == (1, 1)


def test_reproducibility_same_input():
    X = pd.DataFrame({
        "a": [1, 2, 1, 2, 1],
        "b": [5, 4, 5, 4, 5],
        "c": [9, 8, 9, 8, 9]
    })
    y = np.array([0, 1, 0, 1, 0])
    
    n_features = 2
    alpha = 1.0
    k = 2
    
    selector1 = WeightedFuzzyRoughSelector(n_features=n_features, alpha=alpha, k=k)
    selector1.fit(X, y)
    sequence1 = selector1.feature_sequence_
    importances1 = selector1.feature_importances_.copy()
    
    selector2 = WeightedFuzzyRoughSelector(n_features=n_features, alpha=alpha, k=k)
    selector2.fit(X, y)
    sequence2 = selector2.feature_sequence_
    importances2 = selector2.feature_importances_.copy()
    
    assert sequence1 == sequence2, "Feature sequences differ across runs"
    
    pd.testing.assert_frame_equal(importances1, importances2)
    
    X_trans1 = selector1.transform(X)
    X_trans2 = selector2.transform(X)
    pd.testing.assert_frame_equal(X_trans1, X_trans2)


def test_n_features_greater_or_equal_than_columns_raises():
    X = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "c": [7, 8, 9]
    })
    y = np.array([0, 1, 0])
    
    selector_equal = WeightedFuzzyRoughSelector(n_features=20, alpha=1.0, k=2)
    with pytest.raises(ValueError):
        selector_equal.fit(X, y)
    
    selector_greater = WeightedFuzzyRoughSelector(n_features=5, alpha=1.0, k=2)
    with pytest.raises(ValueError):
        selector_greater.fit(X, y)


def test_weightedfuzzyroughselector_selected_columns_match_sequence():
    X = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [4, 3, 2, 1],
        "c": [10, 20, 30, 40]
    })
    y = np.array([0, 1, 0, 1])

    selector = WeightedFuzzyRoughSelector(n_features=2, k=2)
    selector.fit(X, y)

    transformed = selector.transform(X)

    assert hasattr(selector, "feature_sequence_")
    assert len(selector.feature_sequence_) == X.shape[1]

    expected_cols = [X.columns[i] for i in selector.feature_sequence_[:2]]
    assert list(transformed.columns) == expected_cols


def test_weightedfuzzyroughselector_invalid_y_type():
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    y = ["a", "b"]
    selector = WeightedFuzzyRoughSelector(n_features=1, k=2)

    with pytest.raises(ValueError):
        selector.fit(X, y)




# ----- _identify_high_density_region ---------------------------------------

def test_identify_high_density_region():
    X = pd.DataFrame({
        "a": [1.0, 2.0, 1.1, 3.0],
        "b": [4.0, 5.0, 4.1, 6.0]
    })
    y = pd.Series([0, 1, 0, 1])
    
    selector = WeightedFuzzyRoughSelector(n_features=1, k=2)
    selector._compute_HEC = lambda X1: np.array([[0,1,0.1,2],
                                                 [1,0,1.1,1],
                                                 [0.1,1.1,0,2.1],
                                                 [2,1,2.1,0]])
    selector._compute_density = lambda distances, knn: np.ones(X.shape[0])
    selector._compute_LDF = lambda rho, knn: np.ones(X.shape[0])
    
    H_neighbors = selector._identify_high_density_region(X, y)
    
    assert isinstance(H_neighbors, np.ndarray)
    assert H_neighbors.size > 0


#----- _compute_HEC ---------------------------------------------------------

def test_compute_HEC_numeric_and_categorical():
    X = pd.DataFrame({
        "num": [1.0, 2.0, np.nan],
        "cat": ["a", "b", "a"]
    })
    
    selector = WeightedFuzzyRoughSelector(n_features=1, k=2)
    distances = selector._compute_HEC(X)
    
    assert distances.shape == (3,3)
    assert np.all(distances >= 0)


# ----- _compute_density -----------------------------------------------------

def test_compute_density_simple():
    distances = np.array([[0,1,2],[1,0,3],[2,3,0]])
    knn_indices = np.array([[1,2],[0,2],[0,1]])
    
    selector = WeightedFuzzyRoughSelector(n_features=1, k=2)
    rho = selector._compute_density(distances, knn_indices)
    
    assert rho.shape[0] == 3
    assert np.all(rho > 0)


# ----- _compute_LDF ---------------------------------------------------------
def test_compute_LDF_simple():
    rho = np.array([1.0, 2.0, 1.5])
    knn_indices = np.array([[1,2],[0,2],[0,1]])
    
    selector = WeightedFuzzyRoughSelector(n_features=1, k=2)
    LDF = selector._compute_LDF(rho, knn_indices)
    
    assert LDF.shape[0] == 3
    assert np.all(LDF > 0)


# ----- _compute_fuzzy_similarity_relations ---------------------------------

def test_compute_fuzzy_similarity_relations_small():
    X = pd.DataFrame({
        "a":[1,2,1],
        "b":[3,4,3]
    })
    H = [0,2]
    selector = WeightedFuzzyRoughSelector(n_features=1, k=2, alpha=0.5)
    selector._compute_HEC = lambda X1, X2=None, W=None: np.zeros((3,2))
    
    relations_single, relations_pair = selector._compute_fuzzy_similarity_relations(X,H)
    
    assert isinstance(relations_single, dict)
    assert isinstance(relations_pair, dict)
    assert all(0 <= v.max() <= 1 for v in relations_single.values())



# ----- _compute_redundancy --------------------------------------------------

def test_compute_redundancy_basic():
    POS_all = {0: np.array([0.1,0.2])}
    NOG_all = {0: np.array([0.1,0.2])}
    relations_pair = {(0,1): np.array([[1,1],[1,1]])}
    relevance = {0:0.2, 1:0.2}
    y = np.array([0,1])
    H = [0,1]
    
    selector = WeightedFuzzyRoughSelector(n_features=1)
    selector._compute_POS_NOG = lambda rels, y, H: ({0: np.array([0.1,0.1])}, {0: np.array([0.1,0.1])})
    
    redundancy = selector._compute_redundancy(y,H,relevance,relations_pair)
    assert isinstance(redundancy, dict)


# ----- _compute_feature_weights --------------------------------------------

def test_compute_feature_weights_basic():
    relevance = {0:0.2, 1:0.4}
    redundancy = {(0,1):0.3}
    
    selector = WeightedFuzzyRoughSelector(n_features=1)
    weights = selector._compute_feature_weights(relevance, redundancy)
    
    assert isinstance(weights, dict)
    assert set(weights.keys()) == {0,1}


# ----- _update_weight_matrix -----------------------------------------------

def test_update_weight_matrix_basic():
    weights = {0:0.0,1:1.0}
    selector = WeightedFuzzyRoughSelector(n_features=1)
    W = selector._update_weight_matrix(weights, n_total_features=2)
    
    assert W.shape == (2,2)
    assert np.all(W >= 0)


# ----- _compute_gamma -------------------------------------------------------

def test_compute_gamma_basic():
    POS_all = {0: np.array([0.2,0.4]), 1: np.array([0.1,0.3])}
    NOG_all = {0: np.array([0.1,0.1]), 1: np.array([0.2,0.2])}
    features = [0,1]
    
    selector = WeightedFuzzyRoughSelector(n_features=1)
    gamma_P, gamma_N = selector._compute_gamma(POS_all, NOG_all, features)
    
    assert 0 <= gamma_P <= 1
    assert 0 <= gamma_N <= 1


# ----- _build_weighted_feature_sequence ------------------------------------

def test_build_weighted_feature_sequence_equal_weights():
    X = pd.DataFrame({
        'a': [1, 2, 3, 4, 5, 6],
        'b': [2, 3, 4, 5, 6, 7],
        'c': [3, 4, 5, 6, 7, 8],
        'd': [4, 5, 6, 7, 8, 9]
    })
    
    y = np.array([0, 1, 0, 1, 0, 1])
    
    selector = WeightedFuzzyRoughSelector(n_features=4, alpha=1.0, k=2)
    
    selector.W_ = np.eye(X.shape[1])    
    H = np.arange(len(X))
    relations_single, relations_pair = selector._compute_fuzzy_similarity_relations(X, H, W=selector.W_)
    sequence, Rw = selector._build_weighted_feature_sequence(relations_single, relations_pair, X, y, H)    
    assert sorted(sequence) == list(range(X.shape[1]))
    
    expected_weights = [selector.W_[i, i] for i in sequence]
    np.testing.assert_array_almost_equal(np.diag(Rw), expected_weights)
    
    separability_values = []
    selected_features = []
    remaining_features = list(range(X.shape[1]))
    while remaining_features:
        sep = selector._compute_separability(X, y, H, selector.W_, selected_features, remaining_features)
        best = max(sep, key=sep.get)
        separability_values.append(sep[best])
        selected_features.append(best)
        remaining_features.remove(best)
    assert all(separability_values[i] >= separability_values[i + 1] for i in range(len(separability_values) - 1))


# ----- _compute_POS_NOG_B --------------------------------------------------

def test_compute_POS_NOG_B_basic():
    R_B = np.array([
        [1.0, 0.2],
        [0.1, 0.9],
        [0.5, 0.5]
    ])
    y = np.array([0,1,0])
    H = [0,1]

    selector = WeightedFuzzyRoughSelector(n_features=1)
    POS, NOG = selector._compute_POS_NOG_B(R_B, y, H)

    assert POS.shape[0] == len(y)
    assert NOG.shape[0] == len(y)
    assert np.all((POS >= 0) & (POS <= 1))
    assert np.all((NOG >= 0) & (NOG <= 1))


# ----- _compute_relevance_B --------------------------------------------------

def test_compute_relevance_B_basic():
    R_B = np.array([
        [1.0, 0.2],
        [0.1, 0.9],
        [0.5, 0.5]
    ])
    y = np.array([0,1,0])
    H = [0,1]

    selector = WeightedFuzzyRoughSelector(n_features=1)
    RelB, POS_B, NOG_B = selector._compute_relevance_B(R_B, y, H)

    assert isinstance(RelB, float)
    assert POS_B.shape[0] == len(y)
    assert NOG_B.shape[0] == len(y)


# ----- _compute_relation_for_subset --------------------------------------------------

def test_compute_relation_for_subset_basic():
    X = pd.DataFrame({
        'a':[1,2,3],
        'b':[4,5,6]
    })
    H = [0,2]
    feature_subset = [0,1]

    selector = WeightedFuzzyRoughSelector(n_features=2, alpha=0.5)
    relation = selector._compute_relation_for_subset(X, H, feature_subset)

    assert relation.shape == (len(X), len(H))
    assert np.all((relation >= 0) & (relation <= 1))


# ----- __compute_separability --------------------------------------------------

def test_compute_separability_basic():
    X = pd.DataFrame({
        'a':[1,2,3],
        'b':[4,5,6],
        'c':[7,8,9]
    })
    y = np.array([0,1,0])
    H = [0,2]

    selector = WeightedFuzzyRoughSelector(n_features=2, alpha=1.0)
    selector.W_ = np.eye(3)

    selected_features = []
    remaining_features = [0,1,2]
    
    separability = selector._compute_separability(X, y, H, selector.W_, selected_features, remaining_features)
    
    assert isinstance(separability, dict)
    assert set(separability.keys()) == set(remaining_features)
    assert all(isinstance(v, float) for v in separability.values())
