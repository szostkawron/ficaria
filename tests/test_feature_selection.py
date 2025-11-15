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
]

y_list = [
    np.array([0, 1, 0, 1, 1]),
    np.array([1, 1, 1, 0, 0, 0, 0]),
    np.random.randint(0, 2, size=10)
]

selector_params_list = [
    (0.1, 2),
    (0.5, 3),
    (0.6, 4),
    (1.0, 5)
]

@pytest.mark.parametrize("alpha,k", selector_params_list)
def test_weightedfuzzyroughselector_init_parametrized(alpha, k):
    selector = WeightedFuzzyRoughSelector(alpha=alpha, k=k)
    assert selector.alpha == alpha
    assert selector.k == k


def test_weightedfuzzyroughselector_fit_raises_if_k_too_large():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    y = np.array([0, 1, 0])
    selector = WeightedFuzzyRoughSelector(k=5)
    with pytest.raises(ValueError, match="Invalid value for k: 5. Must be lower than number of samples"):
        selector.fit(X, y)


def test_weightedfuzzyroughselector_transform_raises_if_not_fitted():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    selector = WeightedFuzzyRoughSelector()
    with pytest.raises(AttributeError, match="fit must be called before transform"):
        selector.transform(X)


def test_weightedfuzzyroughselector_fit_raises_if_y_has_missing_values():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    y = pd.Series([1, np.nan, 0])
    selector = WeightedFuzzyRoughSelector(k=2)

    with pytest.raises(ValueError, match="Target variable y contains missing values"):
        selector.fit(X, y)


def test_weightedfuzzyroughselector_fit_raises_if_y_length_mismatch():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    y = pd.Series([0, 1])
    selector = WeightedFuzzyRoughSelector(k=2)

    with pytest.raises(ValueError, match="Length mismatch: X has 3 samples but y has 2 entries"):
        selector.fit(X, y)


def test_weightedfuzzyroughselector_transform_raises_if_columns_differ():
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [4.0, 5.0, 6.0, 7.0, 8.0]})
    X_test = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "c": [7.0, 8.0, 9.0, 10.0, 11.0]})
    y = np.array([0, 1, 0, 1, 0])

    selector = WeightedFuzzyRoughSelector(k=2)
    selector.fit(X_train, y)

    with pytest.raises(ValueError, match="Columns in transform do not match columns seen during fit"):
        selector.transform(X_test)


def test_weightedfuzzyroughselector_fit_creates_attributes():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [4.0, 5.0, 6.0, 7.0, 8.0]})
    y = np.array([0, 1, 0, 1, 0])
    selector = WeightedFuzzyRoughSelector(k=2)
    selector.fit(X, y)
    
    assert hasattr(selector, "feature_sequence_")
    assert hasattr(selector, "feature_importances_")
    assert hasattr(selector, "Rw_")


def test_weightedfuzzyroughselector_transform_returns_reduced_features():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [4.0, 5.0, 6.0, 7.0, 8.0], "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
    y = pd.Series([0, 1, 0, 1, 0])
    selector = WeightedFuzzyRoughSelector(k=2)
    selector.fit(X, y)
    
    X_transformed = selector.transform(X, n_features=2)
    
    assert X_transformed.shape[1] == 2
    assert all(col in X.columns for col in X_transformed.columns)


@pytest.mark.parametrize("X, y, params", [
    (X, y, params) 
    for X, y in zip(dataframes_list, y_list) 
    for params in selector_params_list
])
def test_weightedfuzzyroughselector_fit_transform_combinations(X, y, params):
    alpha, k = params

    if k >= len(X):
        pytest.skip(f"Skipping k={k} >= number of samples {len(X)}")

    selector = WeightedFuzzyRoughSelector(alpha=alpha, k=k)
    
    selector.fit(X, y)
    
    n_features = min(len(X.columns), k)
    X_transformed = selector.transform(X, n_features=n_features)
    
    assert X_transformed.shape[0] == len(X)
    assert X_transformed.shape[1] == n_features
    assert all(col in X.columns for col in X_transformed.columns)
    assert selector.feature_sequence_ is not None
    assert selector.feature_importances_ is not None


# ----- _identify_high_density_region ---------------------------------------

def test_identify_high_density_region():
    X = pd.DataFrame({
        "a": [1.0, 2.0, 1.1, 3.0],
        "b": [4.0, 5.0, 4.1, 6.0]
    })
    
    selector = WeightedFuzzyRoughSelector(k=2)
    selector._compute_HEC = lambda X1: np.array([[0,1,0.1,2],
                                                 [1,0,1.1,1],
                                                 [0.1,1.1,0,2.1],
                                                 [2,1,2.1,0]])
    selector._compute_density = lambda distances, knn: np.ones(X.shape[0])
    selector._compute_LDF = lambda rho, knn: np.ones(X.shape[0])
    
    H_neighbors = selector._identify_high_density_region(X)
    
    assert isinstance(H_neighbors, np.ndarray)
    assert H_neighbors.size > 0


#----- _compute_HEC ---------------------------------------------------------

def test_compute_HEC_numeric_and_categorical():
    X = pd.DataFrame({
        "num": [1.0, 2.0, np.nan],
        "cat": ["a", "b", "a"]
    })
    
    selector = WeightedFuzzyRoughSelector(k=2)
    distances = selector._compute_HEC(X)
    
    assert distances.shape == (3,3)
    assert np.all(distances >= 0)


# ----- _compute_density -----------------------------------------------------

def test_compute_density_simple():
    distances = np.array([[0,1,2],[1,0,3],[2,3,0]])
    knn_indices = np.array([[1,2],[0,2],[0,1]])
    
    selector = WeightedFuzzyRoughSelector(k=2)
    rho = selector._compute_density(distances, knn_indices)
    
    assert rho.shape[0] == 3
    assert np.all(rho > 0)


# ----- _compute_LDF ---------------------------------------------------------
def test_compute_LDF_simple():
    rho = np.array([1.0, 2.0, 1.5])
    knn_indices = np.array([[1,2],[0,2],[0,1]])
    
    selector = WeightedFuzzyRoughSelector(k=2)
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
    selector = WeightedFuzzyRoughSelector(k=2, alpha=0.5)
    selector._compute_HEC = lambda X1, X2=None, W=None: np.zeros((3,2))
    
    relations_single, relations_pair = selector._compute_fuzzy_similarity_relations(X,H)
    
    assert isinstance(relations_single, dict)
    assert isinstance(relations_pair, dict)
    assert all(0 <= v.max() <= 1 for v in relations_single.values())


# ----- _compute_relation_for_subset ----------------------------------------

def test_compute_relation_for_subset_basic():
    X = pd.DataFrame({"a":[1,2,1], "b":[3,4,3]})
    H = [0,2]
    feature_subset = [0]
    selector = WeightedFuzzyRoughSelector(alpha=0.5)
    selector._compute_HEC = lambda X_sub, X_H, W=None: np.zeros((3,2))
    
    relation = selector._compute_relation_for_subset(X,H,feature_subset)
    
    assert relation.shape == (3, len(H))
    assert np.all(relation <= 1)


# ----- _compute_POS_NOG -----------------------------------------------------

def test_compute_POS_NOG_basic():
    rel_matrix = np.array([[1,0],[0,1]])
    relations_single = {0: rel_matrix}
    y = np.array([0,1])
    H = [0,1]
    
    selector = WeightedFuzzyRoughSelector()
    POS, NOG = selector._compute_POS_NOG(relations_single, y, H)
    
    assert 0 in POS
    assert 0 in NOG
    assert len(POS[0]) == len(y)


# ----- _compute_relevance ---------------------------------------------------

def test_compute_relevance_basic():    
    POS_all = {0: np.array([0.1,0.2]), 1: np.array([0.3,0.4])}
    NOG_all = {0: np.array([0.1,0.1]), 1: np.array([0.2,0.2])}
    
    selector = WeightedFuzzyRoughSelector()
    relevance = selector._compute_relevance(POS_all,NOG_all)
    
    assert set(relevance.keys()) == {0,1}
    assert all(0 <= v <= 1 for v in relevance.values())


# ----- _compute_redundancy --------------------------------------------------

def test_compute_redundancy_basic():
    POS_all = {0: np.array([0.1,0.2])}
    NOG_all = {0: np.array([0.1,0.2])}
    relations_pair = {(0,1): np.array([[1,1],[1,1]])}
    relevance = {0:0.2, 1:0.2}
    y = np.array([0,1])
    H = [0,1]
    
    selector = WeightedFuzzyRoughSelector()
    selector._compute_POS_NOG = lambda rels, y, H: ({0: np.array([0.1,0.1])}, {0: np.array([0.1,0.1])})
    
    redundancy = selector._compute_redundancy(y,H,relevance,relations_pair)
    assert isinstance(redundancy, dict)


# ----- _compute_feature_weights --------------------------------------------

def test_compute_feature_weights_basic():
    relevance = {0:0.2, 1:0.4}
    redundancy = {(0,1):0.3}
    
    selector = WeightedFuzzyRoughSelector()
    weights = selector._compute_feature_weights(relevance, redundancy)
    
    assert isinstance(weights, dict)
    assert set(weights.keys()) == {0,1}


# ----- _update_weight_matrix -----------------------------------------------

def test_update_weight_matrix_basic():
    weights = {0:0.0,1:1.0}
    selector = WeightedFuzzyRoughSelector()
    W = selector._update_weight_matrix(weights, n_total_features=2)
    
    assert W.shape == (2,2)
    assert np.all(W >= 0)


# ----- _compute_gamma -------------------------------------------------------

def test_compute_gamma_basic():
    POS_all = {0: np.array([0.2,0.4]), 1: np.array([0.1,0.3])}
    NOG_all = {0: np.array([0.1,0.1]), 1: np.array([0.2,0.2])}
    features = [0,1]
    
    selector = WeightedFuzzyRoughSelector()
    gamma_P, gamma_N = selector._compute_gamma(POS_all, NOG_all, features)
    
    assert 0 <= gamma_P <= 1
    assert 0 <= gamma_N <= 1


# ----- _compute_separability -----------------------------------------------

def test_compute_separability_basic():
    X = pd.DataFrame({"a":[1,2,1],"b":[3,4,3]})
    y = np.array([0,1,0])
    H = [0,2]
    W = np.eye(2)
    selected_features = []
    remaining_features = [0,1]
    
    selector = WeightedFuzzyRoughSelector(alpha=0.5)
    selector._compute_relation_for_subset = lambda X,H,B,W=None: np.zeros((3,len(B)))
    selector._compute_POS_NOG = lambda rels,y,H: ({f: np.zeros(3) for f in rels},{f: np.zeros(3) for f in rels})
    selector._compute_gamma = lambda POS,NOG,features: (0.1,0.1)
    
    sep = selector._compute_separability(X,y,H,W,selected_features,remaining_features)
    assert isinstance(sep, dict)


# ----- _build_weighted_feature_sequence -----------------------------------

def test_build_weighted_feature_sequence_basic():
    X = pd.DataFrame({"a":[1,2,1],"b":[3,4,3]})
    y = np.array([0,1,0])
    H = [0,2]

    POS_all = {0: np.zeros(3).reshape(3,1), 1: np.zeros(3).reshape(3,1)}
    NOG_all = {0: np.zeros(3).reshape(3,1), 1: np.zeros(3).reshape(3,1)}
    weights = {0:0.2, 1:0.4}

    selector = WeightedFuzzyRoughSelector(alpha=0.5)

    selector._compute_separability = lambda X,y,H,W,selected,remaining: {f:0.1 for f in remaining}
    selector._compute_POS_NOG = lambda rels,y,H: ({f: np.zeros(3) for f in rels},{f: np.zeros(3) for f in rels})
    selector._compute_relevance = lambda POS,NOG: {0:0.2,1:0.3}
    selector._compute_redundancy = lambda y,H,relevance,relations_pair,cached_POS_NOG=None: {(0,1):0.1}
    selector._compute_feature_weights = lambda relevance, redundancy: {0:0.2,1:0.3}
    selector._update_weight_matrix = lambda weights,n: np.eye(n)

    sequence, Rw = selector._build_weighted_feature_sequence(POS_all,NOG_all,weights,X,y,H,POS_all,NOG_all)

    assert set(sequence) == {0,1}
    assert Rw.shape == (2,2)

