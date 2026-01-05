import os
import sys

import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ficaria.feature_selection import *


# ----- FuzzyGranularitySelector -----------------------------------------------

@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        "num1": [0.1, 0.4, 0.5, 0.9, 0.3],
        "num2": [1, 2, 1, 2, 1],
        "cat1": ["x", "y", "x", "x", "y"]
    })
    y = pd.Series([0, 1, 0, 1, 0])
    return X, y


def test_transform_single_row(sample_data):
    X, y = sample_data
    selector = FuzzyGranularitySelector()

    selector.fit(X, y)

    single_row = X.iloc[[0]]

    X_trans = selector.transform(single_row)

    assert X_trans.shape[0] == 1, "Transforming a single row should return exactly one row"
    assert X_trans.shape[1] > 0, "Transform result should contain at least one column"
    assert not X_trans.isnull().values.any(), "Transform result should not contain any NaN values"


def test_deterministic_results(sample_data):
    X, y = sample_data
    selector1 = FuzzyGranularitySelector(eps=0.5, random_state=123)
    selector2 = FuzzyGranularitySelector(eps=0.5, random_state=123)

    selector1.fit(X, y)
    selector2.fit(X, y)
    assert selector1.S_ == selector2.S_
    transformed1 = selector1.transform(X)
    transformed2 = selector2.transform(X)
    pd.testing.assert_frame_equal(transformed1, transformed2)


def test_transform_without_fit(sample_data):
    X, y = sample_data
    selector = FuzzyGranularitySelector()
    with pytest.raises(NotFittedError):
        selector.transform(X)


def test_init_valid(sample_data):
    selector = FuzzyGranularitySelector(n_features=3, eps=0.5, max_features=5, random_state=42)
    assert selector.random_state == 42
    assert selector.eps == 0.5
    assert selector.d == 5
    assert selector.k == 3


def test_fit_and_transform(sample_data):
    X, y = sample_data
    selector = FuzzyGranularitySelector(eps=0.5, max_features=3, random_state=42)
    selector.fit(X, y)
    transformed = selector.transform(X)
    assert isinstance(transformed, pd.DataFrame)
    assert all(col in X.columns for col in transformed.columns)


def test_fit_invalid_input_types(sample_data):
    X, y = sample_data
    selector = FuzzyGranularitySelector()

    with pytest.raises(ValueError):
        selector.fit(None, y)

    with pytest.raises(ValueError):
        selector.fit(pd.DataFrame(), y)

    with pytest.raises(ValueError):
        selector.fit(X, y.iloc[:-1])

    invalid_inputs = [
        [1, 2, 3],
        "not a dataframe",
        123,
        pd.Series([1, 2, 3]),
        np.array([1, 2, 3])
    ]
    for bad_X in invalid_inputs:
        with pytest.raises(ValueError):
            selector.fit(bad_X, y)

    arr = np.array(X)
    selector2 = FuzzyGranularitySelector()
    selector2.fit(arr, y)

    ll = X.values.tolist()
    selector3 = FuzzyGranularitySelector()
    selector3.fit(ll, y)


@pytest.mark.parametrize(
    "n_features, max_features, expected_exception, expected_msg", [
        (10, 1, ValueError, "n_features must be <= max_features:"),
        (101, 100, ValueError, "n_features must be <= max_features:"),
    ])
def test_fuzzygranularityselector_init_errors_n_max_features(n_features, max_features, expected_exception,
                                                             expected_msg):
    with pytest.raises(expected_exception, match=expected_msg):
        FuzzyGranularitySelector(n_features=n_features, max_features=max_features)


@pytest.mark.parametrize(
    "X, y, expected_exception, expected_msg", [
        (np.zeros((5, 3)), np.zeros(4), ValueError, "y must have the same number of rows as X:"),
        (pd.DataFrame(np.zeros((10, 2))), pd.Series(np.zeros(9)), ValueError,
         "y must have the same number of rows as X:"),
        (np.zeros((6, 2)), pd.DataFrame(np.zeros((5, 1))), ValueError, "y must have the same number of rows as X:"),
    ]
)
def test_function_raises_error_if_y_length_mismatch(X, y, expected_exception, expected_msg):
    selector = FuzzyGranularitySelector()
    with pytest.raises(expected_exception, match=expected_msg):
        selector.fit(X, y)


def test_missing_values_in_X_raises(sample_data):
    X, y = sample_data
    X_nan = X.copy()
    X_nan.loc[0, "a"] = np.nan
    selector = FuzzyGranularitySelector()
    with pytest.raises(ValueError):
        selector.fit(X_nan, y)


def test_inconsistent_columns_between_fit_and_transform(sample_data):
    X, y = sample_data
    selector = FuzzyGranularitySelector()
    selector.fit(X, y)

    X_bad_order = X[["num2", "num1", "cat1"]].copy()
    with pytest.raises(ValueError, match="X.columns must match the columns seen during fit"):
        selector.transform(X_bad_order)

    X_missing = X.drop(columns=["cat1"])
    with pytest.raises(ValueError, match="X.columns must match the columns seen during fit"):
        selector.transform(X_missing)

    X_extra = X.copy()
    X_extra["extra"] = [0, 0, 0, 0, 0]
    with pytest.raises(ValueError, match="X.columns must match the columns seen during fit"):
        selector.transform(X_extra)


def test_unsupervised_mode_y_none(sample_data):
    X, _ = sample_data
    selector = FuzzyGranularitySelector(eps=0.5, max_features=3, random_state=42)
    selector.fit(X, y=None)
    transformed = selector.transform(X)
    assert isinstance(transformed, pd.DataFrame)
    assert transformed.shape[0] == X.shape[0]


def test_mixed_numerical_and_categorical():
    X = pd.DataFrame({
        "num1": [1.2, 3.4, 2.2, 4.8, 3.1],
        "num2": [10, 15, 10, 20, 15],
        "cat1": ["red", "blue", "red", "green", "blue"],
        "cat2": ["A", "B", "A", "A", "B"]
    })
    y = pd.Series([0, 1, 0, 1, 0])

    selector = FuzzyGranularitySelector(eps=0.5, max_features=3, random_state=42)
    selector.fit(X, y)

    transformed = selector.transform(X)

    assert isinstance(transformed, pd.DataFrame)
    assert set(transformed.columns).issubset(set(X.columns))

    for col in transformed.columns:
        if X[col].dtype == object:
            assert transformed[col].dtype == object


def test__calculate_similarity_matrix_for_df_numeric(sample_data):
    X, y = sample_data
    selector = FuzzyGranularitySelector(eps=0.5)
    selector.fit(X, y)
    mat = selector._calculate_similarity_matrix_for_df('num2', X)
    assert isinstance(mat, np.ndarray)
    assert mat.shape == (len(X), len(X))
    assert np.all((mat >= 0) & (mat <= 1))


def test__calculate_similarity_matrix_for_df_nominal(sample_data):
    X, y = sample_data
    selector = FuzzyGranularitySelector()
    selector.fit(X, y)
    mat = selector._calculate_similarity_matrix_for_df('cat1', X)
    assert mat.shape == (len(X), len(X))
    assert np.all((mat == 0) | (mat == 1))


def test__calculate_delta_for_column_subset_global_and_local(sample_data):
    X, y = sample_data
    selector = FuzzyGranularitySelector()
    selector.fit(X, y)

    B = ['num1', 'num2']
    granule_vec, size = selector._calculate_delta_for_column_subset(0, B)
    assert isinstance(granule_vec, np.ndarray)
    assert isinstance(size, float)
    assert size >= 0

    granule_vec_local, size_local = selector._calculate_delta_for_column_subset(0, B, df=X)
    assert isinstance(granule_vec_local, np.ndarray)
    assert size_local >= 0


def test__calculate_multi_granularity_fuzzy_implication_entropy_basic(sample_data):
    X, y = sample_data
    selector = FuzzyGranularitySelector()
    selector.fit(X, y)

    ent = selector._calculate_multi_granularity_fuzzy_implication_entropy(['num1', 'num2'], type="basic")
    assert isinstance(ent, float)
    assert ent >= 0


@pytest.mark.parametrize("etype", ["basic", "conditional", "joint", "mutual"])
def test__calculate_multi_granularity_fuzzy_implication_entropy_types(sample_data, etype):
    X, y = sample_data
    selector = FuzzyGranularitySelector()
    selector.fit(X, y)

    val = selector._calculate_multi_granularity_fuzzy_implication_entropy(["num2"], type=etype)
    assert isinstance(val, float)
    assert val >= 0


def test__granual_consistency_of_B_subset(sample_data):
    X, y = sample_data
    selector = FuzzyGranularitySelector()
    selector.fit(X, y)

    score = selector._granular_consistency_of_B_subset(["cat1"])
    assert isinstance(score, float)
    assert 0 <= score <= 1


def test__local_granularity_consistency_of_B_subset(sample_data):
    X, y = sample_data
    selector = FuzzyGranularitySelector()
    selector.fit(X, y)

    val = selector._local_granularity_consistency_of_B_subset(['num2'])
    assert isinstance(val, float)
    assert 0 <= val <= 1


def test__create_partitions(sample_data):
    X, y = sample_data
    selector = FuzzyGranularitySelector()
    selector.fit(X, y)

    parts = selector._create_partitions()
    assert isinstance(parts, dict)
    assert set(parts.keys()) == set(y.unique())
    for df_part in parts.values():
        assert isinstance(df_part, pd.DataFrame)
        assert selector.target_name_ in df_part.columns


def test__FIGFS_algorithm_executes(sample_data):
    X, y = sample_data
    selector = FuzzyGranularitySelector(max_features=4)
    selector.fit(X, y)

    result = selector._FIGFS_algorithm()
    assert isinstance(result, list)
    assert all(isinstance(i, str) for i in result)


def test_d_less_than_number_of_features():
    np.random.seed(42)
    X = pd.DataFrame({
        "num1": np.random.rand(10),
        "num2": np.random.rand(10),
        "num3": np.random.rand(10),
        "num4": np.random.rand(10),
        "num5": np.random.rand(10),
        "num6": np.random.rand(10)
    })
    y = np.random.randint(0, 2, size=10)

    selector = FuzzyGranularitySelector(max_features=3)

    selector.fit(X, y)
    X_transformed = selector.transform(X)

    assert len(selector.S_) <= selector.d, f"Selected features ({len(selector.S_)}) exceed d={selector.d}"
    assert X_transformed.shape[1] == len(selector.S_)

    for colname in selector.S_:
        assert colname in X.columns


def test_transform_single_row():
    X = pd.DataFrame({
        "num1": [0.5],
        "num2": [1.2],
        "num3": [0.7]
    })
    y = pd.Series([1])

    selector = FuzzyGranularitySelector(n_features=2, max_features=3, eps=0.5, random_state=42)
    selector.fit(X, y)
    X_transformed = selector.transform(X)

    assert X_transformed.shape[0] == 1, "Number of rows should remain 1"
    assert X_transformed.shape[1] == min(selector.k,
                                         len(selector.S_)), "Number of columns should match selected features"

    for col in X_transformed.columns:
        assert col in X.columns


def test_d_greater_than_number_of_features(sample_data):
    X, y = sample_data

    selector = FuzzyGranularitySelector(max_features=10, n_features=3, eps=0.5, random_state=42)
    selector.fit(X, y)
    X_transformed = selector.transform(X)

    assert len(selector.S_) <= X.shape[1], "Number of selected features should not exceed number of features"
    assert X_transformed.shape[1] == len(selector.S_)

    for col in X_transformed.columns:
        assert col in X.columns


# ----- WeightedFuzzyRoughSelector -----------------------------------------------

dataframes_list = [
    pd.DataFrame({
        "num1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "num2": [4.0, 5.0, 6.0, 7.0, 6.0],
        "cat1": ["n_features", "n_features", "m", "m", "m"],
    }),

    pd.DataFrame({
        "num1": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        "num2": [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    }),

    pd.DataFrame({
        "num1": np.random.rand(10),
        "num2": np.random.rand(10),
        "num3": np.random.rand(10)
    }),

    pd.DataFrame({
        "num1": [1, 2, np.nan, 4, 5, 6, 7],
        "num2": [7, 6, 5, np.nan, 3, 2, 1],
        "cat1": ["x", np.nan, "z", "x", "y", "z", "x"],
        "num4": [0.1, 0.2, 0.3, 0.4, 0.5, np.nan, 0.7]
    }),

    pd.DataFrame({
        "num1": np.random.randint(0, 100, size=50),
        "num2": np.random.rand(50),
        "cat1": np.random.choice(["red", "blue", "green"], size=50),
        "num3": np.random.randn(50),
        "num4": np.random.rand(50),
    }),

    pd.DataFrame({
        "cat1": ["txt", "txt", "txt", "csv", "csv"],
        "cat2": ["A", "B", "B", "C", "C"],
        "cat3": ["n_features", "n_features", "m", "m", "m"],
    }),
]

y_list = [
    np.array([0, 1, 0, 1, 1]),
    np.array([1, 1, 1, 0, 0, 0, 0]),
    np.random.randint(0, 2, size=10),
    np.array([0, 1, 0, 1, 1, 0, 1]),
    np.random.randint(0, 3, size=50),
    np.array(['y', 'n', 'y', 'n', 'y']),
]

selector_params_list = [
    (1, 0.1, 2),
    (1, 0.5, 3),
    (2, 0.6, 4),
    (2, 1.0, 5),
    (3, 0.25, 2),
    (3, 0.75, 6)
]


@pytest.mark.parametrize("alpha, expected_exception, expected_msg", [
    ("xyz", TypeError, "alpha must be int or float, got"),
    ([3], TypeError, "alpha must be int or float, got"),
    (-0.1, ValueError, r"alpha must be in range \(0, 1\], got"),
    (0, ValueError, r"alpha must be in range \(0, 1\], got"),
    (1.5, ValueError, r"alpha must be in range \(0, 1\], got"),
])
def test_fcmcentroidimputer_fit_raises_if_too_many_clusters(alpha, expected_exception, expected_msg):
    X = pd.DataFrame({"num1": [1.0, 2.0, np.nan], "num2": [4.0, 5.0, 6.0]})
    with pytest.raises(expected_exception, match=expected_msg):
        WeightedFuzzyRoughSelector(n_features=2, alpha=alpha)


@pytest.mark.parametrize("n_features,alpha,k", selector_params_list)
def test_weightedfuzzyroughselector_init_parametrized(n_features, alpha, k):
    selector = WeightedFuzzyRoughSelector(n_features=n_features, alpha=alpha, k=k)
    assert selector.n_features == n_features
    assert selector.alpha == alpha
    assert selector.k == k


def test_weightedfuzzyroughselector_transform_raises_if_not_fitted():
    X = pd.DataFrame({"num1": [1.0, 2.0, 3.0], "num2": [4.0, 5.0, 6.0]})
    selector = WeightedFuzzyRoughSelector(n_features=1)
    with pytest.raises(NotFittedError):
        selector.transform(X)


def test_weightedfuzzyroughselector_fit_raises_if_y_has_missing_values():
    X = pd.DataFrame({"num1": [1.0, 2.0, 3.0], "num2": [4.0, 5.0, 6.0]})
    y = pd.Series([1, np.nan, 0])
    selector = WeightedFuzzyRoughSelector(n_features=1, k=2)

    with pytest.raises(ValueError, match="y must not contain missing values"):
        selector.fit(X, y)


def test_weightedfuzzyroughselector_fit_raises_if_y_length_mismatch():
    X = pd.DataFrame({"num1": [1.0, 2.0, 3.0], "num2": [4.0, 5.0, 6.0]})
    y = pd.Series([0, 1])
    selector = WeightedFuzzyRoughSelector(n_features=1, k=2)

    with pytest.raises(ValueError, match="y must have the same number of rows as X"):
        selector.fit(X, y)


def test_weightedfuzzyroughselector_transform_raises_if_columns_differ():
    X_train = pd.DataFrame({"num1": [1.0, 2.0, 3.0, 4.0, 5.0], "num2": [4.0, 5.0, 6.0, 7.0, 8.0]})
    X_test = pd.DataFrame({"num1": [1.0, 2.0, 3.0, 4.0, 5.0], "num3": [7.0, 8.0, 9.0, 10.0, 11.0]})
    y = np.array([0, 1, 0, 1, 0])

    selector = WeightedFuzzyRoughSelector(n_features=1, k=2)
    selector.fit(X_train, y)

    with pytest.raises(ValueError, match="X.columns must match the columns seen during fit"):
        selector.transform(X_test)


def test_weightedfuzzyroughselector_transform_single_row(sample_data):
    X, y = sample_data
    selector = WeightedFuzzyRoughSelector(n_features=2)

    selector.fit(X, y)

    single_row = X.iloc[[0]]

    X_trans = selector.transform(single_row)

    assert X_trans.shape[0] == 1, "Transforming a single row should return exactly one row"
    assert X_trans.shape[1] > 0, "Transform result should contain at least one column"
    assert not X_trans.isnull().values.any(), "Transform result should not contain any NaN values"


def test_weightedfuzzyroughselector_fit_creates_attributes():
    X = pd.DataFrame({"num1": [1.0, 2.0, 3.0, 4.0, 5.0], "num2": [4.0, 5.0, 6.0, 7.0, 8.0]})
    y = np.array([0, 1, 0, 1, 0])
    selector = WeightedFuzzyRoughSelector(n_features=1, k=2)
    selector.fit(X, y)

    assert hasattr(selector, "feature_sequence_")
    assert hasattr(selector, "feature_importances_")
    assert hasattr(selector, "Rw_")


def test_weightedfuzzyroughselector_transform_returns_reduced_features():
    X = pd.DataFrame({"num1": [1.0, 2.0, 3.0, 4.0, 5.0],
                      "num2": [4.0, 5.0, 6.0, 7.0, 8.0],
                      "num3": [1.0, 2.0, 3.0, 4.0, 5.0]})
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
            pytest.skip(f"Skipping n_features for this dataset, not enough samples in smallest class.")

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
    X = pd.DataFrame({"num1": [1, 2, 1, 2, 1]})
    y = np.array([0, 1, 0, 1, 0])

    selector = WeightedFuzzyRoughSelector(n_features=1, alpha=1.0, k=2)

    selector.fit(X, y)

    assert selector.feature_sequence_ == [0]
    assert selector.feature_importances_.shape[0] == 1
    assert selector.feature_importances_['feature'].iloc[0] == "num1"
    X_transformed = selector.transform(X)
    assert X_transformed.shape == X.shape
    assert (X_transformed['num1'] == X['num1']).all()
    assert selector.Rw_.shape == (1, 1)


def test_reproducibility_same_input():
    X = pd.DataFrame({
        "num1": [1, 2, 1, 2, 1],
        "num2": [5, 4, 5, 4, 5],
        "num3": [9, 8, 9, 8, 9]
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
        "num1": [1, 2, 3],
        "num2": [4, 5, 6],
        "num3": [7, 8, 9]
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
        "num1": [1, 2, 3, 4],
        "num2": [4, 3, 2, 1],
        "num3": [10, 20, 30, 40]
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
    X = pd.DataFrame({"num1": [1, 2], "num2": [3, 4]})
    y = ["a", "b"]
    selector = WeightedFuzzyRoughSelector(n_features=1, k=2)

    with pytest.raises(ValueError):
        selector.fit(X, y)


# ----- _identify_high_density_region ---------------------------------------

def test_identify_high_density_region():
    X = pd.DataFrame({
        "num1": [1.0, 2.0, 1.1, 3.0],
        "num2": [4.0, 5.0, 4.1, 6.0]
    })
    y = pd.Series([0, 1, 0, 1])

    selector = WeightedFuzzyRoughSelector(n_features=1, k=2)
    selector._compute_HEC = lambda X1: np.array([[0, 1, 0.1, 2],
                                                 [1, 0, 1.1, 1],
                                                 [0.1, 1.1, 0, 2.1],
                                                 [2, 1, 2.1, 0]])
    selector._compute_density = lambda distances, knn: np.ones(X.shape[0])
    selector._compute_LDF = lambda rho, knn: np.ones(X.shape[0])

    H_neighbors = selector._identify_high_density_region(X, y)

    assert isinstance(H_neighbors, np.ndarray)
    assert H_neighbors.size > 0


# ----- _compute_HEC ---------------------------------------------------------

def test_compute_HEC_numeric_and_categorical():
    X = pd.DataFrame({
        "num1": [1.0, 2.0, np.nan],
        "cat1": ["a", "b", "a"]
    })

    selector = WeightedFuzzyRoughSelector(n_features=1, k=2)
    distances = selector._compute_HEC(X)

    assert distances.shape == (3, 3)
    assert np.all(distances >= 0)


# ----- _compute_density -----------------------------------------------------

def test_compute_density_simple():
    distances = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    knn_indices = np.array([[1, 2], [0, 2], [0, 1]])

    selector = WeightedFuzzyRoughSelector(n_features=1, k=2)
    rho = selector._compute_density(distances, knn_indices)

    assert rho.shape[0] == 3
    assert np.all(rho > 0)


# ----- _compute_LDF ---------------------------------------------------------
def test_compute_LDF_simple():
    rho = np.array([1.0, 2.0, 1.5])
    knn_indices = np.array([[1, 2], [0, 2], [0, 1]])

    selector = WeightedFuzzyRoughSelector(n_features=1, k=2)
    LDF = selector._compute_LDF(rho, knn_indices)

    assert LDF.shape[0] == 3
    assert np.all(LDF > 0)


# ----- _compute_fuzzy_similarity_relations ---------------------------------

def test_compute_fuzzy_similarity_relations_small():
    X = pd.DataFrame({
        "num1": [1, 2, 1],
        "num2": [3, 4, 3]
    })
    H = [0, 2]
    selector = WeightedFuzzyRoughSelector(n_features=1, k=2, alpha=0.5)
    selector._compute_HEC = lambda X1, X2=None, W=None: np.zeros((3, 2))

    relations_single, relations_pair = selector._compute_fuzzy_similarity_relations(X, H)

    assert isinstance(relations_single, dict)
    assert isinstance(relations_pair, dict)
    assert all(0 <= v.max() <= 1 for v in relations_single.values())


# ----- _compute_redundancy --------------------------------------------------

def test_compute_redundancy_basic():
    POS_all = {0: np.array([0.1, 0.2])}
    NOG_all = {0: np.array([0.1, 0.2])}
    relations_pair = {(0, 1): np.array([[1, 1], [1, 1]])}
    relevance = {0: 0.2, 1: 0.2}
    y = np.array([0, 1])
    H = [0, 1]

    selector = WeightedFuzzyRoughSelector(n_features=1)
    selector._compute_POS_NOG = lambda rels, y, H: ({0: np.array([0.1, 0.1])}, {0: np.array([0.1, 0.1])})

    redundancy = selector._compute_redundancy(y, H, relevance, relations_pair)
    assert isinstance(redundancy, dict)


# ----- _compute_feature_weights --------------------------------------------

def test_compute_feature_weights_basic():
    relevance = {0: 0.2, 1: 0.4}
    redundancy = {(0, 1): 0.3}

    selector = WeightedFuzzyRoughSelector(n_features=1)
    weights = selector._compute_feature_weights(relevance, redundancy)

    assert isinstance(weights, dict)
    assert set(weights.keys()) == {0, 1}


# ----- _update_weight_matrix -----------------------------------------------

def test_update_weight_matrix_basic():
    weights = {0: 0.0, 1: 1.0}
    selector = WeightedFuzzyRoughSelector(n_features=1)
    W = selector._update_weight_matrix(weights, n_total_features=2)

    assert W.shape == (2, 2)
    assert np.all(W >= 0)


# ----- _compute_gamma -------------------------------------------------------

def test_compute_gamma_basic():
    POS_all = {0: np.array([0.2, 0.4]), 1: np.array([0.1, 0.3])}
    NOG_all = {0: np.array([0.1, 0.1]), 1: np.array([0.2, 0.2])}
    features = [0, 1]

    selector = WeightedFuzzyRoughSelector(n_features=1)
    gamma_P, gamma_N = selector._compute_gamma(POS_all, NOG_all, features)

    assert 0 <= gamma_P <= 1
    assert 0 <= gamma_N <= 1


# ----- _build_weighted_feature_sequence ------------------------------------

def test_build_weighted_feature_sequence_equal_weights():
    X = pd.DataFrame({
        'num1': [1, 2, 3, 4, 5, 6],
        'num2': [2, 3, 4, 5, 6, 7],
        'num3': [3, 4, 5, 6, 7, 8],
        'num4': [4, 5, 6, 7, 8, 9]
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
    y = np.array([0, 1, 0])
    H = [0, 1]

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
    y = np.array([0, 1, 0])
    H = [0, 1]

    selector = WeightedFuzzyRoughSelector(n_features=1)
    RelB, POS_B, NOG_B = selector._compute_relevance_B(R_B, y, H)

    assert isinstance(RelB, float)
    assert POS_B.shape[0] == len(y)
    assert NOG_B.shape[0] == len(y)


# ----- _compute_relation_for_subset --------------------------------------------------

def test_compute_relation_for_subset_basic():
    X = pd.DataFrame({
        'num1': [1, 2, 3],
        'num2': [4, 5, 6]
    })
    H = [0, 2]
    feature_subset = [0, 1]

    selector = WeightedFuzzyRoughSelector(n_features=2, alpha=0.5)
    relation = selector._compute_relation_for_subset(X, H, feature_subset)

    assert relation.shape == (len(X), len(H))
    assert np.all((relation >= 0) & (relation <= 1))


# ----- __compute_separability --------------------------------------------------

def test_compute_separability_basic():
    X = pd.DataFrame({
        'num1': [1, 2, 3],
        'num2': [4, 5, 6],
        'num3': [7, 8, 9]
    })
    y = np.array([0, 1, 0])
    H = [0, 2]

    selector = WeightedFuzzyRoughSelector(n_features=2, alpha=1.0)
    selector.W_ = np.eye(3)

    selected_features = []
    remaining_features = [0, 1, 2]

    separability = selector._compute_separability(X, y, H, selector.W_, selected_features, remaining_features)

    assert isinstance(separability, dict)
    assert set(separability.keys()) == set(remaining_features)
    assert all(isinstance(v, float) for v in separability.values())
