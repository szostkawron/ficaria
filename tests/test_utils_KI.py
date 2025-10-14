from ficaria.utils import get_neighbors, find_best_k, check_input_dataset, impute_KI, compute_fcm_objective, \
    find_optimal_clusters_fuzzy, impute_FCKI
import pandas as pd
import numpy as np
import pytest
import random
from sklearn.datasets import make_blobs
from fuzzycmeans import FCM
from sklearn.impute import SimpleImputer


@pytest.mark.parametrize(
    "train, test_row, num_neighbors, expected",
    [
        (
                [[1, 2], [3, 4], [5, 6], [7, 8]],
                [1, 2],
                2,
                [[1, 2], [3, 4]]
        ),
        (
                [[1, 2], [3, 4], [5, 6], [7, 8]],
                [5, 6],
                1,
                [[5, 6]]
        ),
        (
                [[1, 2], [3, 4], [5, 6]],
                [2, 3],
                2,
                [[1, 2], [3, 4]]
        )
    ]
)
def test_get_neighbors(train, test_row, num_neighbors, expected):
    result = get_neighbors(train, test_row, num_neighbors)
    assert result == expected
    assert len(result) == num_neighbors


@pytest.mark.parametrize("St, random_col, original_value", [
    (
            pd.DataFrame({
                'height_cm': [165, 170, np.nan, 180, 175, 160, np.nan, 190],
                'weight_kg': [60, 65, 70, np.nan, 80, 55, 68, np.nan],
                'bmi': [22.0, 22.5, 24.2, 26.5, 26.1, 21.5, 23.8, np.nan]}),
            1, 85),
    (
            pd.DataFrame({
                'age': [25, 26, np.nan, 51, 53, 72, 75],
                'income': [50000, 55000, 80000, 85000, 90000, 120000, np.nan]}),
            0, 125000)
])
def test_find_best_k(St, random_col, original_value):
    result1 = find_best_k(St, random_col, original_value)
    result2 = find_best_k(St, random_col, original_value)
    assert result1 == result2
    assert isinstance(result1, int)
    assert result1 > 0
    assert result1 <= len(St)


@pytest.mark.parametrize("X", [
    ([[1, 2, 3], [3, 7, 4]]),
    (pd.DataFrame({
        'height_cm': [165, 170, 175, 180, 175, 160, 175, 190],
        'weight_kg': [60, 65, 70, 75, 80, 55, 68, 80],
        'bmi': [22.0, 22.5, 24.2, 26.5, 26.1, 21.5, 23.8, 25.0]})),
    (np.array([[5, 1], [4, 5]])),
    ([(1, 2), (3, 4)]),
    ([[1, "txt", 3], [3, "txt", 4]]),
    ([[1, 2], [3, "txt"]]),
    (pd.DataFrame({
        'height_cm': [165, 170, np.nan, 180, 175, 160, np.nan, 190],
        'weight_kg': [60, 65, 70, np.nan, 80, 55, 68, np.nan],
        'bmi': [22.0, 22.5, 24.2, 26.5, 26.1, 21.5, 23.8, np.nan]})),
])
def test_check_input_dataset_valid_cases(X):
    result = check_input_dataset(X)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] > 0


@pytest.mark.parametrize("X, require_numeric, allow_nan", [
    ([[1, 2], [1, 5, 7]], False, True),
])
def test_check_input_dataset_wrong_input_types(X, require_numeric, allow_nan):
    with pytest.raises(TypeError,
                       match="Invalid input type: Expected a 2D structure such as a DataFrame, NumPy array, or similar tabular format."):
        check_input_dataset(X, require_numeric=require_numeric, allow_nan=allow_nan)


@pytest.mark.parametrize("X, require_numeric, allow_nan", [
    (5, False, True),
    ("txt", False, True),
    ([5], False, True),
    ([1, 2, 3], False, True),
    ([[[1, 2], [3, 4]], [[5, 7], [3, 6]]], False, True),
    ([{'a': 1}, {'a': 2}], False, True),
    (object(), False, True),
    (5 + 3j, False, True),
    (lambda x: x, False, True)
])
def test_check_input_dataset_wrong_input_dimensions(X, require_numeric, allow_nan):
    with pytest.raises(ValueError,
                       match="Input must be 2-dimensional."):
        check_input_dataset(X, require_numeric=require_numeric, allow_nan=allow_nan)


@pytest.mark.parametrize("X, require_numeric, allow_nan", [
    ([[1, "txt", 3], [3, 7, 4]], True, True),
    (pd.DataFrame({
        'height_cm': [165, 170, 175, 180, 175, 160, 175, 190],
        'weight_kg': [60, 65, 70, 75, 80, 55, 68, 80],
        'bmi': [22.0, 22.5, 24.2, 26.5, 26.1, 21.5, 23.8, 25.0],
        'name': ["txt", "txt", "txt", "txt", "txt", "txt", "txt", "txt"]
    }), True, True),
])
def test_check_input_dataset_check_require_numeric_true(X, require_numeric, allow_nan):
    with pytest.raises(TypeError,
                       match="All columns must be numeric."):
        check_input_dataset(X, require_numeric=require_numeric, allow_nan=allow_nan)


@pytest.mark.parametrize("X, require_numeric, allow_nan", [
    ([[1, np.nan, 3], [3, 7, 4]], False, False),
    (pd.DataFrame({
        'height_cm': [165, 170, np.nan, 180, 175, 160, np.nan, 190],
        'weight_kg': [60, 65, 70, np.nan, 80, 55, 68, np.nan],
        'bmi': [22.0, 22.5, 24.2, 26.5, 26.1, 21.5, 23.8, np.nan]
    }), False, False),
])
def test_check_input_dataset_check_allow_nan_false(X, require_numeric, allow_nan):
    with pytest.raises(ValueError,
                       match="Missing values are not allowed."):
        check_input_dataset(X, require_numeric=require_numeric, allow_nan=allow_nan)


@pytest.mark.parametrize("X", [
    (pd.DataFrame({
        'height_cm': [165, 170, 175, 180, 175, 160, 175, 190],
        'weight_kg': [60, 65, 70, 75, 80, 55, 68, 80],
        'bmi': [22.0, 22.5, 24.2, 26.5, 26.1, 21.5, 23.8, 25.0]})),
    (pd.DataFrame({
        'height_cm': [165, 170, np.nan, 180, 175, 160, np.nan, 190],
        'weight_kg': [60, 65, 70, np.nan, 80, 55, 68, np.nan],
        'bmi': [22.0, 22.5, 24.2, 26.5, 26.1, 21.5, 23.8, np.nan]})),
    (pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6],
        'c': [7, 8, 9]
    })),
    (pd.DataFrame({
        'a': [1, np.nan, 3],
        'b': [4, 5, np.nan],
        'c': [7, 8, 9]
    }))
])
def test_impute_KI(X):
    result = impute_KI(X)
    print(len(result))
    print(result)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(X)
    assert not np.isnan(result).any()


@pytest.mark.parametrize("X, X_train, rng1, rng2", [
    (
            pd.DataFrame({
                'a': [1, np.nan, 3],
                'b': [4, 5, np.nan],
                'c': [7, 8, 9]
            }),
            pd.DataFrame({
                'a': [10, 11, 12],
                'b': [13, 14, 15],
                'c': [16, 17, 18]
            }),
            random.Random(42),
            random.Random(42)
    ),
    (
            pd.DataFrame({
                'a': [1, np.nan, 3],
                'b': [4, 5, np.nan],
                'c': [7, 8, 9]
            }),
            pd.DataFrame({
                'a': [1, np.nan, 3],
                'b': [4, 5, np.nan],
                'c': [7, 8, 9]
            }),
            random.Random(123),
            random.Random(123)
    ),
    (
            pd.DataFrame({
                'a': [1, np.nan, 3],
                'b': [4, 5, np.nan],
                'c': [7, 8, 9]
            }),
            None,
            random.Random(42),
            random.Random(42)
    )
])
def test_impute_KI_with_parameters(X, X_train, rng1, rng2):
    result = impute_KI(X, X_train=X_train, rng=rng1)
    assert isinstance(result, np.ndarray)
    assert result.shape == X.shape
    assert not np.isnan(result).any()
    result_repeat = impute_KI(X, X_train=X_train, rng=rng2)
    np.testing.assert_array_almost_equal(result, result_repeat)


@pytest.mark.parametrize("X, centers, U, m, expected", [
    (
            np.array([[1.0, 2.0],
                      [3.0, 4.0]]),
            np.array([[1.0, 2.0],
                      [3.0, 4.0]]),
            np.array([[1.0, 0.0],
                      [0.0, 1.0]]),
            2, 0.0),
    (
            np.array([[0.0],
                      [10.0]]),
            np.array([[0.0],
                      [10.0]]),
            np.array([[0.5, 0.5],
                      [0.5, 0.5]]),
            2, 50.0),
    (
            np.array([[1.0], [2.0]]),
            np.array([[1.0], [2.0]]),
            np.array([[0.6, 0.4],
                      [0.3, 0.7]]),
            1, 0.7),
    (
            np.array([[1.0], [10.0]]),
            np.array([[10.0], [1.0]]),
            np.array([[1.0, 0.0],
                      [0.0, 1.0]]),
            2, 162.0),
])
def test_compute_fcm_objective(X, centers, U, m, expected):
    result = compute_fcm_objective(X, centers, U, m)
    assert isinstance(result, float)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("X, min_clusters, max_clusters, random_state, m, expected_k", [
    (
            pd.DataFrame(make_blobs(n_samples=150, centers=3, cluster_std=0.5, random_state=42)[0]),
            2, 6, 42, 2.0, 3
    ),
    (
            pd.DataFrame(make_blobs(n_samples=200, centers=4, cluster_std=0.4, random_state=1)[0]),
            2, 7, 1, 2.0, 4
    ),
    (
            pd.DataFrame(make_blobs(n_samples=100, centers=2, cluster_std=0.6, random_state=10)[0]),
            1, 5, 10, 2.0, 2
    ),
    (
            pd.DataFrame(make_blobs(n_samples=150, centers=8, cluster_std=0.6, random_state=42)[0]),
            5, 11, 42, 1.1, 8
    ),
    (
            pd.DataFrame(make_blobs(n_samples=150, centers=6, cluster_std=0.6, random_state=42)[0]),
            3, 9, 42, 1.1, 6
    ),
])
def test_find_optimal_clusters_fuzzy(X, min_clusters, max_clusters, random_state, m, expected_k):
    result = find_optimal_clusters_fuzzy(X, min_clusters, max_clusters, random_state, m)
    print(result)
    assert isinstance(result, int)
    assert abs(result - expected_k) <= 3


@pytest.mark.parametrize("X, X_train, n_clusters, seed", [
    (
            pd.DataFrame({
                "a": [1.0, np.nan, 3.0],
                "b": [4.0, 5.0, 6.0]
            }),
            pd.DataFrame({
                "a": [1.0, 2.0, 3.0],
                "b": [4.0, 5.0, 6.0]
            }),
            2, 42
    ),
    (
            pd.DataFrame({
                "x": [np.nan, 2.5, 3.0, 4.5],
                "y": [1.0, np.nan, 3.0, 4.0]
            }),
            pd.DataFrame({
                "x": [1.0, 2.0, 3.0, 4.0],
                "y": [1.0, 2.0, 3.0, 4.0]
            }),
            2, 42
    ),
    (
            pd.DataFrame({
                "x": [1.0, 2.0, 3.0],
                "y": [4.0, 5.0, 6.0]
            }),
            pd.DataFrame({
                "x": [1.0, 2.0, 3.0],
                "y": [4.0, 5.0, 6.0]
            }),
            2, 42)
])
def test_impute_FCKI(X, X_train, n_clusters, seed):
    rng = random.Random(seed)
    imputer = SimpleImputer(strategy="mean")
    X_train_filled = imputer.fit_transform(X_train)
    X_train_filled_df = pd.DataFrame(X_train_filled, columns=X_train.columns)
    np.random.seed(seed)
    fcm = FCM(n_clusters=n_clusters)
    fcm.fit(X_train_filled_df.values)
    result = impute_FCKI(X, X_train, fcm, n_clusters, imputer, rng)
    assert isinstance(result, np.ndarray)
    assert result.shape == X.shape
    assert not np.isnan(result).any()
