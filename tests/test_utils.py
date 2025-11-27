import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import euclidean
from sklearn.datasets import make_blobs

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ficaria.utils import split_complete_incomplete, euclidean_distance, fuzzy_c_means, validate_params, \
    check_input_dataset, compute_fcm_objective, find_optimal_clusters_fuzzy, fcm_predict


# ----- validate_params ---------------------------------------

@pytest.mark.parametrize(
    "params, expected_exception, expected_msg",
    [
        # n_clusters
        ({"n_clusters": "3"}, TypeError, "n_clusters must be int, got"),
        ({"n_clusters": [3]}, TypeError, "n_clusters must be int, got"),
        ({"n_clusters": 0}, ValueError, "n_clusters must be >= 1, got"),

        # max_clusters
        ({"max_clusters": "3"}, TypeError, "max_clusters must be int, got"),
        ({"max_clusters": [3]}, TypeError, "max_clusters must be int, got"),
        ({"max_clusters": 0.1}, TypeError, "max_clusters must be int, got"),
        ({"max_clusters": -1}, ValueError, "max_clusters must be >= 1, got"),
        ({"max_clusters": 0}, ValueError, "max_clusters must be >= 1, got"),

        # max_iter
        ({"max_iter": "100"}, TypeError, "max_iter must be int, got"),
        ({"max_iter": [100]}, TypeError, "max_iter must be int, got"),
        ({"max_iter": 2.5}, TypeError, "max_iter must be int, got"),
        ({"max_iter": 1}, ValueError, "max_iter must be > 1, got"),
        ({"max_iter": -5}, ValueError, "max_iter must be > 1, got"),

        # max_FCM_iter
        ({"max_FCM_iter": "100"}, TypeError, "max_FCM_iter must be int, got"),
        ({"max_FCM_iter": [100]}, TypeError, "max_FCM_iter must be int, got"),
        ({"max_FCM_iter": 2.5}, TypeError, "max_FCM_iter must be int, got"),
        ({"max_FCM_iter": 1}, ValueError, "max_FCM_iter must be > 1, got"),
        ({"max_FCM_iter": -5}, ValueError, "max_FCM_iter must be > 1, got"),

        # max_II_iter
        ({"max_II_iter": "100"}, TypeError, "max_II_iter must be int, got"),
        ({"max_II_iter": [100]}, TypeError, "max_II_iter must be int, got"),
        ({"max_II_iter": 2.5}, TypeError, "max_II_iter must be int, got"),
        ({"max_II_iter": 1}, ValueError, "max_II_iter must be > 1, got"),
        ({"max_II_iter": -5}, ValueError, "max_II_iter must be > 1, got"),

        # max_outer_iter
        ({"max_outer_iter": "100"}, TypeError, "max_outer_iter must be int, got"),
        ({"max_outer_iter": [100]}, TypeError, "max_outer_iter must be int, got"),
        ({"max_outer_iter": 2.5}, TypeError, "max_outer_iter must be int, got"),
        ({"max_outer_iter": 0}, ValueError, "max_outer_iter must be >= 1, got"),
        ({"max_outer_iter": -5}, ValueError, "max_outer_iter must be >= 1, got"),

        # max_k
        ({"max_k": "100"}, TypeError, "max_k must be int, got"),
        ({"max_k": [100]}, TypeError, "max_k must be int, got"),
        ({"max_k": 2.5}, TypeError, "max_k must be int, got"),
        ({"max_k": 0}, ValueError, "max_k must be >= 1, got"),
        ({"max_k": -5}, ValueError, "max_k must be >= 1, got"),

        # random_state
        ({"random_state": "abc"}, TypeError, "random_state must be int or None, got"),
        ({"random_state": [10]}, TypeError, "random_state must be int or None, got"),
        ({"random_state": 2.5}, TypeError, "random_state must be int or None, got"),

        # m (fuzziness)
        ({"m": "2.0"}, TypeError, "m must be int or float, got"),
        ({"m": [2.0]}, TypeError, "m must be int or float, got"),
        ({"m": 1.0}, ValueError, "m must be > 1.0, got"),
        ({"m": -3}, ValueError, "m must be > 1.0, got"),

        # tol
        ({"tol": "1e-5"}, TypeError, "tol must be int or float, got"),
        ({"tol": [1e-5]}, TypeError, "tol must be int or float, got"),
        ({"tol": 0}, ValueError, "tol must be > 0, got"),
        ({"tol": -1.0}, ValueError, "tol must be > 0, got"),

        # wl
        ({"wl": "0.5"}, TypeError, "wl must be int or float, got"),
        ({"wl": [0.5]}, TypeError, "wl must be int or float, got"),
        ({"wl": -0.1}, ValueError, "wl must be in range (0, 1], got"),
        ({"wl": 1.5}, ValueError, "wl must be in range (0, 1], got"),

        # wb
        ({"wb": "0.2"}, TypeError, "wb must be int or float, got"),
        ({"wb": [0.2]}, TypeError, "wb must be int or float, got"),
        ({"wb": -0.1}, ValueError, "wb must be in range [0, 1], got"),
        ({"wb": 1.5}, ValueError, "wb must be in range [0, 1], got"),

        # tau
        ({"tau": "0.5"}, TypeError, "tau must be int or float, got"),
        ({"tau": [0.5]}, TypeError, "tau must be int or float, got"),
        ({"tau": -0.1}, ValueError, "tau must be >= 0, got"),

        # k
        ({"k": "ABC"}, TypeError, "k must be int, got"),
        ({"k": [5]}, TypeError, "k must be int, got"),
        ({"k": 3.5}, TypeError, "k must be int, got"),
        ({"k": 0}, ValueError, "k must be > 1, got"),
        ({"k": -3}, ValueError, "k must be > 1, got"),

        # n_features
        ({"n_features": "ABC"}, TypeError, "n_features must be int, got"),
        ({"n_features": [6]}, TypeError, "n_features must be int, got"),
        ({"n_features": 7.9}, TypeError, "n_features must be int, got"),
        ({"n_features": -3}, ValueError, "n_features must be >= 1, got"),
        ({"n_features": 0}, ValueError, "n_features must be >= 1, got"),

        # max_features
        ({"max_features": "ABC"}, TypeError, "max_features must be int, got "),
        ({"max_features": [2]}, TypeError, "max_features must be int, got "),
        ({"max_features": 3.7}, TypeError, "max_features must be int, got "),
        ({"max_features": -3}, ValueError, "max_features must be >= 1, got"),
        ({"max_features": 0}, ValueError, "max_features must be >= 1, got"),

        # stop_threshold
        ({"stop_threshold": "0.5"}, TypeError, "stop_threshold must be int or float, got"),
        ({"stop_threshold": [0.5]}, TypeError, "stop_threshold must be int or float, got"),
        ({"stop_threshold": -0.1}, ValueError, "stop_threshold must be >= 0, got"),

        # min_samples_leaf
        ({"min_samples_leaf": "0.5"}, TypeError, "min_samples_leaf must be int or float, got"),
        ({"min_samples_leaf": [0.5]}, TypeError, "min_samples_leaf must be int or float, got"),
        ({"min_samples_leaf": -0.1}, ValueError, "min_samples_leaf must be > 0, got"),
        ({"min_samples_leaf": 0}, ValueError, "min_samples_leaf must be > 0, got"),

        # learning_rate
        ({"learning_rate": "0.5"}, TypeError, "learning_rate must be int or float, got"),
        ({"learning_rate": [1]}, TypeError, "learning_rate must be int or float, got"),
        ({"learning_rate": -0.1}, ValueError, "learning_rate must be > 0, got"),
        ({"learning_rate": 0}, ValueError, "learning_rate must be > 0, got"),

        # eps
        ({"eps": "0.5"}, TypeError, "eps must be int or float, got"),
        ({"eps": [9]}, TypeError, "eps must be int or float, got"),
        ({"eps": -0.1}, ValueError, "eps must be > 0, got"),
        ({"eps": 0}, ValueError, "eps must be > 0, got"),
    ]
)
def test_validate_params_errors(params, expected_exception, expected_msg):
    with pytest.raises(expected_exception) as excinfo:
        validate_params(params)
    assert expected_msg in str(excinfo.value)


# ----- split_complete_incomplete ---------------------------------------

def test_split_complete_incomplete_basic():
    X = pd.DataFrame({
        "a": [1, 2, np.nan, 4],
        "b": [5, 6, 7, np.nan],
    })

    complete, incomplete = split_complete_incomplete(X)

    assert not complete.isna().any().any()
    assert incomplete.isna().any(axis=1).any()
    pd.testing.assert_frame_equal(pd.concat([complete, incomplete]).sort_index(), X.sort_index())


def test_split_complete_incomplete_all_complete():
    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    complete, incomplete = split_complete_incomplete(X)
    assert len(incomplete) == 0
    pd.testing.assert_frame_equal(complete, X)


def test_split_complete_incomplete_all_missing():
    X = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
    complete, incomplete = split_complete_incomplete(X)
    assert len(complete) == 0
    assert len(incomplete) == len(X)


# ----- euclidean_distance -----------------------------------------------

vector_pairs = [
    (np.array([1.0, 2.0, 3.0]), np.array([4.0, 6.0, 8.0])),
    (np.array([1.0, np.nan, 3.0, 4.0]), np.array([1.0, 2.0, 3.0, 5.0])),
    (np.array([np.nan, 2.0, 3.0, np.nan, 5.0]), np.array([1.0, 2.0, 4.0, 8.0, 5.0])),
]


@pytest.mark.parametrize("a, b", vector_pairs)
def test_euclidean_distance_matches_scipy(a, b):
    mask = ~np.isnan(a) & ~np.isnan(b)
    expected = euclidean(a[mask], b[mask])
    result = euclidean_distance(a, b)
    assert np.isclose(result, expected, atol=1e-12), f"Mismatch for vectors {a}, {b}"


# ----- fuzzy_c_means -----------------------------------------------

dataframes_list = [
    pd.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "b": [5.0, 4.0, 3.0],
    }),

    pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [5.0, 4.0, 3.0, 2.0, 1.0],
    }),

    pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 17.0, 5.0, 6.0, 7.0],
        "b": [5.0, 4.0, 3.0, 20.0, 2.0, 1.0, 0.0],
        "c": [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0],
        "d": [10.0, 12.5, 15.0, 16.0, 17.0, 18.0, 19.0],
        "e": [9.0, 8.7, 7.0, 6.0, 5.0, 2.2, 2.1],
    }),
]


@pytest.mark.parametrize("X", dataframes_list)
def test_fuzzy_c_means_output_shapes(X):
    centers, memberships = fuzzy_c_means(X, n_clusters=2, m=2.0, max_iter=50, tol=1e-4, random_state=42)

    assert centers.shape == (2, X.shape[1])
    assert memberships.shape == (X.shape[0], 2)
    np.testing.assert_allclose(np.sum(memberships, axis=1), 1.0, atol=1e-5)


def test_fuzzy_c_means_converges_reasonably():
    X = np.vstack([
        np.random.normal(0, 0.1, (5, 2)),
        np.random.normal(5, 0.1, (5, 2))
    ])
    centers, memberships = fuzzy_c_means(X, n_clusters=2, random_state=0)
    assert np.linalg.norm(centers[0] - centers[1]) > 1.0


def test_fuzzy_c_means_same_random_state_reproducible():
    X = np.array([[1, 2], [3, 4], [5, 6], [8, 9]])
    n_clusters = 3

    centers_1, memberships_1 = fuzzy_c_means(
        X, n_clusters=n_clusters, m=2.0, max_iter=50, tol=1e-4, random_state=42
    )
    centers_2, memberships_2 = fuzzy_c_means(
        X, n_clusters=n_clusters, m=2.0, max_iter=50, tol=1e-4, random_state=42
    )

    np.testing.assert_allclose(centers_1, centers_2, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(memberships_1, memberships_2, rtol=1e-8, atol=1e-8)


# ----- check_input_dataset ---------------------------------------

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


@pytest.mark.parametrize("X", [
    ([[1, 2], [1, 5, 7]]),
])
def test_check_input_dataset_wrong_input_types(X):
    with pytest.raises(TypeError,
                       match="X must be array-like or DataFrame, got"):
        check_input_dataset(X)


@pytest.mark.parametrize("X", [
    (5),
    ("txt"),
    ([5]),
    ([1, 2, 3]),
    ([[[1, 2], [3, 4]], [[5, 7], [3, 6]]]),
    ([{'a': 1}, {'a': 2}]),
    (object()),
    (5 + 3j),
    (lambda x: x)
])
def test_check_input_dataset_wrong_input_dimensions(X):
    with pytest.raises(ValueError,
                       match="X must be a 2D array-like structure, got"):
        check_input_dataset(X)


@pytest.mark.parametrize("X", [
    [[]], pd.DataFrame()
])
def test_check_input_dataset_empty_dataset(X):
    with pytest.raises(ValueError,
                       match="X must contain at least one sample, got an empty dataset instead"):
        check_input_dataset(X)


@pytest.mark.parametrize("X, require_numeric", [
    ([[1, "txt", 3], [3, 7, 4]], True),
    (pd.DataFrame({
        'height_cm': [165, 170, 175, 180, 175, 160, 175, 190],
        'weight_kg': [60, 65, 70, 75, 80, 55, 68, 80],
        'bmi': [22.0, 22.5, 24.2, 26.5, 26.1, 21.5, 23.8, 25.0],
        'name': ["txt", "txt", "txt", "txt", "txt", "txt", "txt", "txt"]
    }), True),
])
def test_check_input_dataset_check_require_numeric_true(X, require_numeric):
    with pytest.raises(TypeError,
                       match="X must be numeric, got non-numeric columns:"):
        check_input_dataset(X, require_numeric=require_numeric)


@pytest.mark.parametrize("X,allow_nan", [
    ([[1, np.nan, 3], [3, 7, 4]], False),
    (pd.DataFrame({
        'height_cm': [165, 170, np.nan, 180, 175, 160, np.nan, 190],
        'weight_kg': [60, 65, 70, np.nan, 80, 55, 68, np.nan],
        'bmi': [22.0, 22.5, 24.2, 26.5, 26.1, 21.5, 23.8, np.nan]
    }), False),
])
def test_check_input_dataset_check_allow_nan_false(X, allow_nan):
    with pytest.raises(ValueError, match="X must not contain missing values"):
        check_input_dataset(X, allow_nan=allow_nan)


@pytest.mark.parametrize("X, require_complete_rows", [
    ([[1, np.nan, 3], [3, 7, np.nan]], True),
    (pd.DataFrame({
        'height_cm': [165, 170, np.nan, 180],
        'weight_kg': [np.nan, 65, 70, np.nan],
        'bmi': [22.0, np.nan, 24.2, 26.5]
    }), True),
])
def test_check_input_dataset_check_require_complete(X, require_complete_rows):
    with pytest.raises(ValueError,
                       match="X must contain at least one row with no missing values"):
        check_input_dataset(X, require_complete_rows=require_complete_rows)


@pytest.mark.parametrize("X, no_nan_rows", [
    ([[np.nan, np.nan, np.nan], [3, 7, 4]], True),
    (pd.DataFrame({
        'height_cm': [165, 170, np.nan, 180, 175, 160, np.nan, np.nan],
        'weight_kg': [60, 65, 70, np.nan, 80, 55, 68, np.nan],
        'bmi': [22.0, 22.5, 24.2, 26.5, 26.1, 21.5, 23.8, np.nan]
    }), True),
])
def test_check_input_dataset_check_no_nan_rows(X, no_nan_rows):
    with pytest.raises(ValueError,
                       match="X must not contain rows with all NaNs"):
        check_input_dataset(X, no_nan_rows=no_nan_rows)


@pytest.mark.parametrize("X, no_nan_columns", [
    ([[np.nan, 4, 2], [np.nan, 7, 4]], True),
    (pd.DataFrame({
        'height_cm': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        'weight_kg': [60, 65, 70, np.nan, 80, 55, 68, 70],
        'bmi': [22.0, 22.5, 24.2, 26.5, 26.1, 21.5, 23.8, np.nan]
    }), True),
])
def test_check_input_dataset_check_no_nan_columns(X, no_nan_columns):
    with pytest.raises(ValueError,
                       match="X must not contain columns with all NaNs, got"):
        check_input_dataset(X, no_nan_columns=no_nan_columns)


# ----- fcm_predict ---------------------------------------

@pytest.mark.parametrize("X_new, centers, m", [
    (
            np.array([[1.0, 2.0], [2.0, 1.0]]),
            np.array([[1.0, 1.0], [2.0, 2.0]]),
            2.0
    ),
    (
            np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
            np.array([[0.0, 0.0], [3.0, 3.0]]),
            1.5
    ),
    (
            np.random.rand(5, 3),
            np.random.rand(3, 3),
            2.5
    ),
])
def test_fcm_predict(X_new, centers, m):
    u = fcm_predict(X_new, centers, m)

    n_samples = X_new.shape[0]
    n_clusters = centers.shape[0]

    assert u.shape == (n_samples, n_clusters)

    assert np.all(u >= 0) and np.all(u <= 1)

    row_sums = np.sum(u, axis=1)
    np.testing.assert_allclose(row_sums, np.ones(n_samples), atol=1e-5)


# ----- compute_fcm_objective ---------------------------------------

@pytest.mark.parametrize("X, centers, u, m, expected", [
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
def test_compute_fcm_objective(X, centers, u, m, expected):
    result = compute_fcm_objective(X, centers, u, m)
    assert isinstance(result, float)
    assert result == pytest.approx(expected)


# ----- find_optimal_clusters_fuzzy ---------------------------------------

@pytest.mark.parametrize("X, min_clusters, max_clusters, random_state, m, expected_k", [
    (
            pd.DataFrame(make_blobs(n_samples=150, centers=3, cluster_std=0.5, random_state=42)[0]),
            2, 6, 42, 2.0, 3),
    (
            pd.DataFrame(make_blobs(n_samples=200, centers=4, cluster_std=0.4, random_state=1)[0]),
            2, 7, 1, 2.0, 4),
    (
            pd.DataFrame(make_blobs(n_samples=100, centers=2, cluster_std=0.6, random_state=10)[0]),
            1, 5, 10, 2.0, 2),
    (
            pd.DataFrame(make_blobs(n_samples=150, centers=8, cluster_std=0.6, random_state=42)[0]),
            5, 11, 42, 1.1, 8),
    (
            pd.DataFrame(make_blobs(n_samples=150, centers=6, cluster_std=0.6, random_state=42)[0]),
            3, 9, 42, 1.1, 6),
])
def test_find_optimal_clusters_fuzzy(X, min_clusters, max_clusters, random_state, m, expected_k):
    result = find_optimal_clusters_fuzzy(X, min_clusters, max_clusters, random_state, m)
    assert isinstance(result, int)
    assert abs(result - expected_k) <= 3
