import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from ficaria.utils import check_input_dataset, compute_fcm_objective, find_optimal_clusters_fuzzy, fcm_predict


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


@pytest.mark.parametrize("X, require_numeric, allow_nan", [
    ([[1, 2], [1, 5, 7]], False, True),
])
def test_check_input_dataset_wrong_input_types(X, require_numeric, allow_nan):
    with pytest.raises(TypeError,
                       match="Invalid input: Expected a 2D structure such as a DataFrame, NumPy array, or similar tabular format"):
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
                       match="Invalid input: Expected a 2D structure"):
        check_input_dataset(X, require_numeric=require_numeric, allow_nan=allow_nan)


@pytest.mark.parametrize("X", [
    [[]], pd.DataFrame()
])
def test_check_input_dataset_empty_dataset(X):
    with pytest.raises(ValueError,
                       match="Invalid input: Input dataset is empty"):
        check_input_dataset(X)


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
                       match="Invalid input: Input dataset contains not numeric values"):
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
                       match="Invalid input: Input dataset contains missing values"):
        check_input_dataset(X, require_numeric=require_numeric, allow_nan=allow_nan)


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
                       match="Invalid input: Input dataset contains no complete rows"):
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
                       match="Invalid input: Input dataset contains a row with only NaN values"):
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
                       match="Invalid input: Input dataset contains a column with only NaN values"):
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
