import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from sklearn.impute import SimpleImputer

from ficaria.utils import get_neighbors, find_best_k, check_input_dataset, impute_KI, compute_fcm_objective, \
    find_optimal_clusters_fuzzy, impute_FCKI, fuzzy_c_means, fcm_predict, fuzzy_c_means_categorical


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


@pytest.mark.parametrize("St, random_col, original_value, max_iter", [
    (
            pd.DataFrame({
                'height_cm': [165, 170, np.nan, 180, 175, 160, np.nan, 190],
                'weight_kg': [60, 65, 70, np.nan, 80, 55, 68, np.nan],
                'bmi': [22.0, 22.5, 24.2, 26.5, 26.1, 21.5, 23.8, np.nan]}),
            1, 85, 20),
    (
            pd.DataFrame({
                'age': [25, 26, np.nan, 51, 53, 72, 75],
                'income': [50000, 55000, 80000, 85000, 90000, 120000, np.nan]}),
            0, 125000, 15)
])
def test_find_best_k(St, random_col, original_value, max_iter):
    result1 = find_best_k(St, random_col, original_value, max_iter)
    result2 = find_best_k(St, random_col, original_value, max_iter)
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
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(X)
    assert result.isna().sum().sum() == 0


@pytest.mark.parametrize("X", [
    (pd.DataFrame({
        'height_cm': [165, 170, 175, 180, 175, 160, 175, 190],
        'weight_kg': [60, 65, 70, 75, 80, 55, 68, 80],
        'bmi': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]})),
])
def test_impute_KI_error_no_complete(X):
    with pytest.raises(ValueError,
                       match="Invalid input: No rows with valid values found in columns:"):
        impute_KI(X)


@pytest.mark.parametrize("X, X_train, random_state, max_iter", [
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
            42, 15
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
            123, 30
    ),
    (
            pd.DataFrame({
                'a': [1, np.nan, 3],
                'b': [4, 5, np.nan],
                'c': [7, 8, 9]
            }),
            None,
            42, 20
    )
])
def test_impute_KI_with_parameters(X, X_train, random_state, max_iter):
    np_rng_1 = np.random.RandomState(random_state)
    result = impute_KI(X, X_train=X_train, np_rng=np_rng_1, max_iter=max_iter)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == X.shape
    assert result.isna().sum().sum() == 0
    np_rng_2 = np.random.RandomState(random_state)
    result_repeat = impute_KI(X, X_train=X_train, np_rng=np_rng_2, max_iter=max_iter)
    np.testing.assert_array_almost_equal(result, result_repeat)


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
    assert isinstance(result, int)
    assert abs(result - expected_k) <= 3


@pytest.mark.parametrize("X, X_train, n_clusters, random_state, m, max_iter", [
    (
            pd.DataFrame({
                "a": [1.0, np.nan, 3.0],
                "b": [4.0, 5.0, 6.0]
            }),
            pd.DataFrame({
                "a": [1.0, 2.0, 3.0],
                "b": [4.0, 5.0, 6.0]
            }),
            2, 42, 1.1, 15
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
            2, 42, 2, 20
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
            2, 42, 1.7, 5)
])
def test_impute_FCKI(X, X_train, n_clusters, random_state, m, max_iter):
    np_rng = np.random.RandomState(random_state)
    imputer = SimpleImputer(strategy="mean")
    X_train_filled = imputer.fit_transform(X_train)
    X_train_filled_df = pd.DataFrame(X_train_filled, columns=X_train.columns)
    centers, u = fuzzy_c_means(
        X_train_filled_df.values,
        n_clusters=n_clusters,
        m=m,
        random_state=random_state,
    )
    result = impute_FCKI(X, X_train, centers, u, n_clusters, imputer, m, np_rng, random_state, max_iter)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == X.shape
    assert result.isna().sum().sum() == 0


############################################

@pytest.mark.parametrize("X, n_clusters, random_state, m", [
    (pd.DataFrame({
        "x1": [1.0, 2.0, 3.0],
        "x2": [4.0, 5.0, 6.0]
    }), 2, 42, 2),

    (pd.DataFrame({
        "cat1": ["A", "B", "A", "C", "A", "B", "C"],
        "cat2": ["X", "X", "Y", "Y", "X", "Y", "X"]
    }), 3, 123, 1.1),

    (pd.DataFrame({
        "num1": [1, 2, 3, 4],
        "cat1": ["A", "B", "A", "B"]
    }), 2, 25, 1.5),

    (pd.DataFrame({
        "x1": [2.0, 2.0, 2.0],
        "x2": [5.0, 5.0, 5.0]
    }), 2, 42, 2),
])
def test_fuzzy_c_means_categorical(X, n_clusters, random_state, m):
    centers, u = fuzzy_c_means_categorical(X, n_clusters=n_clusters, m=m, max_iter=50, random_state=random_state)

    assert u.shape == (X.shape[0], n_clusters), "Membership matrix shape is incorrect"
    assert centers.shape[0] == n_clusters, "Number of centers is incorrect"
    assert set(centers.columns) == set(X.columns), "Centers columns mismatch"

    row_sums = u.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(X.shape[0]), rtol=1e-5, atol=1e-8)

    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            assert all(val in X[col].values for val in centers[col]), f"Categorical center {col} contains invalid value"

    centers2, u2 = fuzzy_c_means_categorical(X, n_clusters=n_clusters, m=m, max_iter=50, random_state=random_state)

    pd.testing.assert_frame_equal(centers.sort_index(axis=1), centers2.sort_index(axis=1), check_dtype=False, atol=1e-8,
                                  rtol=1e-5, obj="Cluster centers")

    np.testing.assert_allclose(u, u2, rtol=1e-5, atol=1e-8,
                               err_msg="Membership matrix differs between runs with the same random_state")
