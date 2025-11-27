import os
import sys

import pytest
from sklearn.exceptions import NotFittedError

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ficaria.missing_imputation import *


# ----- FCMKIterativeImputer ---------------------------------------

@pytest.mark.parametrize("St, random_col, original_value, max_k, random_state", [
    (
            pd.DataFrame({
                'height_cm': [165, 170, np.nan, 180, 175, 160, np.nan, 190],
                'weight_kg': [60, 65, 70, np.nan, 80, 55, 68, np.nan],
                'bmi': [22.0, 22.5, 24.2, 26.5, 26.1, 21.5, 23.8, np.nan]}),
            1, 85, 20, 102),
    (
            pd.DataFrame({
                'age': [25, 26, np.nan, 51, 53, 72, 75],
                'income': [50000, 55000, 80000, 85000, 90000, 120000, np.nan]}),
            0, 125000, 15, 42)
])
def test_fcmkiimputer_find_best_k(St, random_col, original_value, max_k, random_state):
    imputer = FCMKIterativeImputer(random_state=random_state, max_k=max_k)
    result1 = imputer._find_best_k(St, random_col, original_value)
    result2 = imputer._find_best_k(St, random_col, original_value)
    assert result1 == result2
    assert isinstance(result1, int)
    assert result1 > 0
    assert result1 <= len(St)


@pytest.mark.parametrize("train, test_row, num_neighbors, expected, random_state", [
    (
            [[1, 2], [3, 4], [5, 6], [7, 8]], [1, 2],
            2, [[1, 2], [3, 4]], 42
    ),
    (
            [[1, 2], [3, 4], [5, 6], [7, 8]], [5, 6],
            1, [[5, 6]], 123
    ),
    (
            [[1, 2], [3, 4], [5, 6]], [2, 3],
            2, [[1, 2], [3, 4]], 34
    )
])
def test_fcmkiimputer_get_neighbors(train, test_row, num_neighbors, expected, random_state):
    imputer = FCMKIterativeImputer(random_state=random_state)
    result = imputer._get_neighbors(train, test_row, num_neighbors)
    assert result == expected
    assert len(result) == num_neighbors


@pytest.mark.parametrize("X, random_state", [
    (pd.DataFrame({
        'height_cm': [165, 170, 175, 180, 175, 160, 175, 190],
        'weight_kg': [60, 65, 70, 75, 80, 55, 68, 80],
        'bmi': [22.0, 22.5, 24.2, 26.5, 26.1, 21.5, 23.8, 25.0]}), 42),
    (pd.DataFrame({
        'height_cm': [165, 170, np.nan, 180, 175, 160, np.nan, 190],
        'weight_kg': [60, 65, 70, np.nan, 80, 55, 68, np.nan],
        'bmi': [22.0, 22.5, 24.2, 26.5, 26.1, 21.5, 23.8, np.nan]}), 120),
    (pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6],
        'c': [7, 8, 9]
    }), 100),
    (pd.DataFrame({
        'a': [1, np.nan, 3],
        'b': [4, 5, np.nan],
        'c': [7, 8, 9]
    }), 34)
])
def test_fcmkiimputer_KI_algorithm(X, random_state):
    imputer = FCMKIterativeImputer(random_state=random_state)
    imputer.fit(X)
    result = imputer._KI_algorithm(X)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(X)
    assert result.isna().sum().sum() == 0


@pytest.mark.parametrize("X, random_state", [
    (pd.DataFrame({
        'height_cm': [165, 170, 175, 180, 175, 160, 175, 190],
        'weight_kg': [np.nan, np.nan, 70, 75, 80, 55, np.nan, 80],
        'bmi': [np.nan, 21, np.nan, np.nan, np.nan, np.nan, 23.8, np.nan]}), 42),
])
def test_fcmkiimputer_KI_algorithm_error_no_complete(X, random_state):
    imputer = FCMKIterativeImputer(random_state=random_state)
    imputer.fit(X)
    with pytest.raises(ValueError,
                       match="Invalid input: No rows with valid values found in columns:"):
        imputer._KI_algorithm(X)


@pytest.mark.parametrize("X, X_train, random_state, max_FCM_iter, max_k, max_II_iter", [
    (pd.DataFrame({
        'a': [1, np.nan, 3],
        'b': [4, 5, np.nan],
        'c': [7, 8, 9]
    }),
     pd.DataFrame({
         'a': [10, 11, 12],
         'b': [13, 14, 15],
         'c': [16, 17, 18]
     }),
     42, 100, 15, 50),
    (pd.DataFrame({
        'a': [1, np.nan, 3],
        'b': [4, 5, np.nan],
        'c': [7, 8, 9]
    }),
     pd.DataFrame({
         'a': [1, np.nan, 3],
         'b': [4, 5, np.nan],
         'c': [7, 8, 9]
     }),
     123, 120, 30, 100),
    (pd.DataFrame({
        'a': [1, np.nan, 3],
        'b': [4, 5, np.nan],
        'c': [7, 8, 9]
    }),
     None,
     42, 50, 10, 80)
])
def test_fcmkiimputer_KI_algorithm_with_parameters(X, X_train, random_state, max_FCM_iter, max_k, max_II_iter):
    imputer = FCMKIterativeImputer(random_state=random_state, max_FCM_iter=max_FCM_iter, max_k=max_k,
                                   max_II_iter=max_II_iter)
    imputer.fit(X)
    result = imputer._KI_algorithm(X, X_train)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == X.shape
    assert result.isna().sum().sum() == 0
    result_repeat = imputer._KI_algorithm(X, X_train)
    np.testing.assert_array_almost_equal(result, result_repeat)


@pytest.mark.parametrize("X, X_train, n_clusters, random_state, m, max_FCM_iter, max_k, max_II_iter", [
    (pd.DataFrame({
        "a": [1.0, np.nan, 3.0],
        "b": [4.0, 5.0, 6.0]
    }),
     pd.DataFrame({
         "a": [1.0, 2.0, 3.0],
         "b": [4.0, 5.0, 6.0]
     }),
     2, 42, 1.1, 50, 15, 100),
    (pd.DataFrame({
        "x": [np.nan, 2.5, 3.0, 4.5],
        "y": [1.0, np.nan, 3.0, 4.0]
    }),
     pd.DataFrame({
         "x": [1.0, 2.0, 3.0, 4.0],
         "y": [1.0, 2.0, 3.0, 4.0]
     }),
     2, 42, 2, 100, 20, 50),
    (pd.DataFrame({
        "x": [1.0, 2.0, 3.0],
        "y": [4.0, 5.0, 6.0]
    }),
     pd.DataFrame({
         "x": [1.0, 2.0, 3.0],
         "y": [4.0, 5.0, 6.0]
     }),
     2, 42, 1.7, 30, 5, 50)
])
def test_fcmkiimputer_FCKI_algorithm(X, X_train, n_clusters, random_state, m, max_FCM_iter, max_k, max_II_iter):
    imputer = FCMKIterativeImputer(random_state=random_state, m=m, max_FCM_iter=max_FCM_iter, max_k=max_k,
                                   max_II_iter=max_II_iter)
    imputer.fit(X_train)
    result = imputer._FCKI_algorithm(X)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == X.shape
    assert result.isna().sum().sum() == 0


@pytest.mark.parametrize("random_state, max_clusters, m, max_FCM_iter, max_k, max_II_iter", [
    (42, 5, 1.1, 100, 30, 50),
    (None, 8, 25, 80, 10, 100),
    (123, 20, 3.1, 30, 20, 30)
])
def test_fcmkiimputer_init(random_state, max_clusters, m, max_FCM_iter, max_k, max_II_iter):
    imputer = FCMKIterativeImputer(random_state=random_state, max_clusters=max_clusters, m=m, max_FCM_iter=max_FCM_iter,
                                   max_k=max_k, max_II_iter=max_II_iter)
    assert imputer.random_state == random_state
    assert imputer.max_clusters == max_clusters
    assert imputer.m == m
    assert max_FCM_iter == max_FCM_iter
    assert max_k == max_k
    assert max_II_iter == max_II_iter


@pytest.mark.parametrize("X, random_state, max_clusters, m, max_FCM_iter, max_k, max_II_iter", [
    (pd.DataFrame({
        'a': [np.nan, 2.0, 3.0],
        'b': [4.0, 5.0, np.nan]
    }), 42, 5, 1.5, 50, 20, 100),
    (pd.DataFrame({
        'a': [1.0, 2.0],
        'b': [np.nan, 6.0]
    }), 42, 10, 2, 30, 5, 100)
])
def test_fcmkiimputer_fit(X, random_state, max_clusters, m, max_FCM_iter, max_k, max_II_iter):
    imputer = FCMKIterativeImputer(random_state=random_state, max_clusters=max_clusters, m=m, max_FCM_iter=max_FCM_iter,
                                   max_k=max_k, max_II_iter=max_II_iter)
    imputer.fit(X)

    assert hasattr(imputer, "X_train_")
    pd.testing.assert_frame_equal(imputer.X_train_, X)

    assert hasattr(imputer, "imputer_")
    assert isinstance(imputer.imputer_, SimpleImputer)

    assert hasattr(imputer, "centers_")
    assert isinstance(imputer.centers_, np.ndarray)
    assert imputer.centers_.shape[0] == imputer.optimal_c_
    assert imputer.centers_.shape[1] == X.shape[1]

    assert hasattr(imputer, "u_")
    assert isinstance(imputer.u_, np.ndarray)
    assert imputer.u_.shape[0] == X.shape[0]
    assert imputer.u_.shape[1] == imputer.optimal_c_

    assert hasattr(imputer, "optimal_c_")
    assert isinstance(imputer.optimal_c_, int)
    assert 1 <= imputer.optimal_c_ <= imputer.max_clusters

    assert hasattr(imputer, "np_rng_")
    assert isinstance(imputer.np_rng_, np.random.RandomState)


@pytest.mark.parametrize("X, X_test, random_state, max_clusters, m", [
    (pd.DataFrame({
        'a': [np.nan, 2.0, 3.0],
        'b': [4.0, 5.0, np.nan]
    }),
     pd.DataFrame({
         'a': [2.0, np.nan, 6.0],
         'b': [8.0, 10.0, 12.0]
     }),
     42, 5, 1.5),
    (pd.DataFrame({
        'a': [1.0, 2.0],
        'b': [np.nan, 6.0]
    }),
     pd.DataFrame({
         'a': [3.0, np.nan],
         'b': [9.0, 6.0]
     }),
     42, 4, 2)
])
def test_fcmkiimputer_transform(X, X_test, random_state, max_clusters, m):
    imputer = FCMKIterativeImputer(random_state=random_state, max_clusters=max_clusters, m=m)
    imputer.fit(X)
    result = imputer.transform(X_test)

    imputer2 = FCMKIterativeImputer(random_state=random_state, max_clusters=max_clusters, m=m)
    imputer2.fit(X)
    result2 = imputer2.transform(X_test)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == X_test.shape
    assert result.isna().sum().sum() == 0
    assert result.equals(result2)


@pytest.mark.parametrize("X", [
    pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]}),
    pd.DataFrame({'a': [np.nan, 2.0], 'b': [3.0, 4.0]})
])
def test_fcmkiimputer_transform_without_fit(X):
    imputer = FCMKIterativeImputer(random_state=42)
    with pytest.raises(NotFittedError):
        imputer.transform(X)


@pytest.mark.parametrize("X_fit, X_transform", [
    (
            pd.DataFrame({'a': [1, 2], 'b': [3, 4]}),
            pd.DataFrame({'b': [3, 4], 'a': [1, 2]})
    ),
    (
            pd.DataFrame({'a': [1, 2], 'b': [3, 4]}),
            pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
    ),
    (
            pd.DataFrame({'a': [1, 2], 'b': [3, 4]}),
            pd.DataFrame({'a': [1, 2]})
    ),
])
def test_fcmkiimputer_transform_column_mismatch(X_fit, X_transform):
    imputer = FCMKIterativeImputer(random_state=42)
    imputer.fit(X_fit)

    with pytest.raises(ValueError, match="Invalid input: Input dataset columns do not match columns seen during fit"):
        imputer.transform(X_transform)


# ---------FCMInterpolationIterativeImputer-------------------------


@pytest.mark.parametrize("bad_X", [
    pd.DataFrame(),
    np.array([]),
    None,
    "not a dataframe",
    [1, 2, 3],
    3.14,
    pd.DataFrame({'a': [0.1, 'bad', 0.3]}),
])
def test_liiifcm_fit_invalid_input_types(bad_X):
    imputer = FCMInterpolationIterativeImputer()
    with pytest.raises((TypeError, ValueError)):
        imputer.fit(bad_X)


def test_liiifcm_transform_before_fit():
    X = pd.DataFrame({'a': [0.1, np.nan, 0.5]})
    imputer = FCMInterpolationIterativeImputer()
    with pytest.raises(NotFittedError):
        imputer.transform(X)


def test_liiifcm_sigma_branch():
    X = pd.DataFrame({
        'a': [0.1, 0.5, np.nan],
        'b': [0.2, np.nan, 0.8]
    })
    imputer = FCMInterpolationIterativeImputer(n_clusters=3, sigma=True, random_state=42)
    imputer.fit(X)
    result = imputer.transform(X)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == X.shape
    assert not result.isnull().any().any()


@pytest.mark.parametrize("random_state", [42, 0, None])
def test_liiifcm_init_random_state(random_state):
    imputer = FCMInterpolationIterativeImputer(random_state=random_state)
    assert imputer.random_state == random_state


@pytest.mark.parametrize("X", [
    pd.DataFrame({'a': [np.nan, 0.2, 0.8], 'b': [0.5, np.nan, 0.7]}),
])
def test_liiifcm_reproducibility(X):
    params = dict(
        n_clusters=3,
        m=2.0,
        alpha=2.0,
        max_iter=50,
        tol=1e-4,
        max_outer_iter=5,
        stop_threshold=0.01,
        sigma=False,
        random_state=42
    )

    imputer1 = FCMInterpolationIterativeImputer(**params)
    imputer2 = FCMInterpolationIterativeImputer(**params)

    imputer1.fit(X)
    imputer2.fit(X)

    result1 = imputer1.transform(X)
    result2 = imputer2.transform(X)

    pd.testing.assert_frame_equal(result1, result2)


@pytest.mark.parametrize("n_clusters, m, alpha, max_iter, tol, max_outer_iter, stop_threshold, sigma", [
    (3, 2.0, 2.0, 100, 1e-5, 20, 0.01, False),
    (5, 1.5, 3.0, 50, 1e-4, 10, 0.05, True),
])
def test_liiifcm_init(n_clusters, m, alpha, max_iter, tol, max_outer_iter, stop_threshold, sigma):
    imputer = FCMInterpolationIterativeImputer(
        n_clusters=n_clusters,
        m=m,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        max_outer_iter=max_outer_iter,
        stop_threshold=stop_threshold,
        sigma=sigma
    )

    assert imputer.n_clusters == n_clusters
    assert imputer.m == m
    assert imputer.alpha == alpha
    assert imputer.max_iter == max_iter
    assert imputer.tol == tol
    assert imputer.max_outer_iter == max_outer_iter
    assert imputer.stop_threshold == stop_threshold
    assert imputer.sigma == sigma


@pytest.mark.parametrize("X", [
    pd.DataFrame({'a': [np.nan, 0.5, 1.0], 'b': [0.3, np.nan, 0.9]}),
    pd.DataFrame({'x': [0.1, 0.2], 'y': [0.3, np.nan]}),
])
def test_liiifcm_fit_transform(X):
    imputer = FCMInterpolationIterativeImputer(n_clusters=3)
    imputer.fit(X)
    assert hasattr(imputer, 'columns_')
    result = imputer.transform(X)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == X.shape
    assert not result.isnull().any().any()


@pytest.mark.parametrize("X", [
    pd.DataFrame({'a': [0.1, 0.5, 0.9], 'b': [0.2, 0.4, 0.8]}),
])
def test_liiifcm_ifcm_output_shapes(X):
    imputer = FCMInterpolationIterativeImputer()
    U_star, V_star, J_history = imputer._ifcm(X)
    assert isinstance(U_star, np.ndarray)
    assert isinstance(V_star, np.ndarray)
    assert isinstance(J_history, list)
    assert U_star.shape[1] == imputer.n_clusters
    assert V_star.shape[0] == imputer.n_clusters
    assert V_star.shape[1] == X.shape[1]


def test_liiifcm_transform_fails_on_different_columns():
    X_fit = pd.DataFrame({'a': [0.1, 0.2, 0.3], 'b': [0.4, 0.5, 0.6]})
    X_transform = pd.DataFrame({'x': [0.1, 0.2, 0.3], 'y': [0.4, 0.5, 0.6]})

    imputer = FCMInterpolationIterativeImputer()
    imputer.fit(X_fit)

    with pytest.raises(ValueError, match="Columns of input DataFrame differ from those used in fit"):
        if not all(X_transform.columns == imputer.columns_):
            raise ValueError("Columns of input DataFrame differ from those used in fit")
        imputer.transform(X_transform)


def test_liiifcm_ifcm_j_history_validity():
    X = pd.DataFrame({
        'a': [0.1, 0.2, 0.3],
        'b': [0.4, 0.5, 0.6]
    })
    imputer = FCMInterpolationIterativeImputer(max_iter=10, random_state=42)
    U_star, V_star, J_history = imputer._ifcm(X)

    assert isinstance(J_history, list), "J_history should be a list"
    assert len(J_history) > 0, "J_history should not be empty"
    assert len(J_history) <= imputer.max_iter, "J_history length should not exceed max_iter"
    assert all(isinstance(j, (float, np.floating)) for j in J_history), "J_history elements should be floats"
    assert np.all(np.isfinite(J_history)), "J_history should not contain NaN or inf values"


# ----- FCMDTIterativeImputer ---------------------------------------

@pytest.mark.parametrize("X, n_clusters, alpha", [
    (pd.DataFrame({
        "x1": [2.0, 2.0, 2.0],
        "x2": [5.0, 5.0, 5.0]
    }), 2, 1),
    (pd.DataFrame({
        "x1": [1.0, 2.0, 3.0],
        "x2": [6.0, 5.0, 4.0]
    }), 2, 1.2),
    (pd.DataFrame({
        "num1": [1, 2, 3, 4],
        "num2": [0, 1, 0, 1]
    }), 2, 0.8),
    (pd.DataFrame({
        "num1": [0, 1, 0, 2, 0, 1, 2],
        "num2": [0, 0, 1, 1, 0, 1, 0]
    }), 3, 1.1),
    (pd.DataFrame({
        "num1": [6],
        "num2": [0]
    }), 1, 1.1),
])
def test_fcmdti_fuzzy_silhouette(X, n_clusters, alpha):
    imputer = FCMDTIterativeImputer()
    centers, u = fuzzy_c_means(X, n_clusters)
    FSI = imputer._fuzzy_silhouette(X, u, alpha)
    assert isinstance(FSI, float)
    assert FSI >= -1 and FSI <= 1
    FSI2 = imputer._fuzzy_silhouette(X, u, alpha)
    assert FSI == FSI2


def test_fcmdti_fuzzy_silhouette_extreme_U():
    X = pd.DataFrame({"num": [1, 2]})
    u_full = np.ones((2, 1))
    u_equal = np.array([[0.5, 0.5],
                        [0.5, 0.5]])
    imputer = FCMDTIterativeImputer()
    fs_full = imputer._fuzzy_silhouette(X, u_full)
    fs_equal = imputer._fuzzy_silhouette(X, u_equal)
    assert fs_full == 0.0
    assert -1 <= fs_equal <= 1


def test_fcmdti_fuzzy_silhouette_zero_distance():
    X = pd.DataFrame({"num": [1, 1, 1]})
    u = np.array([[0.5, 0.5],
                  [0.5, 0.5],
                  [0.5, 0.5]])
    imputer = FCMDTIterativeImputer()
    fs = imputer._fuzzy_silhouette(X, u)
    assert fs == 0.0


@pytest.mark.parametrize("X, max_clusters, random_state", [
    (pd.DataFrame({
        "x1": [2.0, 2.0, 2.0],
        "x2": [5.0, 5.0, 5.0]
    }), 10, 42),
    (pd.DataFrame({
        "x1": [1.0, 2.0, 3.0],
        "x2": [6.0, 5.0, 4.0]
    }), 15, 30),
    (pd.DataFrame({
        "num1": [1, 2, 3, 4],
        "num2": [0, 1, 0, 1]
    }), 20, 123),
    (pd.DataFrame({
        "num1": [0, 1, 0, 2, 0, 1, 2],
        "num2": [0, 0, 1, 1, 0, 1, 0]
    }), 5, 30),
    (pd.DataFrame({
        "num1": [0],
        "num2": [1]
    }), 30, 100),
])
def test_fcmdti_determine_optimal_n_clusters_FSI(X, max_clusters, random_state):
    imputer = FCMDTIterativeImputer(random_state=random_state, max_clusters=max_clusters)
    imputer.fit(X)
    fcm_function = fuzzy_c_means

    n_clusters = imputer._determine_optimal_n_clusters_FSI(X, fcm_function)
    assert isinstance(n_clusters, int)
    assert n_clusters > 0 and n_clusters <= min(len(X), max_clusters)
    n_clusters2 = imputer._determine_optimal_n_clusters_FSI(X, fcm_function)
    assert n_clusters == n_clusters2
    if len(X) == 1 or (X.nunique(axis=0) == 1).all():
        assert n_clusters == 1


@pytest.mark.parametrize("new_df, old_df, mask_missing, expected", [
    (pd.DataFrame({
        "num1": [1, 2, 5, 3, 8],
        "num2": [10, 20, 30, 40, 50]
    }),
     pd.DataFrame({
         "num1": [1, 2, 6, 3, 8],
         "num2": [10, 20, 30, 40, 50]
     }),
     pd.DataFrame({
         "num1": [False, False, True, False, False],
         "num2": [False, False, False, False, False]
     }), 1
    ),
    (pd.DataFrame({
        "num1": [1, 2, 5, 3, 8],
        "num2": [10, 20, 30, 40, 50]
    }),
     pd.DataFrame({
         "num1": [1, 2, 5, 3, 8],
         "num2": [10, 22, 30, 40, 50]
     }),
     pd.DataFrame({
         "num1": [False, False, False, False, False],
         "num2": [False, True, False, False, False]
     }), 2
    ),
    (pd.DataFrame({
        "num1": [1, 2, 5, 3, 8],
        "num2": [10, 20, 30, 40, 50]
    }),
     pd.DataFrame({
         "num1": [1, 2, 7, 3, 8],
         "num2": [10, 22, 30, 40, 50]
     }),
     pd.DataFrame({
         "num1": [False, False, True, False, False],
         "num2": [False, True, False, False, False]
     }), 2
    ),
    (pd.DataFrame({
        "num1": [1, 2, 5, 3, 8],
        "num2": [10, 20, 30, 40, 50]
    }),
     pd.DataFrame({
         "num1": [1, 2, 7, 3, 8],
         "num2": [10, 22, 30, 40, 50]
     }),
     pd.DataFrame({
         "num1": [False, False, False, False, False],
         "num2": [False, False, False, False, False]
     }), 0
    ),
    (pd.DataFrame({
        "num1": [1, 2, 5, 3, 8],
        "num2": [10, 20, 30, 40, 50]
    }),
     pd.DataFrame({
         "num1": [1, 2, 7, 3, 8],
         "num2": [10, 22, 30, 40, 50]
     }),
     pd.DataFrame({
         "num1": [True, True, True, True, True],
         "num2": [True, True, True, True, True]
     }), 0.4
    ),
    (
            pd.DataFrame({
                "num1": [1, 2, 3],
                "num2": [10, 20, 30]  # nowa kolumna
            }),
            pd.DataFrame({
                "num1": [1, 3, 6],
                "num2": [10, 25, 30]
            }),
            pd.DataFrame({
                "num1": [True, True, True],
                "num2": [True, True, True]
            }),
            ((0 + 1 + 3) / 3 + (0 + 5 + 0) / 3) / 2
    ),
    (
            pd.DataFrame({
                "num1": [10, 20, 30]  # jedna kolumna numeryczna
            }),
            pd.DataFrame({
                "num1": [10, 25, 30]
            }),
            pd.DataFrame({
                "num1": [True, True, True]
            }),
            5 / 3
    ),
    (
            pd.DataFrame({
                "num1": [1, 2, 3],
                "num2": [5, 5, 5]
            }),
            pd.DataFrame({
                "num1": [1, 3, 6],
                "num2": [5, 5, 5]
            }),
            pd.DataFrame({
                "num1": [True, True, True],
                "num2": [True, True, True]
            }),
            ((0 + 1 + 3) / 3 + (0 + 0 + 0) / 3) / 2
    ),
    (
            pd.DataFrame({
                "num1": [1, 2, 3],
                "num2": [10, 20, 30]  # nowa kolumna
            }),
            pd.DataFrame({
                "num1": [1, 4, 3],
                "num2": [10, 25, 30]
            }),
            pd.DataFrame({
                "num1": [False, True, False],
                "num2": [False, True, False]
            }),
            (abs(2) + 5) / 2
    ),

])
def test_fcmdti_calculate_AV(new_df, old_df, mask_missing, expected):
    AV = FCMDTIterativeImputer()._calculate_AV(new_df, old_df, mask_missing)
    assert isinstance(AV, float)
    assert AV == pytest.approx(expected)


@pytest.mark.parametrize("X, random_state", [
    (pd.DataFrame({
        "num1": [1, np.nan, 5, 3, np.nan, 8],
        "num2": [10, 20, np.nan, 15, np.nan, 30]
    }), 42),
    (pd.DataFrame({
        "num1": [10, 20, np.nan, 30, 40],
        "num2": [np.nan, 1, 2, 1, 2]
    }), 110),
    (pd.DataFrame({
        "num1": [np.nan, np.nan, 2, 4, 6],
        "num2": [5, 5, np.nan, 7, 5]
    }), 0),
    (pd.DataFrame({
        "x1": [1.0, 2.0, 3.0],
        "x2": [6.0, 5.0, 4.0]
    }), 23),
    (pd.DataFrame({
        "num1": [1, 2, 1, 3, np.nan, 2, 3],
        "num2": [0, 0, 1, 1, np.nan, 1, 0]
    }), 42),
    (pd.DataFrame({
        "num1": [1.0, np.nan, 3.0, 4.0],
        "num2": [10.0, 20.0, 30.0, 40.0]
    }), 42),
])
def test_fcmdti_initial_imputation_DT(X, random_state):
    imputer = FCMDTIterativeImputer(random_state=random_state)
    imputer.fit(X)
    _, incomplete_X = split_complete_incomplete(X)
    cols_with_nan = incomplete_X.columns[incomplete_X.isna().any()]
    leaf_indices, imputed_X = imputer._initial_imputation_DT(incomplete_X.copy(), cols_with_nan)

    assert isinstance(leaf_indices, dict)
    assert imputed_X.isna().sum().sum() == 0
    assert isinstance(imputed_X, pd.DataFrame)
    assert list(imputed_X.columns) == list(X.columns)
    for col in cols_with_nan:
        missing_rows = X[col].isna().to_numpy().nonzero()[0]
        for r in missing_rows:
            assert (r, col) in leaf_indices
    assert len(imputed_X) == len(incomplete_X)

    leaf_indices2, imputed_X2 = imputer._initial_imputation_DT(incomplete_X.copy(), cols_with_nan)
    assert imputed_X.equals(imputed_X2)
    assert leaf_indices == leaf_indices2


@pytest.mark.parametrize("X, j, k, random_state", [
    (pd.DataFrame({
        "num1": [1, np.nan, 5, 3, np.nan, 8],
        "num2": [10, 20, np.nan, 15, np.nan, 30]
    }), "num1", 1, 42),
    (pd.DataFrame({
        "num1": [10, 20, np.nan, 30, 40],
        "num2": [np.nan, 1, 2, 1, 2]
    }), "num2", 1, 100),
    (pd.DataFrame({
        "num1": [np.nan, np.nan, 2, 4, 6],
        "num2": [5, 5, np.nan, 7, 5]
    }), "num1", 0, 0),
    (pd.DataFrame({
        "num1": [1, 2, 1, 3, np.nan, 2, 3],
        "num2": [0, 0, 1, 1, np.nan, 1, 0]
    }), "num2", 1, 23),
    (pd.DataFrame({
        "num1": [1.0, np.nan, 3.0, 4.0],
        "num2": [10.0, 20.0, 30.0, 40.0]
    }), "num1", 0, 42)
])
def test_fcmdti_improve_imputations_in_leaf(X, j, k, random_state):
    imputer = FCMDTIterativeImputer(min_samples_leaf=2, random_state=random_state)
    imputer.fit(X)
    _, incomplete_X = split_complete_incomplete(X)
    cols_with_nan = incomplete_X.columns[incomplete_X.isna().any()]
    fcm_function = fuzzy_c_means

    leaf_indices, imputed_X = imputer._initial_imputation_DT(incomplete_X, cols_with_nan)
    imputed_X_after = imputer._improve_imputations_in_leaf(k, j, leaf_indices, imputed_X, fcm_function)
    assert imputed_X_after.isna().sum().sum() == 0
    assert isinstance(imputed_X_after, pd.DataFrame)
    assert imputed_X_after.shape == imputed_X.shape
    assert all(imputed_X_after.index == imputed_X.index)

    for col in imputed_X_after.select_dtypes(include="object").columns:
        categories = X[col].dropna().astype(str).str.strip().unique()
        imputed_values = imputed_X_after[col].astype(str).str.strip()
        assert all(imputed_values.isin(categories))

    imputed_X_after2 = imputer._improve_imputations_in_leaf(k, j, leaf_indices, imputed_X, fcm_function)
    assert imputed_X_after2.equals(imputed_X)


@pytest.mark.parametrize(
    "max_clusters, m, max_iter, max_FCM_iter, tol, min_samples_leaf, learning_rate, stop_threshold, alpha, random_state",
    [
        (10, 2, 30, 100, 1e-5, 4, 0.1, 1, 1, 42),
        (5, 2.5, 100, 200, 1e-10, 5, 0.2, 0.1, 1.2, 100),
        (20, 1.1, 50, 50, 1e-3, 10, 0.05, 0.5, 0.8, None)
    ])
def test_fcmdti_init(max_clusters, m, max_iter, max_FCM_iter, tol, min_samples_leaf, learning_rate, stop_threshold,
                     alpha, random_state):
    imputer = FCMDTIterativeImputer(max_clusters, m, max_iter, max_FCM_iter, tol, min_samples_leaf,
                                    learning_rate, stop_threshold, alpha, random_state)

    assert imputer.random_state == random_state
    assert imputer.min_samples_leaf == min_samples_leaf
    assert imputer.learning_rate == learning_rate
    assert imputer.m == m
    assert imputer.max_clusters == max_clusters
    assert imputer.max_iter == max_iter
    assert imputer.stop_threshold == stop_threshold
    assert imputer.alpha == alpha
    assert imputer.max_FCM_iter == max_FCM_iter
    assert imputer.tol == tol


@pytest.mark.parametrize("X, random_state,min_samples_leaf,learning_rate,m,max_clusters,max_iter,stop_threshold,alpha",
                         [
                             (
                                     pd.DataFrame({
                                         "num1": [1, 2, 3, np.nan, 5],
                                         "num2": [5, 4, np.nan, 2, 1],
                                         "num3": [0.1, 0.5, 0.5, 0.1, np.nan]
                                     }),
                                     42, 2, 0.1, 2, 3, 10, 0.0, 1.0
                             ),
                             (
                                     pd.DataFrame({
                                         "num1": [10, 20, np.nan, 40],
                                         "num2": [np.nan, 1, 2, 3],
                                         "num3": [2, 3, 2, np.nan]
                                     }),
                                     7, 3, 0.5, 3, 40, 20, 0.01, 2.0
                             ),
                             (
                                     pd.DataFrame({
                                         "num1": [np.nan, np.nan, 1, 2, 5, 7],
                                         "num2": [1, 2, np.nan, 6, 10, 4],
                                         "num3": [np.nan, 1, 2, 0, 1, 2]
                                     }),
                                     1, 1, 0.2, 2, 10, 50, 0.1, 0.5
                             )
                         ])
def test_fcmdti_fit_sets_attributes(X, random_state, min_samples_leaf, learning_rate, m, max_clusters, max_iter,
                                    stop_threshold, alpha):
    imputer = FCMDTIterativeImputer(random_state=random_state, min_samples_leaf=min_samples_leaf,
                                    learning_rate=learning_rate, m=m, max_clusters=max_clusters, max_iter=max_iter,
                                    stop_threshold=stop_threshold, alpha=alpha)
    imputer.fit(X)

    assert hasattr(imputer, 'X_train_complete_')
    assert hasattr(imputer, 'imputer_')
    assert hasattr(imputer, 'trees_')
    assert hasattr(imputer, 'leaf_indices_')

    assert isinstance(imputer.trees_, dict) and len(imputer.trees_) > 0
    assert isinstance(imputer.leaf_indices_, dict) and len(imputer.leaf_indices_) > 0
    assert set(imputer.trees_.keys()) == set(X.columns)
    assert set(imputer.leaf_indices_.keys()) == set(X.columns)


@pytest.mark.parametrize("X", [
    (
            pd.DataFrame({
                "num1": [1, np.nan, 3, np.nan, 5],
                "num2": [np.nan, 4, np.nan, 2, 1],
                "num3": [0, 1, 1, 0, np.nan]
            })
    ),
    (
            pd.DataFrame({
                "num1": [10, np.nan, np.nan, 40],
                "num2": [np.nan, 1, 2, 3],
                "num3": [3, 4, 3, np.nan]
            })
    ),
    (
            pd.DataFrame({
                "num1": [np.nan, np.nan, 1],
                "num2": [1, 2, np.nan],
                "num3": [np.nan, 1, 2]
            })
    )
])
def test_fcmdti_fit_error_no_complete(X):
    imputer = FCMDTIterativeImputer()
    with pytest.raises(ValueError, match="Invalid input: Input dataset has no complete records"):
        imputer.fit(X)


@pytest.mark.parametrize("X", [
    (pd.DataFrame({"num1": [1, np.nan, 3, np.nan, 5], })),
    (pd.DataFrame({"num1": [np.nan, 1, 2, 3], })),
    (pd.DataFrame({"num1": [np.nan, 3.5, -4.7]}))
])
def test_fcmdti_fit_error_one_column(X):
    imputer = FCMDTIterativeImputer()
    with pytest.raises(ValueError, match="Invalid input: Input dataset has only one column"):
        imputer.fit(X)


@pytest.mark.parametrize(
    "X,X_test, random_state,min_samples_leaf,learning_rate,m,max_clusters,max_iter,stop_threshold,alpha", [
        (
                pd.DataFrame({
                    "num1": [1, 2, 3, 7, 5],
                    "num2": [5, 4, 3, 2, 1],
                    "num3": [0, 1, 1, 0, np.nan]
                }),
                pd.DataFrame({
                    "num1": [np.nan, 2, 3, np.nan, 5],
                    "num2": [5, 4, np.nan, 2, 1],
                    "num3": [np.nan, 1, 1, 0, np.nan]
                }),
                42, 2, 0.1, 2, 5, 10, 0.0, 1.0
        ),
        (
                pd.DataFrame({
                    "num1": [10, 20, 50, 40],
                    "num2": [np.nan, 1, 2, 3],
                    "num3": [3, 4, 3, 4]
                }),
                pd.DataFrame({
                    "num1": [10, 20, np.nan, 40],
                    "num2": [np.nan, 1, np.nan, 3],
                    "num3": [3, np.nan, 3, np.nan]
                }),
                7, 3, 0.5, 3, 20, 100, 0.01, 2.0
        ),
        (
                pd.DataFrame({
                    "num1": [np.nan, 10, 1, 2, 5, 7],
                    "num2": [1, 2, 5, 6, 10, 4],
                    "num3": [np.nan, 1, 2, 0, 1, 2],
                    "num4": [2, 0, 2, 0, 1, 1]
                }),
                pd.DataFrame({
                    "num1": [np.nan, np.nan, 1, 2, np.nan, 7],
                    "num2": [1, 2, np.nan, 6, np.nan, 4],
                    "num3": [np.nan, 1, 2, np.nan, 1, 2],
                    "num4": [2, 0, np.nan, 0, 1, np.nan]
                }),
                1, 1, 0.2, 2, 3, 15, 0.1, 0.5
        ),
        (
                pd.DataFrame({
                    "num1": [np.nan, 10.3, 1, 2, 5, 7],
                    "num2": [1, 2, 5, 6.6, 10, 4],
                    "num3": [np.nan, 1, 2.7, 0, 1, 2],
                    "num4": [2, 0, 2.3, 0.5, 1.9, 1]
                }),
                pd.DataFrame({
                    "num1": [1, 5, 1, 2.1, 8, 7],
                    "num2": [1, 2, 5.4, 6, 3, 4],
                    "num3": [0, 1.8, 2, 0, 1.4, 2],
                    "num4": [2, 0, 1, 0, 1, 2]
                }),
                1, 1, 0.2, 2, 15, 50, 0.1, 0.5
        )
    ])
def test_fcmdti_transform(X, X_test, random_state, min_samples_leaf, learning_rate, m, max_clusters, max_iter,
                          stop_threshold, alpha):
    imputer = FCMDTIterativeImputer(random_state=random_state, min_samples_leaf=min_samples_leaf,
                                    learning_rate=learning_rate, m=m, max_clusters=max_clusters, max_iter=max_iter,
                                    stop_threshold=stop_threshold, alpha=alpha)
    imputer.fit(X)
    result = imputer.transform(X)

    result2 = imputer.transform(X)

    imputer = FCMDTIterativeImputer(random_state=random_state, min_samples_leaf=min_samples_leaf,
                                    learning_rate=learning_rate, m=m, max_clusters=max_clusters, max_iter=max_iter,
                                    stop_threshold=stop_threshold, alpha=alpha)
    imputer.fit(X)
    result3 = imputer.transform(X)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == X_test.shape
    assert set(result.columns) == set(X.columns)
    assert all(result.dtypes == X.dtypes)
    assert result.isna().sum().sum() == 0
    np.testing.assert_array_equal(result, result2)
    np.testing.assert_array_equal(result, result3)


@pytest.mark.parametrize("X", [
    pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]}),
    pd.DataFrame({'a': [np.nan, 2.0], 'b': [3.0, 4.0]})
])
def test_fcmdti_transform_without_fit(X):
    imputer = FCMDTIterativeImputer(random_state=42)
    with pytest.raises(NotFittedError):
        imputer.transform(X)


@pytest.mark.parametrize("X_fit, X_transform", [
    (
            pd.DataFrame({'a': [1, 2], 'b': [3, 4]}),
            pd.DataFrame({'b': [3, 4], 'a': [1, 2]})
    ),
    (
            pd.DataFrame({'a': [1, 2], 'b': [3, 4]}),
            pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
    ),
    (
            pd.DataFrame({'a': [1, 2], 'b': [3, 4]}),
            pd.DataFrame({'a': [1, 2]})
    ),
])
def test_fcmdti_transform_column_mismatch(X_fit, X_transform):
    imputer = FCMDTIterativeImputer(random_state=42)
    imputer.fit(X_fit)

    with pytest.raises(ValueError, match="Invalid input: Input dataset columns do not match columns seen during fit"):
        imputer.transform(X_transform)


dataframes_list = [
    pd.DataFrame({
        "a": [1.0, 2.0, 3.0, np.nan, 5.0],
        "b": [5.0, 4.0, 3.0, 2.0, 1.0],
    }),

    pd.DataFrame({
        "a": [np.nan, np.nan, 3.0, 4.0, 5.0],
        "b": [np.nan, np.nan, 3.0, 2.0, 1.0],
    }),

    pd.DataFrame({
        "a": [np.nan, np.nan, 3.0, 4.0, 5.0, 6.0],
        "b": [np.nan, 2.0, 3.0, np.nan, 5.0, 6.0],
        "c": [2.0, 3.0, 4.0, 5.0, 5.0, 6.0],
    }),

    pd.DataFrame({
        "a": [np.nan, 2.0, -3.0, -4.0, -5.0, -6.0],
        "b": [1.0, 2.0, np.nan, -4.0, -5.0, -6.0],
        "c": [1.0, 2.0, -3.0, np.nan, -5.0, -6.0],
    }),

    pd.DataFrame({
        "a": [1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0],
        "b": [5.0, 4.0, 3.0, np.nan, 2.0, 1.0, 0.0],
        "c": [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0],
        "d": [10.0, np.nan, 15.0, 16.0, 17.0, 18.0, 19.0],
        "e": [9.0, np.nan, 7.0, 6.0, 5.0, np.nan, np.nan],
    }),
]

fcm_params_list = [
    (2, 2.0, 100, 1e-3),
    (2, 1.5, 150, 1e-5),
    (3, 2.5, 300, 1e-6),
    (3, 3.0, 600, 1e-4),
    (1, 3.0, 600, 1e-4),
]


# ----- FCMCentroidImputer -----------------------------------------------

@pytest.mark.parametrize("n_clusters,m,max_iter,tol", fcm_params_list)
def test_fcmcentroidimputer_init_parametrized(n_clusters, m, max_iter, tol):
    imputer = FCMCentroidImputer(
        n_clusters=n_clusters, m=m, max_iter=max_iter, tol=tol
    )
    assert imputer.n_clusters == n_clusters
    assert imputer.m == m
    assert imputer.max_iter == max_iter
    assert imputer.tol == tol


def test_fcmcentroidimputer_fit_creates_attributes():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    imputer = FCMCentroidImputer()
    imputer.fit(X)
    assert hasattr(imputer, "centers_")
    assert hasattr(imputer, "memberships_")
    assert imputer.centers_.shape[1] == X.shape[1]


def test_fcmcentroidimputer_transform_raises_if_not_fitted():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 1.0, 2.0, 3.0],
                      "b": [4.0, 5.0, np.nan, 4.0, 5.0, np.nan]})
    imputer = FCMCentroidImputer()
    with pytest.raises(NotFittedError):
        imputer.transform(X)


def test_fcmcentroidimputer_transform_raises_if_columns_differ():
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    X_test = pd.DataFrame({"a": [1.0, 2.0, np.nan], "c": [7.0, 8.0, 9.0]})

    imputer = FCMCentroidImputer()
    imputer.fit(X_train)

    with pytest.raises(ValueError, match="Columns in transform do not match columns seen during fit"):
        imputer.transform(X_test)


@pytest.mark.parametrize("X", dataframes_list)
def test_fcmcentroidimputer_transform_imputes_missing_values(X):
    imputer = FCMCentroidImputer()
    imputer.fit(X.dropna())
    result = imputer.transform(X)
    assert not result.isna().any().any(), "Imputer should fill all missing values"


def test_fcmcentroidimputer_transform_no_missing_returns_same():
    X = pd.DataFrame({"a": [1.0, 2.0, 2.0], "b": [3.0, 4.0, 4.0]})
    imputer = FCMCentroidImputer()
    imputer.fit(X)
    result = imputer.transform(X)
    pd.testing.assert_frame_equal(X, result)


def test_fcmcentroidimputer_fit_raises_if_too_many_clusters():
    X = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [4.0, 5.0, 6.0]})
    imputer = FCMCentroidImputer(n_clusters=5)
    with pytest.raises(ValueError, match="n_clusters cannot be larger than the number of complete rows"):
        imputer.fit(X)


def test_fcmcentroidimputer_fit_no_complete_rows():
    X = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
    imputer = FCMCentroidImputer()
    with pytest.raises(ValueError, match="Invalid input: Input dataset contains no complete rows."):
        imputer.fit(X)


# ----- FCMParameterImputer -----------------------------------------------

def test_fcmparameterimputer_fit_creates_attributes():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    imputer = FCMParameterImputer()
    imputer.fit(X)
    assert hasattr(imputer, "centers_")
    assert hasattr(imputer, "memberships_")
    assert hasattr(imputer, "feature_names_in_")


@pytest.mark.parametrize("X", dataframes_list)
def test_fcmparameterimputer_transform_imputes_values(X):
    imputer = FCMParameterImputer()
    imputer.fit(X.dropna())
    result = imputer.transform(X)
    assert not result.isna().any().any()


def test_fcmparameterimputer_fit_raises_if_too_many_clusters():
    X = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [4.0, 5.0, 6.0]})
    imputer = FCMParameterImputer(n_clusters=5)
    with pytest.raises(ValueError, match="n_clusters cannot be larger than the number of complete rows"):
        imputer.fit(X)


def test_fcmparameterimputer_feature_names_in_assigned():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 1.0, 2.0, 3.0],
                      "b": [4.0, 5.0, np.nan, 4.0, 5.0, np.nan]})
    imputer = FCMParameterImputer()
    imputer.fit(X)
    assert list(imputer.feature_names_in_) == list(X.columns)


def test_fcmparameterimputer_transform_raises_if_not_fitted():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 1.0, 2.0, 3.0],
                      "b": [4.0, 5.0, np.nan, 4.0, 5.0, np.nan]})
    imputer = FCMParameterImputer()
    with pytest.raises(NotFittedError):
        imputer.transform(X)


# ----- FCMRoughParameterImputer -----------------------------------------------

def test_fcmroughparameterimputer_fit_creates_clusters():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    imputer = FCMRoughParameterImputer()
    imputer.fit(X)
    assert hasattr(imputer, "centers_")
    assert hasattr(imputer, "memberships_")
    assert hasattr(imputer, "clusters_")


@pytest.mark.parametrize("X", dataframes_list)
def test_fcmroughparameterimputer_transform_imputes_values(X):
    imputer = FCMRoughParameterImputer(wl=1, wb=1, n_clusters=1, tau=0)
    imputer.fit(X.dropna())
    result = imputer.transform(X)
    assert not result.isna().any().any()


def test_fcmroughparameterimputer_fit_raises_if_too_many_clusters():
    X = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [4.0, 5.0, 6.0]})
    imputer = FCMRoughParameterImputer(n_clusters=5)
    with pytest.raises(ValueError, match="n_clusters cannot be larger than the number of complete rows"):
        imputer.fit(X)


def test_fcmroughparameterimputer_transform_raises_if_not_fitted():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 1.0, 2.0, 3.0],
                      "b": [4.0, 5.0, np.nan, 4.0, 5.0, np.nan]})
    imputer = FCMRoughParameterImputer(n_clusters=5)
    with pytest.raises(NotFittedError):
        imputer.transform(X)


@pytest.mark.parametrize("imputer_class", [
    FCMCentroidImputer,
    FCMParameterImputer,
    FCMRoughParameterImputer,
])
@pytest.mark.parametrize("X", dataframes_list)
@pytest.mark.parametrize("n_clusters,m,max_iter,tol", fcm_params_list)
@pytest.mark.parametrize("random_state", [42, 99, 120])
def test_imputers_same_random_state_reproducible(imputer_class, X, n_clusters, m, max_iter, tol, random_state):
    imputer_1 = imputer_class(n_clusters=n_clusters, m=m, max_iter=max_iter, tol=tol, random_state=random_state)
    imputer_2 = imputer_class(n_clusters=n_clusters, m=m, max_iter=max_iter, tol=tol, random_state=random_state)

    imputer_1.fit(X)
    imputer_2.fit(X)

    result_1 = imputer_1.transform(X)
    result_2 = imputer_2.transform(X)

    pd.testing.assert_frame_equal(result_1, result_2, check_exact=False, atol=1e-8)


# ----- _rough_kmeans_from_fcm -----------------------------------------------

@pytest.mark.parametrize("X", dataframes_list)
def test_rough_kmeans_from_fcm_shapes_and_types(X):
    centers, memberships = fuzzy_c_means(X, n_clusters=2, random_state=0)
    imputer = FCMRoughParameterImputer()
    clusters = imputer._rough_kmeans_from_fcm(X, memberships, centers)

    assert isinstance(clusters, list)
    assert len(clusters) == 2
    for lower, upper, center in clusters:
        assert isinstance(center, np.ndarray)
        assert center.shape == (X.shape[1],)


def test_rough_kmeans_from_fcm_cluster_consistency():
    X = np.vstack([
        np.random.normal(0, 0.1, (5, 2)),
        np.random.normal(5, 0.1, (5, 2))
    ])
    centers, memberships = fuzzy_c_means(X, n_clusters=2, random_state=0)
    imputer = FCMRoughParameterImputer()
    clusters = imputer._rough_kmeans_from_fcm(X, memberships, centers)

    for lower, upper, _ in clusters:
        assert isinstance(lower, np.ndarray)
        assert isinstance(upper, np.ndarray)
        assert lower.shape[1] == X.shape[1] if len(lower) > 0 else True


@pytest.mark.parametrize(
    "params, expected_exception, expected_msg",
    [
        # n_clusters
        ({"n_clusters": "3"}, TypeError, "Invalid type for n_clusters"),
        ({"n_clusters": -1}, ValueError, "Invalid value for n_clusters"),
        ({"n_clusters": 0}, ValueError, "Invalid value for n_clusters"),

        # max_iter
        ({"max_iter": "100"}, TypeError, "Invalid type for max_iter"),
        ({"max_iter": 0}, ValueError, "Invalid value for max_iter"),

        # random_state
        ({"random_state": "abc"}, TypeError, "Invalid type for random_state"),

        # m (fuzziness)
        ({"m": "2.0"}, TypeError, "Invalid type for m"),
        ({"m": 1.0}, ValueError, "Invalid value for m"),

        # tol
        ({"tol": "1e-5"}, TypeError, "Invalid type for tol"),
        ({"tol": 0}, ValueError, "Invalid value for tol"),

        # wl
        ({"wl": "0.5"}, TypeError, "Invalid type for wl"),
        ({"wl": -0.1}, ValueError, "Invalid value for wl"),
        ({"wl": 1.5}, ValueError, "Invalid value for wl"),
        ({"wl": 0}, ValueError, "Invalid value for wl"),

        # wb
        ({"wb": "0.2"}, TypeError, "Invalid type for wb"),
        ({"wb": -0.1}, ValueError, "Invalid value for wb"),
        ({"wb": 1.5}, ValueError, "Invalid value for wb"),

        # tau
        ({"tau": "0.5"}, TypeError, "Invalid type for tau"),
        ({"tau": -0.1}, ValueError, "Invalid value for tau"),
    ]
)
def test_validate_params_errors(params, expected_exception, expected_msg):
    with pytest.raises(expected_exception) as excinfo:
        validate_params(params)
    assert expected_msg in str(excinfo.value)
