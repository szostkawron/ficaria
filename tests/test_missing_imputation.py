import os
import sys

import pytest
from sklearn.exceptions import NotFittedError

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ficaria.missing_imputation import *


@pytest.mark.parametrize("random_state", [
    42,
    None,
    123,

])
def test_kiimputer_init(random_state):
    imputer = KIImputer(random_state=random_state)

    assert imputer.random_state == random_state


@pytest.mark.parametrize("random_state", [
    "txt",
    [24],
    [[35]],
    3.5
])
def test_kiimputer_init_errors(random_state):
    with pytest.raises(TypeError,
                       match="Invalid random_state: Expected an integer or None."):
        imputer = KIImputer(random_state=random_state)


@pytest.mark.parametrize("X, random_state", [
    (pd.DataFrame({
        'a': [np.nan, 2.0, 3.0],
        'b': [4.0, 5.0, np.nan]
    }), 42),
    (pd.DataFrame({
        'a': [1.0, 2.0],
        'b': [np.nan, 6.0]
    }), 42)
])
def test_kiimputer_fit(X, random_state):
    imputer = KIImputer(random_state=random_state)
    imputer.fit(X)

    assert hasattr(imputer, 'X_train_')
    pd.testing.assert_frame_equal(imputer.X_train_, X)

    assert hasattr(imputer, 'np_rng_')
    assert isinstance(imputer.np_rng_, np.random.RandomState)


@pytest.mark.parametrize("X, X_test, random_state", [
    (pd.DataFrame({
        'a': [np.nan, 2.0, 3.0],
        'b': [4.0, 5.0, np.nan]
    }),
     pd.DataFrame({
         'a': [2.0, np.nan, 6.0],
         'b': [8.0, 10.0, 12.0]
     }),
     42),
    (pd.DataFrame({
        'a': [1.0, 2.0],
        'b': [np.nan, 6.0]
    }),
     pd.DataFrame({
         'a': [3.0, np.nan],
         'b': [9.0, 6.0]
     }),
     42)
])
def test_kiimputer_transform(X, X_test, random_state):
    imputer = KIImputer(random_state=random_state)
    imputer.fit(X)
    result = imputer.transform(X_test)

    imputer2 = KIImputer(random_state=random_state)
    imputer2.fit(X)
    result2 = imputer2.transform(X_test)

    assert isinstance(result, np.ndarray)
    assert result.shape == X_test.shape
    assert not np.isnan(result).any()
    np.testing.assert_array_equal(result, result2)


@pytest.mark.parametrize("random_state, max_clusters, m", [
    (42, 5, 1.1),
    (None, 8, 25),
    (123, 20, 3.1)
])
def test_fcmkiimputer_init(random_state, max_clusters, m):
    imputer = FCMKIterativeImputer(random_state=random_state, max_clusters=max_clusters, m=m)
    assert imputer.random_state == random_state
    assert imputer.max_clusters == max_clusters
    assert imputer.m == m


@pytest.mark.parametrize("random_state", [
    "txt",
    [24],
    [[35]],
    3.5
])
def test_fcmkiimputer_init_errors_randomstate(random_state):
    with pytest.raises(TypeError,
                       match="Invalid random_state: Expected an integer or None."):
        imputer = FCMKIterativeImputer(random_state=random_state)


@pytest.mark.parametrize("max_clusters", [
    "txt",
    [24],
    [[35]],
    3.5,
    0,
    -5,
    1
])
def test_fcmkiimputer_init_errors_maxclusters(max_clusters):
    with pytest.raises(TypeError,
                       match="Invalid max_clusters: Expected an integer greater than 1."):
        imputer = FCMKIterativeImputer(max_clusters=max_clusters)


@pytest.mark.parametrize("m", [
    "txt",
    [24],
    [[35]],
    0,
    0.5,
    -5,
    1
])
def test_fcmkiimputer_init_errors_m(m):
    with pytest.raises(TypeError,
                       match="Invalid m value: Expected a numeric value greater than 1."):
        imputer = FCMKIterativeImputer(m=m)


@pytest.mark.parametrize("X, random_state, max_clusters, m", [
    (pd.DataFrame({
        'a': [np.nan, 2.0, 3.0],
        'b': [4.0, 5.0, np.nan]
    }), 42, 5, 1.5),
    (pd.DataFrame({
        'a': [1.0, 2.0],
        'b': [np.nan, 6.0]
    }), 42, 10, 2)
])
def test_fcmkiimputer_fit(X, random_state, max_clusters, m):
    imputer = FCMKIterativeImputer(random_state=random_state, max_clusters=max_clusters, m=m)
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

    assert isinstance(result, np.ndarray)
    assert result.shape == X_test.shape
    assert not np.isnan(result).any()
    np.testing.assert_array_equal(result, result2)


##################

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
        "cat1": ["A", "B", "A", "B"]
    }), 2, 0.8),
    (pd.DataFrame({
        "cat1": ["A", "B", "A", "C", "A", "B", "C"],
        "cat2": ["X", "X", "Y", "Y", "X", "Y", "X"]
    }), 3, 1.1),
    (pd.DataFrame({
        "cat1": ["A"],
        "cat2": ["X"]
    }), 1, 1.1),
])
def test_fcmdti_fuzzy_silhouette(X, n_clusters, alpha):
    imputer = FCMDTIterativeImputer()
    centers, u = fuzzy_c_means_categorical(X, n_clusters)
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
        "cat1": ["A", "B", "A", "B"]
    }), 20, 123),
    (pd.DataFrame({
        "cat1": ["A", "B", "A", "C", "A", "B", "C"],
        "cat2": ["X", "X", "Y", "Y", "X", "Y", "X"]
    }), 5, 30),
    (pd.DataFrame({
        "cat1": ["A"],
        "cat2": ["X"]
    }), 30, 100),
])
def test_fcmdti_determine_optimal_n_clusters_FSI(X, max_clusters, random_state):
    imputer = FCMDTIterativeImputer(random_state=random_state, max_clusters=max_clusters)
    imputer.fit(X)
    if X.select_dtypes(exclude=["number"]).empty:
        fcm_function = fuzzy_c_means
    else:
        fcm_function = fuzzy_c_means_categorical

    n_clusters = imputer._determine_optimal_n_clusters_FSI(X, fcm_function)
    assert isinstance(n_clusters, int)
    assert n_clusters > 0 and n_clusters <= min(len(X), max_clusters)
    n_clusters2 = imputer._determine_optimal_n_clusters_FSI(X, fcm_function)
    assert n_clusters == n_clusters2
    if len(X) == 1 or (X.nunique(axis=0) == 1).all():
        assert n_clusters == 1


def test_fcmdti_create_mixed_imputer():
    X = pd.DataFrame({
        "num1": [1, 2, 3, 4],
        "cat1": ["A", "B", "A", "B"]
    })
    mixed_imputer = FCMDTIterativeImputer()._create_mixed_imputer(X)
    assert isinstance(mixed_imputer, ColumnTransformer)
    assert set(mixed_imputer.transformers[0][2]) == {"num1"}
    assert set(mixed_imputer.transformers[1][2]) == {"cat1"}


@pytest.mark.parametrize("X", [
    pd.DataFrame({
        "num1": [1, np.nan, 5, 3, np.nan],
        "cat1": ["A", "B", np.nan, "A", np.nan]
    }),
    pd.DataFrame({
        "num2": [10, 20, np.nan, 30, 40],
        "cat2": [np.nan, "X", "Y", "X", "X"]
    }),
    pd.DataFrame({
        "num3": [np.nan, np.nan, 2, 4, 6],
        "cat3": ["C", "C", np.nan, "D", "C"]
    }),
    pd.DataFrame({
        "x1": [1.0, 2.0, 3.0],
        "x2": [6.0, 5.0, 4.0]
    }),
    pd.DataFrame({
        "cat1": ["A", "B", "A", "C", "A", "B", "C"],
        "cat2": ["X", "X", "Y", "Y", "X", "Y", "X"]
    }),
])
def test_fcmdti_create_mixed_imputer_fill_missing(X):
    imputer = FCMDTIterativeImputer()._create_mixed_imputer(X)
    X_transformed = imputer.transform(X)
    X_transformed.columns = imputer.get_feature_names_out()
    X_transformed.columns = [col.split("__")[-1] for col in X_transformed.columns]
    X_transformed_df = X_transformed[X.columns]

    assert not X_transformed_df.isna().any().any()

    for col in X.columns:
        mask = X[col].isna()
        if is_numeric_dtype(X[col]):
            mean_val = X[col].mean(skipna=True)
            assert np.all(np.isclose(X_transformed_df.loc[mask, col], mean_val))
        else:
            mode_val = X[col].mode()[0]
            assert np.all(X_transformed_df.loc[mask, col] == mode_val)

    X_transformed2 = imputer.transform(X)

    np.testing.assert_array_equal(X_transformed, X_transformed2)


@pytest.mark.parametrize("new_df, old_df, mask_missing, expected", [
    (pd.DataFrame({
        "num1": [1, 2, 5, 3, 8],
        "cat1": ["A", "B", "A", "A", "B"]
    }),
     pd.DataFrame({
         "num1": [1, 2, 6, 3, 8],
         "cat1": ["A", "B", "A", "A", "B"]
     }),
     pd.DataFrame({
         "num1": [False, False, True, False, False],
         "cat1": [False, False, False, False, False]
     }), 1
    ),
    (pd.DataFrame({
        "num1": [1, 2, 5, 3, 8],
        "cat1": ["A", "B", "A", "A", "B"]
    }),
     pd.DataFrame({
         "num1": [1, 2, 5, 3, 8],
         "cat1": ["A", "A", "A", "A", "B"]
     }),
     pd.DataFrame({
         "num1": [False, False, False, False, False],
         "cat1": [False, True, False, False, False]
     }), 1
    ),
    (pd.DataFrame({
        "num1": [1, 2, 5, 3, 8],
        "cat1": ["A", "B", "A", "A", "B"]
    }),
     pd.DataFrame({
         "num1": [1, 2, 7, 3, 8],
         "cat1": ["A", "A", "A", "A", "B"]
     }),
     pd.DataFrame({
         "num1": [False, False, True, False, False],
         "cat1": [False, True, False, False, False]
     }), 1.5
    ),
    (pd.DataFrame({
        "num1": [1, 2, 5, 3, 8],
        "cat1": ["A", "B", "A", "A", "B"]
    }),
     pd.DataFrame({
         "num1": [1, 2, 5, 3, 8],
         "cat1": ["A", "B", "A", "A", "B"]
     }),
     pd.DataFrame({
         "num1": [False, False, False, False, False],
         "cat1": [False, False, False, False, False]
     }), 0
    ),
    (pd.DataFrame({
        "num1": [1, 2, 5, 3, 8],
        "cat1": ["A", "B", "A", "A", "B"]
    }),
     pd.DataFrame({
         "num1": [1, 2, 5, 3, 8],
         "cat1": ["A", "B", "A", "A", "B"]
     }),
     pd.DataFrame({
         "num1": [True, True, True, True, True],
         "cat1": [True, True, True, True, True]
     }), 0
    ),
    (
            pd.DataFrame({
                "num1": [1, 2, 3],
                "cat1": ["A", "B", "C"]
            }),
            pd.DataFrame({
                "num1": [1, 3, 6],
                "cat1": ["A", "X", "C"]
            }),
            pd.DataFrame({
                "num1": [True, True, True],
                "cat1": [True, True, True]
            }),
            ((0 + 1 + 3) / 3 + (0 + 1 + 0) / 3) / 2
    ),
    (
            pd.DataFrame({
                "cat1": ["A", "B", "C"]
            }),
            pd.DataFrame({
                "cat1": ["A", "X", "C"]
            }),
            pd.DataFrame({
                "cat1": [True, True, True]
            }),
            1 / 3
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
                "cat1": ["A", "B", "C"]
            }),
            pd.DataFrame({
                "num1": [1, 4, 3],
                "cat1": ["A", "X", "C"]
            }),
            pd.DataFrame({
                "num1": [False, True, False],
                "cat1": [False, True, False]
            }),
            (abs(2) + 1) / 2
    ),

])
def test_fcmdti_calculate_AV(new_df, old_df, mask_missing, expected):
    AV = FCMDTIterativeImputer()._calculate_AV(new_df, old_df, mask_missing)
    assert isinstance(AV, float)
    assert AV == pytest.approx(expected)


@pytest.mark.parametrize("X, random_state", [
    (pd.DataFrame({
        "num1": [1, np.nan, 5, 3, np.nan, 8],
        "cat1": ["A", "B", np.nan, "A", np.nan, "B"]
    }), 42),
    (pd.DataFrame({
        "num2": [10, 20, np.nan, 30, 40],
        "cat2": [np.nan, "X", "Y", "X", "Y"]
    }), 110),
    (pd.DataFrame({
        "num3": [np.nan, np.nan, 2, 4, 6],
        "cat3": ["C", "C", np.nan, "D", "C"]
    }), 0),
    (pd.DataFrame({
        "x1": [1.0, 2.0, 3.0],
        "x2": [6.0, 5.0, 4.0]
    }), 23),
    (pd.DataFrame({
        "cat1": ["A", "B", "A", "C", np.nan, "B", "C"],
        "cat2": ["X", "X", "Y", "Y", np.nan, "Y", "X"]
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
        "num1": [1, np.nan, 5, 3, np.nan, 8, 10, 2, 13],
        "cat1": ["A", "B", np.nan, "A", np.nan, "B", "A", "C", "A"]
    }), "num1", 1, 42),
    (pd.DataFrame({
        "num2": [10, 20, np.nan, 30, 40, 100, 50],
        "cat2": [np.nan, "X", "Y", "X", "Y", "X", "Y"]
    }), "cat2", 1, 100),
    (pd.DataFrame({
        "num3": [np.nan, np.nan, 2, 4, 6],
        "cat3": ["C", "C", np.nan, "D", "C"]
    }), "num3", 0, 0),
    (pd.DataFrame({
        "cat1": ["A", "B", "A", "C", "A", "B", "C"],
        "cat2": ["X", "X", "Y", "Y", np.nan, "Y", "X"]
    }), "cat2", 1, 23),
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
    if X.select_dtypes(exclude=["number"]).empty:
        fcm_function = fuzzy_c_means
    else:
        fcm_function = fuzzy_c_means_categorical
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
    "random_state, min_samples_leaf, learning_rate, m, max_clusters, max_iter, stop_threshold, alpha", [
        (42, 0.1, 0.1, 1.1, 3, 10, 0, 0.1),
        (None, 5, 1, 2.0, 20, 100, 1, 0.5),
        (123, 10, 0.5, 1.5, 100, 50, 0.5, 0.9)

    ])
def test_fcmdti_init(random_state, min_samples_leaf, learning_rate, m, max_clusters, max_iter, stop_threshold, alpha):
    imputer = FCMDTIterativeImputer(random_state, min_samples_leaf, learning_rate, m, max_clusters, max_iter,
                                    stop_threshold, alpha)

    assert imputer.random_state == random_state
    assert imputer.min_samples_leaf == min_samples_leaf
    assert imputer.learning_rate == learning_rate
    assert imputer.m == m
    assert imputer.max_clusters == max_clusters
    assert imputer.max_iter == max_iter
    assert imputer.stop_threshold == stop_threshold
    assert imputer.alpha == alpha


@pytest.mark.parametrize("random_state", [
    "txt",
    [24],
    [[35]],
    3.5
])
def test_fcmdti_init_errors_randomstate(random_state):
    with pytest.raises(TypeError,
                       match="Invalid random_state: Expected an integer or None"):
        imputer = FCMDTIterativeImputer(random_state=random_state)


invalid_values = ["txt", [24], [[35]], 3.5, 0, -5, 1]
params_to_test = ["max_iter", "max_clusters"]


@pytest.mark.parametrize("param_name,value", [(p, v) for p in params_to_test for v in invalid_values])
def test_fcmdti_init_errors(param_name, value):
    kwargs = {param_name: value}
    with pytest.raises(TypeError, match=f"Invalid {param_name} value: Expected an integer greater than 1."):
        imputer = FCMDTIterativeImputer(**kwargs)


@pytest.mark.parametrize("m", [
    "txt",
    [24],
    [[35]],
    0,
    0.5,
    -5,
    1
])
def test_fcmdti_init_errors_m(m):
    with pytest.raises(TypeError,
                       match="Invalid m value: Expected a numeric value greater than 1"):
        imputer = FCMDTIterativeImputer(m=m)


invalid_values = ["txt", [24], [[35]], -3.5, -5]
params_to_test = ["min_samples_leaf", "learning_rate", "stop_threshold", "alpha"]


@pytest.mark.parametrize("param_name,value", [(p, v) for p in params_to_test for v in invalid_values])
def test_fcmdti_init_errors(param_name, value):
    kwargs = {param_name: value}
    with pytest.raises(TypeError, match=f"Invalid {param_name} value: Expected a numeric value"):
        imputer = FCMDTIterativeImputer(**kwargs)


@pytest.mark.parametrize("X, random_state,min_samples_leaf,learning_rate,m,max_clusters,max_iter,stop_threshold,alpha",
                         [
                             (
                                     pd.DataFrame({
                                         "num1": [1, 2, 3, np.nan, 5],
                                         "num2": [5, 4, np.nan, 2, 1],
                                         "cat": ["a", "b", "b", "a", np.nan]
                                     }),
                                     42, 2, 0.1, 2, 3, 10, 0.0, 1.0
                             ),
                             (
                                     pd.DataFrame({
                                         "num1": [10, 20, np.nan, 40],
                                         "num2": [np.nan, 1, 2, 3],
                                         "cat": ["x", "y", "x", np.nan]
                                     }),
                                     7, 3, 0.5, 3, 40, 20, 0.01, 2.0
                             ),
                             (
                                     pd.DataFrame({
                                         "num1": [np.nan, np.nan, 1, 2, 5, 7],
                                         "num2": [1, 2, np.nan, 6, 10, 4],
                                         "cat": [np.nan, "b", "c", "a", "b", "c"]
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
    assert hasattr(imputer, 'encoders_')

    assert isinstance(imputer.trees_, dict) and len(imputer.trees_) > 0
    assert isinstance(imputer.leaf_indices_, dict) and len(imputer.leaf_indices_) > 0
    assert isinstance(imputer.encoders_, dict)
    assert set(imputer.encoders_.keys()) == set(X.select_dtypes(exclude=["number"]).columns)
    assert set(imputer.trees_.keys()) == set(X.columns)
    assert set(imputer.leaf_indices_.keys()) == set(X.columns)


@pytest.mark.parametrize("X", [
    (
            pd.DataFrame({
                "num1": [1, np.nan, 3, np.nan, 5],
                "num2": [np.nan, 4, np.nan, 2, 1],
                "cat": ["a", "b", "b", "a", np.nan]
            })
    ),
    (
            pd.DataFrame({
                "num1": [10, np.nan, np.nan, 40],
                "num2": [np.nan, 1, 2, 3],
                "cat": ["x", "y", "x", np.nan]
            })
    ),
    (
            pd.DataFrame({
                "num1": [np.nan, np.nan, 1],
                "num2": [1, 2, np.nan],
                "cat": [np.nan, "b", "c"]
            })
    )
])
def test_fcmdti_fit_error_no_complete(X):
    imputer = FCMDTIterativeImputer()
    with pytest.raises(ValueError, match="Invalid input: Input dataset has no complete records"):
        imputer.fit(X)


@pytest.mark.parametrize("X", [
    (
            pd.DataFrame({
                "num1": [1, np.nan, 3, np.nan, 5],
            })
    ),
    (
            pd.DataFrame({
                "num1": [np.nan, 1, 2, 3],
            })
    ),
    (
            pd.DataFrame({
                "cat": [np.nan, "b", "c"]
            })
    )
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
                    "cat": ["a", "b", "b", "a", np.nan]
                }),
                pd.DataFrame({
                    "num1": [np.nan, 2, 3, np.nan, 5],
                    "num2": [5, 4, np.nan, 2, 1],
                    "cat": [np.nan, "b", "b", "a", np.nan]
                }),
                42, 2, 0.1, 2, 5, 10, 0.0, 1.0
        ),
        (
                pd.DataFrame({
                    "num1": [10, 20, 50, 40],
                    "num2": [np.nan, 1, 2, 3],
                    "cat": ["x", "y", "x", "y"]
                }),
                pd.DataFrame({
                    "num1": [10, 20, np.nan, 40],
                    "num2": [np.nan, 1, np.nan, 3],
                    "cat": ["x", np.nan, "x", np.nan]
                }),
                7, 3, 0.5, 3, 20, 100, 0.01, 2.0
        ),
        (
                pd.DataFrame({
                    "num1": [np.nan, 10, 1, 2, 5, 7],
                    "num2": [1, 2, 5, 6, 10, 4],
                    "cat1": [np.nan, "b", "c", "a", "b", "c"],
                    "cat2": ["z", "x", "z", "x", "y", "y"]
                }),
                pd.DataFrame({
                    "num1": [np.nan, np.nan, 1, 2, np.nan, 7],
                    "num2": [1, 2, np.nan, 6, np.nan, 4],
                    "cat1": [np.nan, "b", "c", np.nan, "b", "c"],
                    "cat2": ["z", "x", np.nan, "x", "y", np.nan]
                }),
                1, 1, 0.2, 2, 3, 15, 0.1, 0.5
        ),
        (
                pd.DataFrame({
                    "num1": [np.nan, 10, 1, 2, 5, 7],
                    "num2": [1, 2, 5, 6, 10, 4],
                    "cat1": [np.nan, "b", "c", "a", "b", "c"],
                    "cat2": ["z", "x", "z", "x", "y", "y"]
                }),
                pd.DataFrame({
                    "num1": [1, 5, 1, 2, 8, 7],
                    "num2": [1, 2, 5, 6, 3, 4],
                    "cat1": ["a", "b", "c", "a", "b", "c"],
                    "cat2": ["z", "x", "y", "x", "y", "z"]
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


#######################################

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
    with pytest.raises(AttributeError, match="fit must be called before transform"):
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
    with pytest.raises(ValueError, match="No complete rows found for fitting"):
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
    imputer = FCMRoughParameterImputer()
    imputer.fit(X.dropna())
    result = imputer.transform(X)
    assert not result.isna().any().any()


def test_fcmroughparameterimputer_fit_raises_if_too_many_clusters():
    X = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [4.0, 5.0, 6.0]})
    imputer = FCMRoughParameterImputer(n_clusters=5)
    with pytest.raises(ValueError, match="n_clusters cannot be larger than the number of complete rows"):
        imputer.fit(X)


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
