import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
import os, sys
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

## ---------FCMInterpolationIterativeImputer-------------------------


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
    with pytest.raises(AttributeError, match="must call fit"):
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


@pytest.mark.parametrize("random_state", ["abc", [1], 3.14])
def test_liiifcm_init_random_state_invalid(random_state):
    with pytest.raises(TypeError, match="Invalid random_state: Expected an integer or None."):
        FCMInterpolationIterativeImputer(random_state=random_state)


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
        stop_criteria=0.01,
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


@pytest.mark.parametrize("n_clusters, m, alpha, max_iter, tol, max_outer_iter, stop_criteria, sigma", [
    (3, 2.0, 2.0, 100, 1e-5, 20, 0.01, False),
    (5, 1.5, 3.0, 50, 1e-4, 10, 0.05, True),
])
def test_liiifcm_init(n_clusters, m, alpha, max_iter, tol, max_outer_iter, stop_criteria, sigma):
    imputer = FCMInterpolationIterativeImputer(
        n_clusters=n_clusters,
        m=m,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        max_outer_iter=max_outer_iter,
        stop_criteria=stop_criteria,
        sigma=sigma
    )

    assert imputer.n_clusters == n_clusters
    assert imputer.m == m
    assert imputer.alpha == alpha
    assert imputer.max_iter == max_iter
    assert imputer.tol == tol
    assert imputer.max_outer_iter == max_outer_iter
    assert imputer.stop_criteria == stop_criteria
    assert imputer.sigma == sigma


@pytest.mark.parametrize("param,value", [
    ("n_clusters", 1),
    ("m", 1.0),
    ("alpha", -1),
    ("max_iter", 0),
    ("tol", -1e-5),
    ("max_outer_iter", 0),
    ("stop_criteria", 0),
    ("sigma", "yes"),
])
def test_liiifcm_init_invalid(param, value):
    kwargs = dict()
    kwargs[param] = value
    with pytest.raises(TypeError):
        FCMInterpolationIterativeImputer(**kwargs)


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
