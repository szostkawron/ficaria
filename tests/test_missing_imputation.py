import numpy as np
import pandas as pd
import pytest

from sklearn.impute import SimpleImputer
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ficaria.missing_imputation import *
import ficaria.utils as utils


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



@pytest.fixture(autouse=True)
def mock_dependencies(monkeypatch):
    def mock_check_input_dataset(X, require_numeric=True):
        return X
    def mock_split_complete_incomplete(X):
        complete = X.dropna()
        incomplete = X[X.isna().any(axis=1)]
        return complete, incomplete
    def mock_fuzzy_c_means(X, n_clusters=3, m=2.0, max_iter=100, tol=1e-5, **kwargs):
        n_features = X.shape[1]
        centers = np.arange(n_clusters * n_features).reshape(n_clusters, n_features) + 1.0
        memberships = np.random.rand(X.shape[0], n_clusters)
        return centers, memberships
    def mock_rough_kmeans_from_fcm(X, memberships, centers, wl, wb, tau, max_iter, tol):
        lower = [np.array([[1, 2], [1, 3]])]
        upper = [np.array([[2, 3], [2, 4]])]
        clusters = [(lower[0], upper[0], centers[0])]
        return clusters
    def mock_euclidean_distance(a, b):
        a = np.nan_to_num(a, nan=0.0)
        b = np.nan_to_num(b, nan=0.0)
        return np.linalg.norm(a - b)

    monkeypatch.setattr(utils, "check_input_dataset", mock_check_input_dataset)
    monkeypatch.setattr(utils, "split_complete_incomplete", mock_split_complete_incomplete)
    monkeypatch.setattr(utils, "fuzzy_c_means", mock_fuzzy_c_means)
    monkeypatch.setattr(utils, "rough_kmeans_from_fcm", mock_rough_kmeans_from_fcm)
    monkeypatch.setattr(utils, "euclidean_distance", mock_euclidean_distance)


def test_fcmcentroidimputer_init():
    imputer = FCMCentroidImputer(n_clusters=4, m=2.5, max_iter=200, tol=1e-4)
    assert imputer.n_clusters == 4
    assert imputer.m == 2.5
    assert imputer.max_iter == 200
    assert imputer.tol == 1e-4


def test_fcmcentroidimputer_fit_creates_attributes():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    imputer = FCMCentroidImputer()
    imputer.fit(X)
    assert hasattr(imputer, "centers_")
    assert hasattr(imputer, "memberships_")
    assert imputer.centers_.shape[1] == X.shape[1]


def test_fcmcentroidimputer_transform_imputes_missing_values():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan]})
    imputer = FCMCentroidImputer()
    imputer.fit(X.dropna())
    result = imputer.transform(X)
    assert not result.isna().any().any(), "Imputer should fill all missing values"


def test_fcmcentroidimputer_transform_no_missing_returns_same():
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    imputer = FCMCentroidImputer()
    imputer.fit(X)
    result = imputer.transform(X)
    pd.testing.assert_frame_equal(X, result)


def test_fcmparameterimputer_fit_creates_attributes():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    imputer = FCMParameterImputer()
    imputer.fit(X)
    assert hasattr(imputer, "centers_")
    assert hasattr(imputer, "memberships_")
    assert hasattr(imputer, "feature_names_in_")


def test_fcmparameterimputer_transform_imputes_values():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan]})
    imputer = FCMParameterImputer()
    imputer.fit(X.dropna())
    result = imputer.transform(X)
    assert not result.isna().any().any()


def test_fcmroughparameterimputer_fit_creates_clusters():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    imputer = FCMRoughParameterImputer()
    imputer.fit(X)
    assert hasattr(imputer, "centers_")
    assert hasattr(imputer, "memberships_")
    assert hasattr(imputer, "clusters_")


def test_fcmroughparameterimputer_transform_imputes_values():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan]})
    imputer = FCMRoughParameterImputer()
    imputer.fit(X.dropna())
    result = imputer.transform(X)
    assert not result.isna().any().any()

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