import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from ficaria.missing_imputation import KIImputer, FCMKIterativeImputer


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
