import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from ficaria.missing_imputation import KIImputer, FCMKIterativeImputer, LinearInterpolationBasedIterativeIntuitionisticFuzzyCMeans


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



## ---------LinearInterpolationBasedIterativeIntuitionisticFuzzyCMeans-------------------------


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
    imputer = LinearInterpolationBasedIterativeIntuitionisticFuzzyCMeans()
    with pytest.raises((TypeError, ValueError)):
        imputer.fit(bad_X)


def test_liiifcm_transform_before_fit():
    X = pd.DataFrame({'a': [0.1, np.nan, 0.5]})
    imputer = LinearInterpolationBasedIterativeIntuitionisticFuzzyCMeans()
    with pytest.raises(AttributeError, match="must call fit"):
        imputer.transform(X)


def test_liiifcm_sigma_branch():
    X = pd.DataFrame({
        'a': [0.1, 0.5, np.nan],
        'b': [0.2, np.nan, 0.8]
    })
    imputer = LinearInterpolationBasedIterativeIntuitionisticFuzzyCMeans(n_clusters=3, sigma=True, random_state=42)
    imputer.fit(X)
    result = imputer.transform(X)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == X.shape
    assert not result.isnull().any().any()

@pytest.mark.parametrize("random_state", [42, 0, None])
def test_liiifcm_init_random_state(random_state):
    imputer = LinearInterpolationBasedIterativeIntuitionisticFuzzyCMeans(random_state=random_state)
    assert imputer.random_state == random_state


@pytest.mark.parametrize("random_state", ["abc", [1], 3.14])
def test_liiifcm_init_random_state_invalid(random_state):
    with pytest.raises(TypeError, match="Invalid random_state: Expected an integer or None."):
        LinearInterpolationBasedIterativeIntuitionisticFuzzyCMeans(random_state=random_state)


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

    imputer1 = LinearInterpolationBasedIterativeIntuitionisticFuzzyCMeans(**params)
    imputer2 = LinearInterpolationBasedIterativeIntuitionisticFuzzyCMeans(**params)

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
    imputer = LinearInterpolationBasedIterativeIntuitionisticFuzzyCMeans(
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
        LinearInterpolationBasedIterativeIntuitionisticFuzzyCMeans(**kwargs)


@pytest.mark.parametrize("X", [
    pd.DataFrame({'a': [np.nan, 0.5, 1.0], 'b': [0.3, np.nan, 0.9]}),
    pd.DataFrame({'x': [0.1, 0.2], 'y': [0.3, np.nan]}),
])
def test_liiifcm_fit_transform(X):
    imputer = LinearInterpolationBasedIterativeIntuitionisticFuzzyCMeans(n_clusters=3)
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
    imputer = LinearInterpolationBasedIterativeIntuitionisticFuzzyCMeans()
    U_star, V_star, J_history = imputer._ifcm(X)
    assert isinstance(U_star, np.ndarray)
    assert isinstance(V_star, np.ndarray)
    assert isinstance(J_history, list)
    assert U_star.shape[1] == imputer.n_clusters
    assert V_star.shape[0] == imputer.n_clusters
    assert V_star.shape[1] == X.shape[1]

