import pandas as pd
import numpy as np
import pytest
import random
from fuzzycmeans import FCM
from sklearn.impute import SimpleImputer
from tensorflow.python.framework.test_ops import None_

from ficaria.missing_imputation import KIImputer


@pytest.mark.parametrize("random_state", [
    42,
    None,
    123

])
def test_kiimputer_init(random_state):
    imputer = KIImputer(random_state=random_state)

    assert imputer.random_state == random_state


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

    assert hasattr(imputer, 'rng_')
    assert isinstance(imputer.rng_, random.Random)


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

    assert isinstance(result, np.ndarray)
    assert result.shape == X_test.shape
    assert not np.isnan(result).any()
