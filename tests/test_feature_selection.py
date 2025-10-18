import pytest
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from ficaria.feature_selection import FuzzyImplicationGranularityFeatureSelection

@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        "a": [0.1, 0.4, 0.5, 0.9, 0.3],
        "b": [1, 2, 1, 2, 1],
        "c": ["x", "y", "x", "x", "y"]
    })
    y = pd.Series([0, 1, 0, 1, 0])
    return X, y


def test_init_invalid_classifier():
    with pytest.raises(ValueError):
        FuzzyImplicationGranularityFeatureSelection("not_a_model")

def test_fit_and_transform(sample_data):
    X, y = sample_data
    clf = DecisionTreeClassifier(random_state=42)
    selector = FuzzyImplicationGranularityFeatureSelection(clf, eps=0.5, d=3, random_state=42)
    selector.fit(X, y)
    transformed = selector.transform(X)
    assert isinstance(transformed, pd.DataFrame)
    assert all(col in X.columns for col in transformed.columns)

def test_deterministic_results(sample_data):
    X, y = sample_data
    clf = DecisionTreeClassifier(random_state=42)
    selector1 = FuzzyImplicationGranularityFeatureSelection(clf, eps=0.5, d=3, random_state=123)
    selector2 = FuzzyImplicationGranularityFeatureSelection(clf, eps=0.5, d=3, random_state=123)

    selector1.fit(X, y)
    selector2.fit(X, y)
    assert selector1.S == selector2.S
    transformed1 = selector1.transform(X)
    transformed2 = selector2.transform(X)
    pd.testing.assert_frame_equal(transformed1, transformed2)

def test_transform_without_fit(sample_data):
    X, y = sample_data
    clf = DecisionTreeClassifier()
    selector = FuzzyImplicationGranularityFeatureSelection(clf)
    with pytest.raises(RuntimeError):
        selector.transform(X)


@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        "a": [0.1, 0.4, 0.5, 0.9, 0.3],
        "b": [1, 2, 1, 2, 1],
        "c": ["x", "y", "x", "x", "y"]
    })
    y = pd.Series([0, 1, 0, 1, 0])
    return X, y


def test_init_valid(sample_data):
    clf = DecisionTreeClassifier()
    selector = FuzzyImplicationGranularityFeatureSelection(clf, eps=0.5, d=5, sigma=10, random_state=42)
    assert selector.random_state == 42
    assert selector.eps == 0.5
    assert selector.d == 5
    assert selector.sigma == 10


@pytest.mark.parametrize("eps,d,sigma,random_state", [
    (-1, 10, 50, None),
    (0, 10, 50, None),
    (0.5, -5, 50, None),
    (0.5, 10, 200, None),
    (0.5, 10, 10, "abc"),
])
def test_init_invalid(eps, d, sigma, random_state):
    clf = DecisionTreeClassifier()
    with pytest.raises(ValueError):
        FuzzyImplicationGranularityFeatureSelection(clf, eps=eps, d=d, sigma=sigma, random_state=random_state)


def test_fit_and_transform(sample_data):
    X, y = sample_data
    clf = DecisionTreeClassifier(random_state=42)
    selector = FuzzyImplicationGranularityFeatureSelection(clf, eps=0.5, d=3, random_state=42)
    selector.fit(X, y)
    transformed = selector.transform(X)
    assert isinstance(transformed, pd.DataFrame)
    assert all(col in X.columns for col in transformed.columns)


def test_fit_invalid_input(sample_data):
    X, y = sample_data
    clf = DecisionTreeClassifier()
    selector = FuzzyImplicationGranularityFeatureSelection(clf)
    with pytest.raises(ValueError):
        selector.fit(None, y)
    with pytest.raises(TypeError):
        selector.fit(np.array(X), y)
    with pytest.raises(ValueError):
        selector.fit(X.iloc[:0, :], y)
    with pytest.raises(ValueError):
        selector.fit(X, y.iloc[:-1])


def test_deterministic_results(sample_data):
    X, y = sample_data
    clf = DecisionTreeClassifier()
    selector1 = FuzzyImplicationGranularityFeatureSelection(clf, eps=0.5, d=3, random_state=123)
    selector2 = FuzzyImplicationGranularityFeatureSelection(clf, eps=0.5, d=3, random_state=123)

    selector1.fit(X, y)
    selector2.fit(X, y)

    assert selector1.S == selector2.S
    transformed1 = selector1.transform(X)
    transformed2 = selector2.transform(X)
    pd.testing.assert_frame_equal(transformed1, transformed2)


def test_transform_without_fit(sample_data):
    X, _ = sample_data
    clf = DecisionTreeClassifier()
    selector = FuzzyImplicationGranularityFeatureSelection(clf)
    with pytest.raises(RuntimeError):
        selector.transform(X)


def test_mixed_numerical_and_categorical():
    X = pd.DataFrame({
        "num1": [1.2, 3.4, 2.2, 4.8, 3.1],
        "num2": [10, 15, 10, 20, 15],
        "cat1": ["red", "blue", "red", "green", "blue"],
        "cat2": ["A", "B", "A", "A", "B"]
    })
    y = pd.Series([0, 1, 0, 1, 0])

    clf = DecisionTreeClassifier(random_state=42)
    selector = FuzzyImplicationGranularityFeatureSelection(clf, eps=0.5, d=3, random_state=42)
    selector.fit(X, y)

    transformed = selector.transform(X)

    assert isinstance(transformed, pd.DataFrame)
    assert set(transformed.columns).issubset(set(X.columns))

    for col in transformed.columns:
        if X[col].dtype == object:
            assert np.issubdtype(transformed[col].dtype, np.integer)
