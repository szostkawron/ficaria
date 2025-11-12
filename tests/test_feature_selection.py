import pytest
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from ficaria.feature_selection import FuzzyGranularitySelector
from sklearn.utils.validation import check_is_fitted


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
        FuzzyGranularitySelector("not_a_model")

def test_deterministic_results(sample_data):
    X, y = sample_data
    clf = DecisionTreeClassifier(random_state=42)
    selector1 = FuzzyGranularitySelector(clf, eps=0.5, d=3, random_state=123)
    selector2 = FuzzyGranularitySelector(clf, eps=0.5, d=3, random_state=123)

    selector1.fit(X, y)
    selector2.fit(X, y)
    assert selector1.S == selector2.S
    transformed1 = selector1.transform(X)
    transformed2 = selector2.transform(X)
    pd.testing.assert_frame_equal(transformed1, transformed2)

def test_transform_without_fit(sample_data):
    X, y = sample_data
    clf = DecisionTreeClassifier()
    selector = FuzzyGranularitySelector(clf)

    with pytest.raises(AttributeError):
        check_is_fitted(selector, attributes=[
            "X_train_", "imputer_", "centers_", "u_", "optimal_c_", "np_rng_"
        ])


def test_init_valid(sample_data):
    clf = DecisionTreeClassifier()
    selector = FuzzyGranularitySelector(clf, eps=0.5, d=5, sigma=10, random_state=42)
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
        FuzzyGranularitySelector(clf, eps=eps, d=d, sigma=sigma, random_state=random_state)


def test_fit_and_transform(sample_data):
    X, y = sample_data
    clf = DecisionTreeClassifier(random_state=42)
    selector = FuzzyGranularitySelector(clf, eps=0.5, d=3, random_state=42)
    selector.fit(X, y)
    transformed = selector.transform(X)
    assert isinstance(transformed, pd.DataFrame)
    assert all(col in X.columns for col in transformed.columns)


def test_fit_invalid_input_types(sample_data):
    X, y = sample_data
    clf = DecisionTreeClassifier()
    selector = FuzzyGranularitySelector(clf)

    with pytest.raises(ValueError):
        selector.fit(None, y)

    with pytest.raises(ValueError):
        selector.fit(pd.DataFrame(), y)

    with pytest.raises(ValueError):
        selector.fit(X, y.iloc[:-1])

    invalid_inputs = [
        [1, 2, 3],    
        "not a dataframe", 
        123,   
        pd.Series([1, 2, 3]),  
        np.array([1, 2, 3])  
    ]
    for bad_X in invalid_inputs:
        with pytest.raises(ValueError):
            selector.fit(bad_X, y)

    arr = np.array(X)
    selector2 = FuzzyGranularitySelector(clf)
    selector2.fit(arr, y)

    ll = X.values.tolist()
    selector3 = FuzzyGranularitySelector(clf)
    selector3.fit(ll, y)


def test_deterministic_results(sample_data):
    X, y = sample_data
    clf = DecisionTreeClassifier()
    selector1 = FuzzyGranularitySelector(clf, eps=0.5, d=3, random_state=123)
    selector2 = FuzzyGranularitySelector(clf, eps=0.5, d=3, random_state=123)

    selector1.fit(X, y)
    selector2.fit(X, y)

    assert selector1.S == selector2.S
    transformed1 = selector1.transform(X)
    transformed2 = selector2.transform(X)
    pd.testing.assert_frame_equal(transformed1, transformed2)

def test_missing_values_in_X_raises(sample_data):
    X, y = sample_data
    X_nan = X.copy()
    X_nan.loc[0, "a"] = np.nan
    clf = DecisionTreeClassifier()
    selector = FuzzyGranularitySelector(clf)
    with pytest.raises(ValueError):
        selector.fit(X_nan, y)


def test_inconsistent_columns_between_fit_and_transform(sample_data):
    X, y = sample_data
    clf = DecisionTreeClassifier(random_state=0)
    selector = FuzzyGranularitySelector(clf)
    selector.fit(X, y)

    X_bad_order = X[["b", "a", "c"]].copy()
    with pytest.raises(ValueError):
        selector.transform(X_bad_order)

    X_missing = X.drop(columns=["c"])
    with pytest.raises(ValueError):
        selector.transform(X_missing)

    X_extra = X.copy()
    X_extra["extra"] = [0, 0, 0, 0, 0]
    with pytest.raises(ValueError):
        selector.transform(X_extra)

def test_unsupervised_mode_y_none(sample_data):
    X, _ = sample_data
    clf = DecisionTreeClassifier(random_state=42)
    selector = FuzzyGranularitySelector(clf, eps=0.5, d=3, random_state=42)
    selector.fit(X, y=None)
    transformed = selector.transform(X)
    assert isinstance(transformed, pd.DataFrame)
    assert transformed.shape[0] == X.shape[0]

def test_mixed_numerical_and_categorical():
    X = pd.DataFrame({
        "num1": [1.2, 3.4, 2.2, 4.8, 3.1],
        "num2": [10, 15, 10, 20, 15],
        "cat1": ["red", "blue", "red", "green", "blue"],
        "cat2": ["A", "B", "A", "A", "B"]
    })
    y = pd.Series([0, 1, 0, 1, 0])

    clf = DecisionTreeClassifier(random_state=42)
    selector = FuzzyGranularitySelector(clf, eps=0.5, d=3, random_state=42)
    selector.fit(X, y)

    transformed = selector.transform(X)

    assert isinstance(transformed, pd.DataFrame)
    assert set(transformed.columns).issubset(set(X.columns))

    for col in transformed.columns:
        if X[col].dtype == object:
            assert np.issubdtype(transformed[col].dtype, np.integer)


def test__calculate_similarity_matrix_for_df_numeric(sample_data):
    X, y = sample_data
    clf = DecisionTreeClassifier()
    selector = FuzzyGranularitySelector(clf, eps=0.5)
    selector.fit(X, y)
    col_index = 0 
    mat = selector._calculate_similarity_matrix_for_df(col_index, X)
    assert isinstance(mat, np.ndarray)
    assert mat.shape == (len(X), len(X))
    assert np.all((mat >= 0) & (mat <= 1))


def test__calculate_similarity_matrix_for_df_nominal(sample_data):
    X, y = sample_data
    clf = DecisionTreeClassifier()
    selector = FuzzyGranularitySelector(clf)
    selector.fit(X, y)
    col_index = 2 
    mat = selector._calculate_similarity_matrix_for_df(col_index, X)
    assert mat.shape == (len(X), len(X))
    assert np.all((mat == 0) | (mat == 1))


def test__calculate_delta_for_column_subset_global_and_local(sample_data):
    X, y = sample_data
    clf = DecisionTreeClassifier()
    selector = FuzzyGranularitySelector(clf)
    selector.fit(X, y)

    B = [0, 1] 
    granule_vec, size = selector._calculate_delta_for_column_subset(0, B)
    assert isinstance(granule_vec, np.ndarray)
    assert isinstance(size, float)
    assert size >= 0

    granule_vec_local, size_local = selector._calculate_delta_for_column_subset(0, B, df=X)
    assert isinstance(granule_vec_local, np.ndarray)
    assert size_local >= 0


def test__calculate_multi_granularity_fuzzy_implication_entropy_basic(sample_data):
    X, y = sample_data
    clf = DecisionTreeClassifier()
    selector = FuzzyGranularitySelector(clf)
    selector.fit(X, y)

    ent = selector._calculate_multi_granularity_fuzzy_implication_entropy([0, 1], type="basic")
    assert isinstance(ent, float)
    assert ent >= 0


@pytest.mark.parametrize("etype", ["basic", "conditional", "joint", "mutual"])
def test__calculate_multi_granularity_fuzzy_implication_entropy_types(sample_data, etype):
    X, y = sample_data
    clf = DecisionTreeClassifier()
    selector = FuzzyGranularitySelector(clf)
    selector.fit(X, y)

    val = selector._calculate_multi_granularity_fuzzy_implication_entropy([0], type=etype)
    assert isinstance(val, float)
    assert val >= 0


def test__granual_consistency_of_B_subset(sample_data):
    X, y = sample_data
    clf = DecisionTreeClassifier()
    selector = FuzzyGranularitySelector(clf)
    selector.fit(X, y)

    score = selector._granular_consistency_of_B_subset([0])
    assert isinstance(score, float)
    assert 0 <= score <= 1


def test__local_granularity_consistency_of_B_subset(sample_data):
    X, y = sample_data
    clf = DecisionTreeClassifier()
    selector = FuzzyGranularitySelector(clf)
    selector.fit(X, y)

    val = selector._local_granularity_consistency_of_B_subset([0])
    assert isinstance(val, float)
    assert 0 <= val <= 1


def test__create_partitions(sample_data):
    X, y = sample_data
    clf = DecisionTreeClassifier()
    selector = FuzzyGranularitySelector(clf)
    selector.fit(X, y)

    parts = selector._create_partitions()
    assert isinstance(parts, dict)
    assert set(parts.keys()) == set(y.unique())
    for df_part in parts.values():
        assert isinstance(df_part, pd.DataFrame)
        assert selector.target_name in df_part.columns


def test__FIGFS_algorithm_executes(sample_data):
    X, y = sample_data
    clf = DecisionTreeClassifier()
    selector = FuzzyGranularitySelector(clf, d=2)
    selector.fit(X, y)

    result = selector._FIGFS_algorithm()
    assert isinstance(result, list)
    assert all(isinstance(i, int) for i in result)


def test_d_less_than_number_of_features():
    np.random.seed(42)
    X = pd.DataFrame({
        "a": np.random.rand(10),
        "b": np.random.rand(10),
        "c": np.random.rand(10),
        "d": np.random.rand(10),
        "e": np.random.rand(10),
        "f": np.random.rand(10)
    })
    y = np.random.randint(0, 2, size=10)

    clf = DecisionTreeClassifier()
    selector = FuzzyGranularitySelector(clf, d=3)
    
    selector.fit(X, y)
    X_transformed = selector.transform(X)
    
    assert len(selector.S_opt) <= selector.d, f"Selected features ({len(selector.S_opt)}) exceed d={selector.d}"
    assert X_transformed.shape[1] == len(selector.S_opt)
    
    for col_index in selector.S_opt:
        assert col_index in range(X.shape[1])
