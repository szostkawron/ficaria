import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import euclidean

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ficaria.utils import split_complete_incomplete, euclidean_distance, fuzzy_c_means


# ----- split_complete_incomplete ---------------------------------------

def test_split_complete_incomplete_basic():
    X = pd.DataFrame({
        "a": [1, 2, np.nan, 4],
        "b": [5, 6, 7, np.nan],
    })

    complete, incomplete = split_complete_incomplete(X)

    assert not complete.isna().any().any()
    assert incomplete.isna().any(axis=1).any()
    pd.testing.assert_frame_equal(pd.concat([complete, incomplete]).sort_index(), X.sort_index())


def test_split_complete_incomplete_all_complete():
    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    complete, incomplete = split_complete_incomplete(X)
    assert len(incomplete) == 0
    pd.testing.assert_frame_equal(complete, X)


def test_split_complete_incomplete_all_missing():
    X = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
    complete, incomplete = split_complete_incomplete(X)
    assert len(complete) == 0
    assert len(incomplete) == len(X)


# ----- euclidean_distance -----------------------------------------------

vector_pairs = [
    (np.array([1.0, 2.0, 3.0]), np.array([4.0, 6.0, 8.0])),
    (np.array([1.0, np.nan, 3.0, 4.0]), np.array([1.0, 2.0, 3.0, 5.0])),
    (np.array([np.nan, 2.0, 3.0, np.nan, 5.0]), np.array([1.0, 2.0, 4.0, 8.0, 5.0])),
]

@pytest.mark.parametrize("a, b", vector_pairs)
def test_euclidean_distance_matches_scipy(a, b):
    mask = ~np.isnan(a) & ~np.isnan(b)
    expected = euclidean(a[mask], b[mask])
    result = euclidean_distance(a, b)
    assert np.isclose(result, expected, atol=1e-12), f"Mismatch for vectors {a}, {b}"


# ----- fuzzy_c_means -----------------------------------------------

dataframes_list = [
    pd.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "b": [5.0, 4.0, 3.0],
    }),

        pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [5.0, 4.0, 3.0, 2.0, 1.0],
    }),

    pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 17.0, 5.0, 6.0, 7.0],
        "b": [5.0, 4.0, 3.0, 20.0, 2.0, 1.0, 0.0],
        "c": [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0],
        "d": [10.0, 12.5, 15.0, 16.0, 17.0, 18.0, 19.0],
        "e": [9.0, 8.7, 7.0, 6.0, 5.0, 2.2, 2.1],
    }),
]

@pytest.mark.parametrize("X", dataframes_list)
def test_fuzzy_c_means_output_shapes(X):
    centers, memberships = fuzzy_c_means(X, n_clusters=2, m=2.0, max_iter=50, tol=1e-4, random_state=42)
    
    assert centers.shape == (2, X.shape[1])
    assert memberships.shape == (X.shape[0], 2)
    np.testing.assert_allclose(np.sum(memberships, axis=1), 1.0, atol=1e-5)


def test_fuzzy_c_means_converges_reasonably():
    X = np.vstack([
        np.random.normal(0, 0.1, (5, 2)),
        np.random.normal(5, 0.1, (5, 2))
    ])
    centers, memberships = fuzzy_c_means(X, n_clusters=2, random_state=0)
    assert np.linalg.norm(centers[0] - centers[1]) > 1.0


def test_fuzzy_c_means_same_random_state_reproducible():
    X = np.array([[1, 2], [3, 4], [5, 6], [8, 9]])
    n_clusters = 3

    centers_1, memberships_1 = fuzzy_c_means(
        X, n_clusters=n_clusters, m=2.0, max_iter=50, tol=1e-4, random_state=42
    )
    centers_2, memberships_2 = fuzzy_c_means(
        X, n_clusters=n_clusters, m=2.0, max_iter=50, tol=1e-4, random_state=42
    )

    np.testing.assert_allclose(centers_1, centers_2, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(memberships_1, memberships_2, rtol=1e-8, atol=1e-8)
