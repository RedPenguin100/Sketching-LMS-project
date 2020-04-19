import pytest
from pytest import approx

from caratheodory import caratheodory, _calculate_weighted_mean, fast_caratheodory, caratheodory_matrix
import numpy as np

from data_generator import get_dummy_data
from lms_coreset import lms_coreset


def test_caratheodory():
    d = 2

    P = [np.array([0, 1]), np.array([0, 0]), np.array([1, 1]), np.array([1, 0]),
         np.array([-1, 0]), np.array([-1, 1]), np.array([0, -1]), np.array([1, -1])]
    u = [0.125, 0.125, 0.125, 0.125,
         0.125, 0.125, 0.125, 0.125]
    assert approx(sum(u)) == 1, "u values need to sum up to 1"
    expected_weighted_mean = _calculate_weighted_mean(P, u)

    S, w = caratheodory(P, u)

    actual_weighted_mean = 0
    for i in range(len(S)):
        actual_weighted_mean += w[i] * S[i]

    assert len(S) <= d + 1
    assert 0 == approx(np.linalg.norm(actual_weighted_mean - expected_weighted_mean))
    assert 1 == approx(sum(w))


def test_fast_caratheodory():
    d = 2

    P = [np.array([0, 1]), np.array([0, 0]), np.array([1, 1]), np.array([1, 0]),
         np.array([-1, 0]), np.array([-1, 1]), np.array([0, -1]), np.array([1, -1])]
    u = [0.125, 0.125, 0.125, 0.125,
         0.125, 0.125, 0.125, 0.125]
    assert approx(sum(u)) == 1, "u values need to sum up to 1"
    expected_weighted_mean = _calculate_weighted_mean(P, u)

    S, w = fast_caratheodory(P, u, 4)

    assert len(S) <= d + 1
    assert 1 == approx(sum(w))
    actual_weighted_mean = 0
    for i in range(len(S)):
        actual_weighted_mean += w[i] * S[i]
    assert 0 == approx(np.linalg.norm(actual_weighted_mean - expected_weighted_mean))


def test_caratheodory_matrix():
    A = np.array([[0, 1], [0, 0], [1, 1], [1, 0],
                  [-1, 0]
                  # , [-1, 1], [0, -1], [1, -1]
                  ])
    (n, d) = A.shape
    S = caratheodory_matrix(A, k=4)

    assert S.shape == (np.power(d, 2) + 1, d)
    assert np.linalg.norm(np.matmul(S.transpose(), S) - np.matmul(A.transpose(), A)) == 0


# TODO: add better test for lms_coreset
def test_lms_coreset():
    A = np.array([[0, 1], [0, 0], [1, 1], [1, 0],
                  [-1, 0], [2, 0], [3, 0], [4, 0],
                  [0, 2], [0, 3], [0, 4], [0, 5]
                  ])
    b = np.array([-1, -1, 1, 1, 0, 2, 4, 5, 1, 2, 3, 4])
    C, y = lms_coreset(A, b, m=2, k=5)


def test_lms_coreset_bad_k_value():
    A = np.array([[0, 1], [0, 0], [1, 1], [1, 0],
                  [-1, 0], [2, 0], [3, 0], [4, 0],
                  [0, 2], [0, 3], [0, 4], [0, 5]
                  ])
    b = np.array([-1, -1, 1, 1, 0, 2, 4, 5, 1, 2, 3, 4])
    with pytest.raises(ValueError):
        C, y = lms_coreset(A, b, m=2, k=2)


import sys

sys.setrecursionlimit(1000000)


def test_lms_generated_data():
    """
    Sometimes fails due to numerical instability.
    """
    A, b = get_dummy_data()
    print("Done generating")
    (n, d) = A.shape
    C, y = lms_coreset(A, b, m=1, k=100)
    x = np.linalg.lstsq(A, b)[0]
    x_fast = np.linalg.lstsq(C, y)[0]
    assert approx(x_fast[0], abs=1e-4) == x[0]
