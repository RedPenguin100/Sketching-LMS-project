import pytest
from pytest import approx

from caratheodory import caratheodory, _calculate_weighted_mean
import numpy as np


def test_caratheodory():
    d = 2

    P = [np.array([0, 1]), np.array([0, 0]), np.array([1, 1]), np.array([1, 0]),
         np.array([-1, 0]), np.array([-1, 1]), np.array([0, -1]), np.array([1, -1])]
    u = [0.125, 0.125, 0.125, 0.125,
         0.125, 0.125, 0.125, 0.125]
    assert approx(sum(u)) == 1, "u values need to sum up to 1"
    expected_weighted_mean = _calculate_weighted_mean(P, u)

    S, w = caratheodory(P, u)

    assert len(S) <= d + 1
    actual_weighted_mean = 0
    for i in range(len(S)):
        actual_weighted_mean += w[i] * S[i]
    assert 0 == approx(np.linalg.norm(actual_weighted_mean-expected_weighted_mean))
    assert 1 == approx(sum(w))
