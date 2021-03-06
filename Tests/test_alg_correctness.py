import pytest
import sys
import numpy as np
import scipy

from pytest import approx
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

from Source.Algorithms.caratheodory import caratheodory_alg, fast_caratheodory, \
    caratheodory_matrix, \
    get_optimal_k_value
from Source.Utils.data_generator import get_easy_data, get_varying_d_data
from Source.Utils.dataset_handler import get_dataset, SECOND_DATASET_PATH, THIRD_DATASET_PATH, FIRST_DATASET_PATH
from Source.Algorithms.lms_boost import linreg_boost, ridgecv_boost, lassocv_boost, elasticcv_boost
from Source.Algorithms.lms_coreset import lms_coreset

sys.setrecursionlimit(1000000)


def _calculate_weighted_mean(P: list, u):
    weight = 0.
    for i in range(len(P)):
        weight += u[i] * P[i]
    return weight


def test_caratheodory():
    d = 2

    P = [np.array([0, 1]), np.array([0, 0]), np.array([1, 1]), np.array([1, 0]),
         np.array([-1, 0]), np.array([-1, 1]), np.array([0, -1]), np.array([1, -1])]
    u = [0.125, 0.125, 0.125, 0.125,
         0.125, 0.125, 0.125, 0.125]
    assert approx(sum(u)) == 1, "u values need to sum up to 1"
    expected_weighted_mean = _calculate_weighted_mean(P, u)

    S, w, _ = caratheodory_alg(P, u, len(P), len(P[0]))

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

    S, w, _ = fast_caratheodory(P, u, 4)

    assert len(S) <= d + 1
    assert 1 == approx(sum(w))
    actual_weighted_mean = 0
    for i in range(len(S)):
        actual_weighted_mean += w[i] * S[i]
    assert 0 == approx(np.linalg.norm(actual_weighted_mean - expected_weighted_mean))


def test_caratheodory_matrix():
    A = np.array([[0, 1], [0, 0], [1, 1], [1, 0],
                  [-1, 0]
                  ])
    (n, d) = A.shape
    S = caratheodory_matrix(A, k=4)

    assert S.shape == (np.power(d, 2) + 1, d)
    assert np.linalg.norm(np.matmul(S.transpose(), S) - np.matmul(A.transpose(), A)) == 0


def test_lms_coreset():
    A = np.array([[0, 1], [0, 0], [1, 1], [1, 0],
                  [-1, 0], [2, 0], [3, 0], [4, 0],
                  [0, 2], [0, 3], [0, 4], [0, 5]
                  ])
    b = np.array([-1, -1, 1, 1, 0, 2, 4, 5, 1, 2, 3, 4])
    C, y = lms_coreset(np.concatenate((A, b.reshape(len(b), 1)), axis=1), m=2, k=5)


def test_lms_coreset_bad_k_value():
    A = np.array([[0, 1], [0, 0], [1, 1], [1, 0],
                  [-1, 0], [2, 0], [3, 0], [4, 0],
                  [0, 2], [0, 3], [0, 4], [0, 5]
                  ])
    b = np.array([-1, -1, 1, 1, 0, 2, 4, 5, 1, 2, 3, 4])
    with pytest.raises(ValueError):
        C, y = lms_coreset(np.concatenate((A, b)), m=2, k=2)


def test_linreg_boost_correctness():
    d = 2
    A_tag = get_easy_data()
    x_fast = linreg_boost(A_tag, m=1, k=100)[0]
    A, b = A_tag[:, 0:d], A_tag[:, d]
    x = scipy.linalg.lstsq(A, b)[0]
    assert approx(x_fast) == x


def test_ridgecv_boost_correctness():
    d = 2
    A_tag = get_easy_data()
    alphas = [0.1, 1, 10]
    m = 10
    res = ridgecv_boost(A_tag, alphas, m=m, k=100)
    A, b = A_tag[:, 0:d], A_tag[:, d]
    res2 = RidgeCV(alphas=alphas, cv=m).fit(A, b)
    assert approx(res.coef_) == res2.coef_


def test_lassocv_boost_correctness():
    d = 2
    A_tag = get_easy_data()
    alphas = (0.1, 1, 10, 100, 1000)
    m = 10
    res = lassocv_boost(A_tag, alphas, m=m, k=100)
    A, b = A_tag[:, 0:d], A_tag[:, d]
    res2 = LassoCV(alphas=alphas, cv=m).fit(A, b)
    assert approx(res.coef_, abs=1e-4) == res2.coef_


def test_elasticcv_boost_correctness():
    d = 2
    A_tag = get_easy_data()
    alphas = (0.1, 1, 10, 100, 1000)
    m = 10
    rho = 0.5
    res = elasticcv_boost(A_tag, m=m, alphas=alphas, rho=rho, k=100)
    A, b = A_tag[:, 0:d], A_tag[:, d]
    res2 = ElasticNetCV(alphas=alphas, cv=m, l1_ratio=rho).fit(A, b)
    assert approx(res.coef_, abs=1e-4) == res2.coef_


@pytest.mark.parametrize('dataset', [FIRST_DATASET_PATH, SECOND_DATASET_PATH, THIRD_DATASET_PATH])
def test_linreg_datasets_accuracy(dataset):
    A_tag = get_dataset(dataset)
    (n, d_tag) = A_tag.shape
    d = d_tag - 1
    k = get_optimal_k_value(d)
    x_fast = linreg_boost(A_tag, m=1, k=k)[0]
    x = scipy.linalg.lstsq(A_tag[:, 0:d], A_tag[:, d])[0]
    err = np.linalg.norm(x - x_fast)
    print(err)
    print(err / np.linalg.norm(x))


@pytest.mark.parametrize('dataset', [FIRST_DATASET_PATH, SECOND_DATASET_PATH, THIRD_DATASET_PATH])
def test_ridge_datasets_accuracy(dataset):
    A_tag = get_dataset(dataset)
    (n, d_tag) = A_tag.shape
    alphas = (0.1, 1, 10, 100, 1000)
    d = d_tag - 1
    m = 64
    k = get_optimal_k_value(d)
    result_fast = ridgecv_boost(A_tag, alphas=alphas, m=m, k=k)
    result = RidgeCV(alphas=alphas, cv=m).fit(A_tag[:, 0:d], A_tag[:, d])
    err = np.linalg.norm(result_fast.coef_ - result.coef_)
    print(err)
    print(err / np.linalg.norm(result.coef_))


@pytest.mark.parametrize('dataset', [FIRST_DATASET_PATH, SECOND_DATASET_PATH, THIRD_DATASET_PATH])
def test_lasso_datasets_accuracy(dataset):
    A_tag = get_dataset(dataset)
    (n, d_tag) = A_tag.shape
    alphas = (0.1, 1, 10, 100, 1000)
    d = d_tag - 1
    m = 64
    k = get_optimal_k_value(d)
    result_fast = lassocv_boost(A_tag, alphas=alphas, m=m, k=k)
    result = LassoCV(alphas=alphas, cv=m).fit(A_tag[:, 0:d], A_tag[:, d])
    err = np.linalg.norm(result_fast.coef_ - result.coef_)
    print(err)
    print(err / np.linalg.norm(result.coef_))


@pytest.mark.parametrize('dataset', [FIRST_DATASET_PATH, SECOND_DATASET_PATH, THIRD_DATASET_PATH])
def test_elastic_datasets_accuracy(dataset):
    A_tag = get_dataset(dataset)
    (n, d_tag) = A_tag.shape
    alphas = (0.1, 1, 10, 100, 1000)
    d = d_tag - 1
    m = 2
    k = get_optimal_k_value(d)
    rho = 0.5
    result_fast = elasticcv_boost(A_tag, alphas=alphas, m=m, k=k, rho=rho)
    result = ElasticNetCV(alphas=alphas, cv=m, l1_ratio=rho).fit(A_tag[:, 0:d], A_tag[:, d])
    err = np.linalg.norm(result_fast.coef_ - result.coef_)
    print(err)
    print(err / np.linalg.norm(result.coef_))
