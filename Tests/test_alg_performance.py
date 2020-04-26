import numpy as np
import pytest
import scipy

from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

from Source.Algorithms.caratheodory import get_optimal_k_value
from Source.Algorithms.lms_boost import ridgecv_boost, lassocv_boost, elasticcv_boost, linreg_boost
from Source.Utils.data_generator import get_varying_d_data
from Source.Utils.dataset_handler import SECOND_DATASET_PATH, FIRST_DATASET_PATH, get_dataset
from Source.Utils.measurement import measure_method

DATA_SIZES = np.arange(start=1000, stop=2300000, step=100000)

RHO = 0.5


@pytest.mark.parametrize('n', DATA_SIZES)
@pytest.mark.parametrize('d', [3, 5, 7])
def test_size_time_for_various_d_values(n, d):
    m = 3
    alphas = np.ones(100)
    A_tag = get_varying_d_data(d=d, n=n)

    ridgecv_mes = measure_method(RidgeCV(alphas=alphas, cv=m).fit, A_tag[:, :d], A_tag[:, d])
    ridge_boost_mes = measure_method(ridgecv_boost, A_tag, alphas, m, get_optimal_k_value(d))

    lassocv_mes = measure_method(LassoCV(alphas=alphas, cv=m).fit, A_tag[:, :d], A_tag[:, d])
    lasso_boost_mes = measure_method(lassocv_boost, A_tag, alphas, m, get_optimal_k_value(d))

    elasticcv_mes = measure_method(ElasticNetCV(alphas=alphas, cv=m, l1_ratio=RHO).fit, A_tag[:, :d], A_tag[:, d])
    elastic_boost_mes = measure_method(elasticcv_boost, A_tag, m, alphas, RHO, get_optimal_k_value(d))

    with open('FIRST_PERFORMANCE_TEST_RESULTS.txt', 'a+') as f:
        f.write("ridge,{n},{d},{duration}\n".format(n=n, d=d, duration=ridgecv_mes.duration))
        f.write("ridge_boost,{n},{d},{duration}\n".format(n=n, d=d, duration=ridge_boost_mes.duration))
        f.write("lasso,{n},{d},{duration}\n".format(n=n, d=d, duration=lassocv_mes.duration))
        f.write("lasso_boost,{n},{d},{duration}\n".format(n=n, d=d, duration=lasso_boost_mes.duration))
        f.write("elastic,{n},{d},{duration}\n".format(n=n, d=d, duration=elasticcv_mes.duration))
        f.write("elastic_boost,{n},{d},{duration}\n".format(n=n, d=d, duration=elastic_boost_mes.duration))


@pytest.mark.parametrize('n', DATA_SIZES)
@pytest.mark.parametrize('alpha_count', [50, 100, 200, 300])
def test_size_time_for_various_alphas(n, alpha_count):
    d = 7
    m = 3
    alphas = np.ones(alpha_count)
    A_tag = get_varying_d_data(d=d, n=n)

    ridgecv_mes = measure_method(RidgeCV(alphas=alphas, cv=m).fit, A_tag[:, :d], A_tag[:, d])
    ridge_boost_mes = measure_method(ridgecv_boost, A_tag, alphas, m, get_optimal_k_value(d))

    lassocv_mes = measure_method(LassoCV(alphas=alphas, cv=m).fit, A_tag[:, :d], A_tag[:, d])
    lasso_boost_mes = measure_method(lassocv_boost, A_tag, alphas, m, get_optimal_k_value(d))

    elasticcv_mes = measure_method(ElasticNetCV(alphas=alphas, cv=m, l1_ratio=RHO).fit, A_tag[:, :d], A_tag[:, d])
    elastic_boost_mes = measure_method(elasticcv_boost, A_tag, m, alphas, RHO, get_optimal_k_value(d))

    with open('SECOND_PERFORMANCE_TEST_RESULTS.txt', 'a+') as f:
        f.write(
            "ridge,{n},{alpha_count},{duration}\n".format(n=n, alpha_count=alpha_count, duration=ridgecv_mes.duration))
        f.write("ridge_boost,{n},{alpha_count},{duration}\n".format(n=n, alpha_count=alpha_count,
                                                                    duration=ridge_boost_mes.duration))
        f.write(
            "lasso,{n},{alpha_count},{duration}\n".format(n=n, alpha_count=alpha_count, duration=lassocv_mes.duration))
        f.write("lasso_boost,{n},{alpha_count},{duration}\n".format(n=n, alpha_count=alpha_count,
                                                                    duration=lasso_boost_mes.duration))
        f.write("elastic,{n},{alpha_count},{duration}\n".format(n=n, alpha_count=alpha_count,
                                                                duration=elasticcv_mes.duration))
        f.write("elastic_boost,{n},{alpha_count},{duration}\n".format(n=n, alpha_count=alpha_count,
                                                                      duration=elastic_boost_mes.duration))


@pytest.mark.parametrize('dataset, dataset_name', [(FIRST_DATASET_PATH, '3D_spatial_network'),
                                                   (SECOND_DATASET_PATH, 'household_power_consumption')])
@pytest.mark.parametrize('alpha_count', np.arange(start=20, stop=200, step=20))
def test_time_for_increasing_alphas(dataset, dataset_name, alpha_count):
    m = 3
    alphas = np.ones(alpha_count)
    A_tag = get_dataset(dataset)
    n, d_tag = A_tag.shape
    d = d_tag - 1
    ridgecv_mes = measure_method(RidgeCV(alphas=alphas, cv=m).fit, A_tag[:, :d], A_tag[:, d])
    ridge_boost_mes = measure_method(ridgecv_boost, A_tag, alphas, m, get_optimal_k_value(d))

    lassocv_mes = measure_method(LassoCV(alphas=alphas, cv=m).fit, A_tag[:, :d], A_tag[:, d])
    lasso_boost_mes = measure_method(lassocv_boost, A_tag, alphas, m, get_optimal_k_value(d))

    elasticcv_mes = measure_method(ElasticNetCV(alphas=alphas, cv=m, l1_ratio=RHO).fit, A_tag[:, :d], A_tag[:, d])
    elastic_boost_mes = measure_method(elasticcv_boost, A_tag, m, alphas, RHO, get_optimal_k_value(d))

    with open('THIRD_PERFORMANCE_TEST_RESULTS.txt', 'a+') as f:
        f.write(
            "ridge,{dataset_name},{alpha_count},{duration}\n".format(dataset_name=dataset_name, alpha_count=alpha_count,
                                                                     duration=ridgecv_mes.duration))
        f.write("ridge_boost,{dataset_name},{alpha_count},{duration}\n".format(dataset_name=dataset_name,
                                                                               alpha_count=alpha_count,
                                                                               duration=ridge_boost_mes.duration))
        f.write(
            "lasso,{dataset_name},{alpha_count},{duration}\n".format(dataset_name=dataset_name, alpha_count=alpha_count,
                                                                     duration=lassocv_mes.duration))
        f.write("lasso_boost,{dataset_name},{alpha_count},{duration}\n".format(dataset_name=dataset_name,
                                                                               alpha_count=alpha_count,
                                                                               duration=lasso_boost_mes.duration))
        f.write("elastic,{dataset_name},{alpha_count},{duration}\n".format(dataset_name=dataset_name,
                                                                           alpha_count=alpha_count,
                                                                           duration=elasticcv_mes.duration))
        f.write("elastic_boost,{dataset_name},{alpha_count},{duration}\n".format(dataset_name=dataset_name,
                                                                                 alpha_count=alpha_count,
                                                                                 duration=elastic_boost_mes.duration))

@pytest.mark.parametrize('n', np.arange(start=1000000, stop=10000000, step=2500000))
@pytest.mark.parametrize('d', [15])
def test_linreg_various_values(n, d):
    m = 64
    A_tag = get_varying_d_data(d, n)
    n, d_tag = A_tag.shape
    d = d_tag - 1
    linreg_mes = measure_method(scipy.linalg.lstsq, A_tag[:, :d], A_tag[:, d])
    linreg_boost_mes = measure_method(linreg_boost, A_tag, m, get_optimal_k_value(d))

    with open('LINREG_PERFORMANCE_TEST_D_VALUES_very_large.txt', 'a+') as f:
        f.write("linreg,{n},{d},{duration}\n".format(n=n, d=d, duration=linreg_mes.duration))
        f.write("linreg_boost,{n},{d},{duration}\n".format(n=n, d=d, duration=linreg_boost_mes.duration))
