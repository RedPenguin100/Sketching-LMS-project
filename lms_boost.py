import numpy as np
from scipy.linalg import lstsq
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

from lms_coreset import lms_coreset


def linreg_boost(A_tag, m, k):
    (C, y) = lms_coreset(A_tag, m, k)
    return lstsq(C, y)


def ridgecv_boost(A_tag, alphas, m, k):
    (C, y) = lms_coreset(A_tag, m, k)
    return RidgeCV(alphas=alphas, cv=m).fit(C, y)


def lassocv_boost(A_tag, alphas, m, k):
    (n, d_tag) = A_tag.shape
    (C, y) = lms_coreset(A_tag, m, k)
    beta = np.sqrt((m * d_tag * d_tag + m) / n)
    return LassoCV(alphas=alphas, cv=m).fit(beta * C, beta * y)


def elasticcv_boost(A_tag, m, alphas, rho, k):
    (n, d_tag) = A_tag.shape
    (C, y) = lms_coreset(A_tag, m, k)
    beta = np.sqrt((m * d_tag * d_tag + m) / n)
    return ElasticNetCV(alphas=alphas, l1_ratio=rho, cv=m).fit(beta * C, beta * y)
