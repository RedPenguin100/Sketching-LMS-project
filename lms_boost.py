from scipy.linalg import lstsq

from lms_coreset import lms_coreset


def linreg_boost(A_tag, m, k):
    (C, y) = lms_coreset(A_tag, m, k)
    return lstsq(C, y)[0]
