import numpy as np


def get_line_approx_data(n=1000000):
    """
    Function creates data consists of n points in 2 dimensions,
    which are located nearby the line y = x.
    """
    A = np.empty((n, 3))
    A[:, 2] = np.arange(n)
    rand = np.random.uniform(size=2 * n)
    A[:, 0] = A[:, 2] + rand[n:]
    A[:, 1] = 1
    A[:, 2] = A[:, 2] + rand[0:n]
    return A


def get_easy_data(n=1000000):
    """
    Function creates fairly trivial data for minimal least squares problems, the solution should be
    (2, 5).
    We assume duplicates do not occur.
    """
    A = np.empty((n, 3))
    A[:, 0] = np.random.uniform(size=n, low=0, high=1000)
    A[:, 1] = np.random.uniform(size=n, low=0, high=1000)
    A[:, 2] = 2 * A[:, 0] + 5 * A[:, 1]
    return A


def get_varying_d_data(d, n=1000000):
    """
    Function creates data for n,d shapes and adds a little bit
    of noise to the solution which we set to be np.arange(d).
    """
    A = np.empty((n, d + 1))
    rand = np.random.uniform(size=(n, d))
    A[:, :d] = rand
    x = np.arange(d)
    A[:, d] = np.matmul(A[:, :d], x) + np.random.uniform(size=n)
    return A
