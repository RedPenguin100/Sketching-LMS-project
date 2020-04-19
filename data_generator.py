import numpy as np


def get_b(n):
    return np.array(range(n))


def get_dummy_data():
    n = 3000000
    A = np.empty((n, 2))
    b = get_b(n)
    A[:, 0] = b
    A[:, 1] = np.ones(n)
    b = b + np.random.uniform(size=n)
    A[:, 0] = A[:, 0] + np.random.uniform(size=n)
    b.resize(n, 1)
    return A, b
