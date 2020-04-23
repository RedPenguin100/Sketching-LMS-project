import numpy as np


def get_dummy_data():
    n = 1000000
    A = np.empty((n, 3))
    A[:, 2] = np.arange(n)
    rand = np.random.uniform(size=2 * n)
    A[:, 0] = A[:, 2] + rand[n:]
    A[:, 1] = 1
    A[:, 2] = A[:, 2] + rand[0:n]
    return A
