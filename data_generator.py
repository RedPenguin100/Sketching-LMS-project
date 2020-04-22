import numpy as np


def get_dummy_data():
    n = 1000000
    A = np.empty((n, 2))
    b = np.arange(n)
    A[:, 0] = b
    A[:, 1] = 1
    rand = np.random.uniform(size=2 * n)
    b = b + rand[0:n]
    A[:, 0] = A[:, 0] + rand[n:]
    b.resize(n, 1)
    return A, b
