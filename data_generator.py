import numpy as np


def get_dummy_data():
    n = 1000000
    A = np.empty((n, 2))
    b = np.arange(n)
    rand = np.random.uniform(size=2 * n)
    A[:, 0] = b + rand[n:]
    A[:, 1] = 1
    b = b + rand[0:n]
    b.resize(n, 1)
    return A, b
