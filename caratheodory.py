import numpy as np
from scipy.linalg import null_space


def _calculate_weighted_mean(P : list, u):
    weight = 0.
    for i in range(len(P)):
        weight += u[i] * P[i]
    return weight


def caratheodory(P: list, u):
    """
    Returns a smaller weighted set as described in caratheodory theorem.
    :param P: list of n points in R^d
    :param u: a list that represents a function that maps each point in P to a positive real value.
              will contain the correct value of the ith point in P, in the ith place.
    :return: A caratheodory set (S,w)
    :note: Running time: O(n^2d^2)
    """
    n = len(P)
    if n == 0:
        raise ValueError("Error: P cannot be empty")
    d = len(P[0])
    if n <= d + 1:
        return P, u
    A_mat = np.zeros((d, n - 1))
    for i in range(n - 1):
        A_mat[:, i] = P[i + 1] - P[0]  # I update A and calculate a_i at the same time, for efficiency.

    # Find v
    almost_v = null_space(A_mat)[:, 0]  # Get the first vector from the null_space.
    v = np.zeros(n)
    v[0] = -sum(almost_v)
    v[1:] = almost_v

    alpha = np.inf
    for i in range(n):
        if v[i] <= 0:
            continue
        alpha = min(alpha, u[i] / v[i])
    w = []
    for i in range(n):
        w.append(u[i] - alpha * v[i])

    S = []
    for i in range(n):
        if w[i] > 0:
            S.append(P[i])
    w_zeros = w.count(0.)
    for j in range(w_zeros):
        w.remove(0.)
    if len(S) > d + 1:
        return caratheodory(S, w)
    return S, w


def fast_cartheodory(P, u, k):
    """
    Performs caratheodory theorem (naively) on the input
    :param P: list of n points in R^d
    :param u: a list that represents a function that maps each point in P to a positive real value.
              will contain the correct value of the ith point in P, in the ith place.
    :return: A caratheodory set (S,w)
    :note: Running time: O(n^2d^2)
    """
    n = len(P)
    if n == 0:
        raise ValueError("Error: P cannot be empty")
    d = len(P[0])
    if n <= d + 1:
        return P, u
    if k > n or k < 1:
        raise ValueError("Error: k needs to be between 1 and n")

    partitions = np.array_split(P, n // k)
    partitions_weights = []
    for i in range(k):
        partitions_weights.append()