import numpy as np
from scipy.linalg import null_space


def _fast_all(p1, p2):
    for i in range(len(p1)):
        if p1[i] != p2[i]:
            return False
    return True


def index_of_point(l, point, prev_index=0):
    for i in range(prev_index, len(l)):
        if _fast_all(l[i], point):
            return i
    raise ValueError("Cannot find {} inside {}".format(point, l))


def _calculate_weighted_mean(P: list, u):
    weight = 0.
    for i in range(len(P)):
        weight += u[i] * P[i]
    return weight


def caratheodory_alg(P, u, n, d, indexes=None):
    """
    Returns a smaller weighted set as described in caratheodory theorem.
    :param P: list of n points in R^d
    :param u: a list that represents a function that maps each point in P to a positive real value.
              will contain the correct value of the ith point in P, in the ith place.
    :return: A caratheodory set (S,w)
    :note: Running time: O(n^2d^2)
    """
    if indexes is None:
        indexes = np.arange(n)
    if n == 0:
        raise ValueError("Error: P cannot be empty")
    if n <= d + 1:
        return P, u, indexes

    S = np.array(P)
    w = np.array(u)
    while n > d + 1:
        # Find v
        almost_v = np.zeros(n - 1)
        mat = (S[1:d + 2] - S[0]).T
        almost_v[0:d + 1] = null_space(mat)[:, 0]  # Get the first vector from the null_space.
        v = np.insert(np.array([-np.sum(almost_v)]), 1, almost_v)

        v_cond = v > 0.
        alpha = np.min(w[v_cond] / v[v_cond])
        w = w[0:n] - alpha * v

        cond = w > 0
        S, indexes = S[cond], indexes[cond]
        w = w[cond]
        n = len(S)
    return S, w, indexes


def get_mus_utag(P_partitions, u_partitions):
    mus = []
    u_partitions = np.array(u_partitions)
    u_shape = u_partitions.shape
    if len(u_shape) == 2:
        u_tag = np.apply_along_axis(np.sum, -1, u_partitions)
    elif len(u_shape) == 1:
        v = np.vectorize(np.sum)
        u_tag = v(u_partitions)
    else:
        raise ValueError("Shape cannot be != 2/1")
    # performance: np.matmul
    for i in np.arange(u_shape[0]):
        new_point = np.matmul(P_partitions[i].T, u_partitions[i])
        mus.append(new_point / u_tag[i])
    return mus, u_tag


def fast_caratheodory(P, u, k, indexes=None):
    """
    Returns a smaller weighted set as described in caratheodory theorem,
    using a fast algorithm O(nd).
    :param P: list of n points in R^d
    :param u: a list that represents a function that maps each point in P to a positive real value.
              will contain the correct value of the ith point in P, in the ith place.
    :param k: an integer between 1 and n for accuracy / speed trade-off.
    :return: A caratheodory set (S,w)
    :note: we assume the points are unique
    """
    n = len(P)
    if indexes is None:
        indexes = np.arange(n)

    if isinstance(u, list):
        u = np.array(u)
    if isinstance(P, list):
        P = np.array(P)

    if n == 0:
        raise ValueError("Error: P cannot be empty")
    d = len(P[0])
    if n <= d + 1:
        return P, u, indexes
    if k > n:  # TODO: decide on the best way to fix this.
        k = n
    if k < 1:
        raise ValueError()
    if k == n:
        return caratheodory_alg(P, u, n, d, indexes)  # Exactly the same in that case.

    parted_saved_indexes = np.array_split(indexes, k)
    p_partition = np.array_split(P, k)
    u_partition = np.array_split(u, k)
    mus, u_tag = get_mus_utag(p_partition, u_partition)
    mu_indexes = np.arange(len(mus))
    (mu_tilde, w_tilde, mu_indexes) = caratheodory_alg(mus, u_tag, len(mus), len(mus[0]), indexes=mu_indexes)

    def get_c_w():
        w = []
        for mu_tilde_i in range(len(mu_tilde)):
            mu_i = mu_indexes[mu_tilde_i]
            w.append((w_tilde[mu_tilde_i] * u_partition[mu_i]) / u_tag[mu_i])
        C = [p_partition[index] for index in mu_indexes]
        saved_indexes = np.concatenate(np.array(parted_saved_indexes)[mu_indexes])
        C, w = np.concatenate(C), np.concatenate(w)
        return C, w, saved_indexes

    C, w, saved_indexes = get_c_w()
    if len(C) == len(P):
        raise RecursionError("Heading to infinite recursion")
    return fast_caratheodory(C, w, k, indexes=np.array(saved_indexes))


def caratheodory_matrix(A, k):
    """
    Performs algorithm 3 as described in the article, receiving data
    and returning a manipulated "compressed" data.
    :param A: matrix of dimensions n * d
    :param k: accuracy / speed trade-off, integer from 1 to n
    :return: A matrix S of dimensions (d ^ 2 + 1) * d and
             A^t A == S^t S
    # TODO: implement / handle case where d^2+1 > n
    """
    if len(A.shape) != 2:
        raise ValueError("Expected A to be matrix, got A with shape: {}".format(A.shape))
    (n, d) = A.shape
    if k < d + 2:
        raise ValueError("k should be larger or equal to d + 2")
    P = np.matmul(A.reshape(n, d, 1), A.reshape(n, 1, d)).reshape((n, np.power(d, 2)))
    u = np.full(n, 1 / n)

    (C, w, indexes) = fast_caratheodory(P, u, k)
    S = np.empty((np.power(d, 2) + 1, d))
    minimum = np.min([np.power(d, 2) + 1, n])
    if minimum == n:
        print("d^2 + 1 is not smaller than n")

    for i in range(minimum):
        S[i] = np.sqrt(n * w[i]) * A[indexes[i]]
    return S
