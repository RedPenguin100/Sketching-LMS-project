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


def _calc_alpha(u, v, n):
    alpha = np.inf
    for i in range(n):
        if v[i] <= 0:
            continue
        alpha = min(alpha, u[i] / v[i])
    return alpha


def _calc_w(alpha, n, u, v):
    w = []
    for i in range(n):
        w.append(u[i] - alpha * v[i])
    return w


def _calc_S(n, w, P):
    S = []
    for i in range(n):
        if w[i] > 0:
            S.append(P[i])
    return S


def _remove_w_zeros(w):
    w_zeros = w.count(0.)
    for j in range(w_zeros):
        w.remove(0.)


def _calc_v(almost_v, n):
    v = np.empty(n)
    v[0] = -sum(almost_v)
    v[1:] = almost_v
    return v


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
    # Find v
    almost_v = null_space((P[1:] - P[0]).T)[:, 0]  # Get the first vector from the null_space.
    v = _calc_v(almost_v, n)
    alpha = _calc_alpha(u, v, n)
    w = _calc_w(alpha, n, u, v)
    S = _calc_S(n, w, P)
    _remove_w_zeros(w)
    if len(S) > d + 1:
        return caratheodory(S, w)
    return S, w


def get_mus_utag(P_partitions, u_partitions):
    mus = []
    u_partitions = np.array(u_partitions)
    if len(u_partitions.shape) == 2:
        u_tag = np.apply_along_axis(np.sum, -1, np.array(u_partitions))
    elif len(u_partitions.shape) == 1:
        v = np.vectorize(np.sum)
        u_tag = v(u_partitions)
    else:
        raise ValueError("Shape cannot be != 2/1")
    for i, (p_partition, u_partition) in enumerate(zip(P_partitions, u_partitions)):
        new_point = np.sum(np.multiply(p_partition, u_partition.reshape(len(u_partition), 1)).T, axis=1)
        mus.append(new_point / u_tag[i])
    return mus, u_tag


def fast_caratheodory(P, u, k):
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

    def conversion_from_list_to_np_array(P, u):
        if isinstance(u, list):
            u = np.array(u)
        if isinstance(P, list):
            P = np.array(P)
        return P, u

    P, u = conversion_from_list_to_np_array(P, u)
    if n == 0:
        raise ValueError("Error: P cannot be empty")
    d = len(P[0])
    if n <= d + 1:
        return P, u
    if k > n:  # TODO: decide on the best way to fix this.
        k = n
    if k < 1:
        raise ValueError()
    if k == n:
        return caratheodory(P, u)  # Exactly the same in that case.
    partition_indices = np.array_split(np.arange(n), k)
    p_partition = np.array_split(P, k)
    u_partition = np.array_split(u, k)
    mus, u_tag = get_mus_utag(p_partition, u_partition)
    (mu_tilde, w_tilde) = caratheodory(mus, u_tag)

    C = []
    w = []
    for mu_tilde_i, mu in enumerate(mu_tilde):
        mu_i = np.where((mus == mu).all(axis=1))[0][0]
        weight_denominator = np.sum(u[partition_indices[mu_i]])
        w.append(np.multiply(w_tilde[mu_tilde_i], u[partition_indices[mu_i]]) / weight_denominator)
        C.append(P[partition_indices[mu_i]])
    C, w = np.concatenate(C), np.concatenate(w)
    if len(C) == len(P):
        raise RecursionError("Heading to infinite recursion")
    return fast_caratheodory(C, w, k)


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
    u = np.ones(n) * (1 / n)
    (C, w) = fast_caratheodory(P, u, k)
    S = np.empty((np.power(d, 2) + 1, d))
    minimum = np.min([np.power(d, 2) + 1, n])
    if minimum == n:
        print("d^2 + 1 is not smaller than n")

    # Note: PERFORMANCE
    for i in range(minimum):
        p = C[i]

        def line1():
            index_in_P = np.where((P == p).all(axis=1))[0]
            return index_in_P[0]

        index_in_P = line1()

        def line2():
            S[i] = np.sqrt(n * w[i]) * A[index_in_P]

        line2()
    return S
