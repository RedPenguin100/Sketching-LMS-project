import numpy as np

from caratheodory import caratheodory_matrix


def lms_coreset(A, b, m, k):
    """
    TODO: write documentation
    :param A:
    :param b:
    :param m:
    :param k:
    :return:
    """
    (n, d) = A.shape
    A_tag = np.zeros((n, d + 1))
    # A' = (A | b)
    for i in range(n):
        for j in range(d):
            A_tag[i][j] = A[i][j]
    for i in range(n):
        A_tag[i][d] = b[i]
    A_block_list = np.array_split(A_tag, m)  # Each block with size at most n // m + 1
    S_list = []
    for block in A_block_list:
        S_list.append(caratheodory_matrix(block, k))

    S = np.concatenate(S_list)
    (S_n, S_m) = S.shape
    if S_n > m + m * np.power(d + 1, 2):
        raise ValueError("In lms_coreset: rows are larger than expected")
    if S_m > d + 1:
        raise ValueError("In lms_coreset: columns are larger than expected.")
    C = S[:, 0:S_m - 1]
    y = S[:, S_m - 1]
    return C, y
