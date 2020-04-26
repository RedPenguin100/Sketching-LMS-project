import numpy as np

from Source.Algorithms.caratheodory import caratheodory_matrix


def lms_coreset(A_tag, m, k):
    (n, d_tag) = A_tag.shape
    d = d_tag - 1 # A_tag is n x (d + 1)
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
    return S[:, 0:S_m - 1], S[:, S_m - 1]
