import numpy as np


def dtw(v1, v2):
    
    length_v1 = len(v1)
    length_v2 = len(v2)

    mat = np.zeros((length_v1 + 1, length_v2 + 1))

    mat[0, 0] = abs(v1[0] - v2[0])

    for i in range(1, length_v1 - 1):
        v20 = v2[0]
        mat[i, 0] = abs(v1[i] - v20) + mat[i-1, 0]

    for j in range(1, length_v2 - 1):
        v10 = v1[0]
        mat[0, j] = abs(v10 - v2[j]) + mat[0, j-1]

    for i in range(1, length_v1-1):
        for j in range(1, length_v2-1):
            mat[i, j] = abs(v1[i] - v2[j]) + min(mat[i-1, j], mat[i, j-1], mat[i-1, j-1])

    return mat[length_v1 - 1, length_v2 - 1]
