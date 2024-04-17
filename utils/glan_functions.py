import numpy as np


def sinkhorn_v1_np(mat):
    for _ in range(5):
        row_sum = np.expand_dims(np.sum(mat, axis=1), axis=1)
        mat = mat / row_sum

        col_sum = np.expand_dims(np.sum(mat, axis=0), axis=0)
        mat = mat / col_sum

    return mat


def check_data(file):

    data = np.load(file)
    height, length = data.shape
    if np.sum(data) != min(height, length):
        return False
    else:
        return True


def greedy_map(pred_matrix):
    height, length = pred_matrix.shape
    result = np.zeros_like(pred_matrix)
    for hh in range(height):
        row, col = np.unravel_index(np.argmax(pred_matrix), pred_matrix.shape)
        result[row, col] = 1
        pred_matrix[row, :] = 0
        pred_matrix[:, col] = 0

    return result
