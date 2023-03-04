import numpy as np


# add column of ones to left of examples
def get_padded_examples(training_examples):
    num_ex = len(training_examples)
    padding = np.ones(shape=(num_ex, 1))
    padded_mat = np.append(padding, training_examples, axis=1)
    return padded_mat


def get_prob_matrix(samples, weights):
    k = len(weights)
    m = len(samples)
    x_transpose = samples.transpose()

    # WX is k x m matrix
    p_mat = np.matmul(weights, x_transpose)

    # TODO: test setting to ones after exponentiation / other options
    # avoid overflow (only do rows 1 to k-1)
    for i in range(0, len(p_mat[0])):
        col_sum = sum(p_mat[:, i])
        p_mat[:, i] *= (1 / col_sum)

    # P[i][j] = exp(P[i][j])
    p_mat = np.vectorize(np.exp)(p_mat)
    # set last row to 1's
    p_mat[k - 1] = np.ones(shape=(1, m))
    # axis 0 -> cols, axis 1 -> rows
    column_sums = p_mat.sum(axis=0)

    # divide each column entry by the column sum, softmax
    for i in range(0, len(column_sums)):
        # max_wx = max(p_mat[])
        normal_term = 1 / column_sums[i]
        p_mat[:, i] *= normal_term

    return p_mat


def get_delta_matrix(class_list, example_classes):
    # total number of examples
    m = len(example_classes)
    # total number of classes
    k = len(class_list)
    delta_mat = np.zeros(shape=(k, m))

    # classes = [c_0, c_1,..., c_k] -> class_to_ind[c_i] = i
    class_to_ind = dict()
    for ind in range(0, k):
        class_to_ind[class_list[ind]] = ind

    # ∆[i][j] = 1 if y_j = c_i else 0
    for ind in range(0, m):
        delta_col = ind
        delta_row = class_to_ind[example_classes[ind]]
        delta_mat[delta_row][delta_col] = 1

    return delta_mat
