import random

import numpy as np


# add column of ones to left of examples
def get_padded_examples(training_examples):
    padding = np.ones(shape=(num_ex, 1))
    padded_mat = np.append(padding, training_examples, axis=1)
    return padded_mat


def get_prob_matrix(samples, weights):
    k = len(weights)
    m = len(samples)
    x_transpose = samples.transpose()

    # WX is k x m matrix
    p_mat = np.matmul(weights, x_transpose)

    # P[i][j] = exp(P[i][j])
    p_mat = np.vectorize(np.exp)(p_mat)

    # set last row to 1's
    p_mat[k - 1] = np.ones(shape=(1, m))

    # axis 0 -> cols, axis 1 -> rows
    column_sums = p_mat.sum(axis=0)

    # divide each column entry by the column sum
    for i in range(0, len(column_sums)):
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
        delta_row = class_to_ind[y[ind]]
        delta_mat[delta_row][delta_col] = 1

    return delta_mat


num_ex = 5
num_att = 3
classes = [2, 5, 7, 4, 3]  # arbitrary set of classes

x = get_padded_examples(np.random.rand(num_ex, num_att))
w = np.random.rand(len(classes), num_att + 1)
y = [2, 4, 3, 7, 3]  # or random.choices(classes, k=num_ex)
delta = get_delta_matrix(class_list=classes, example_classes=y)
prob_mat = get_prob_matrix(samples=x, weights=w)
print(f'sample_classes={y}')
print(f'probabilities\n{prob_mat}')
print(f'delta\n{delta}')
print(f'examples\n{x}')
print(f'weights\n{w}')

penalty_term = 0.01
learning_rate = 0.01
total_iterations = 1_000
iterations = 0
prob_mat = get_prob_matrix(samples=x, weights=w)
while iterations < total_iterations:
    prob_mat = get_prob_matrix(samples=x, weights=w)
    error_mat = np.add(delta, (-1 * prob_mat))
    sample_error_mat = np.matmul(error_mat, x)
    penalty_mat = -penalty_term * w
    weight_update_mat = learning_rate * (np.add(sample_error_mat, penalty_mat))
    w = np.add(w, weight_update_mat)
    iterations += 1

print(f'prob\n{prob_mat}')
for sample_ind in range(0, len(y)):
    argmax_prob = max(prob_mat[:, sample_ind])
    predicted = -1
    for class_ind in range(0, len(prob_mat)):
        if prob_mat[class_ind][sample_ind] == argmax_prob:
            predicted = classes[class_ind]
            break
    print(f'pred_class[x_{sample_ind}]={predicted}, actual_class[x_{sample_ind}]={y[sample_ind]}')
